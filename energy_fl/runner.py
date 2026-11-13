"""Round controller: private updates, transactional checkpoints, live public status."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from engine import initial_model, flatten, PROFILES
from .privacy import spent, privatize
from .state import RunStore, atomic_json, fingerprint
from .scheduler import plan, fairness, epoch_cost


DEFAULTS = dict(rounds=30, clients_per_round=10, epochs=2, batch=32, lr=.03,
                seed=42, policy='resource', adaptive_epochs=False, adaptive_batch=False, dropout=.08,
                deadline=1., clip=.5, sigma=0., delta=1e-5, epsilon_budget=8.,
                threshold=.67, secure=True, backend='numpy', workers=2,
                validation_target=.29)


@contextmanager
def exclusive_run(folder):
    path = Path(folder); path.mkdir(parents=True, exist_ok=True)
    with (path/'run.lock').open('a+b') as lock:
        lock.seek(0); lock.write(b'0'); lock.flush(); lock.seek(0)
        if os.name == 'nt':
            import msvcrt
            try: msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc: raise RuntimeError('Another controller owns this output directory') from exc
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try: yield
        finally:
            lock.seek(0)
            if os.name == 'nt': msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(lock, fcntl.LOCK_UN)


def validate(config):
    for key in ('rounds', 'clients_per_round', 'epochs', 'batch', 'workers'):
        if not isinstance(config[key], int) or config[key] < 1: raise ValueError(f'Invalid {key}')
    if config['clients_per_round'] < 3 or config['clients_per_round'] > 100:
        raise ValueError('Use 3..100 clients per round')
    for key in ('lr', 'deadline', 'clip', 'epsilon_budget'):
        if not np.isfinite(config[key]) or config[key] <= 0: raise ValueError(f'Invalid {key}')
    if not 0 <= config['dropout'] <= 1 or not .5 < config['threshold'] <= 1:
        raise ValueError('Invalid dropout/threshold')
    if not np.isfinite(config['sigma']) or config['sigma'] < 0 or not 0 < config['delta'] < 1:
        raise ValueError('Invalid privacy parameters')
    if config['sigma'] > 0 and not config['secure']:
        raise ValueError('This implementation requires SecAgg+ for private mode')
    if config['backend'] == 'cuda' and config['workers'] != 1:
        raise ValueError('Use --workers 1 for CUDA to bound GPU memory')


def run(config, output, resume=False, stop_after=None, calibration=None):
    config = {**DEFAULTS, **config}; validate(config)
    config['data'] = str(Path(config['data']).resolve())
    with exclusive_run(output):
        return _run(config, Path(output), resume, stop_after, calibration)


def _run(config, output, resume, stop_after, calibration):
    from .transport import ProcessGrid
    from .protocol import aggregate_round, evaluate
    logging.getLogger('flwr').setLevel(logging.ERROR)
    store = RunStore(output)
    manifest = json.loads((Path(config['data'])/'manifest.json').read_text(encoding='utf-8'))
    ids = [c['id'] for c in manifest['clients']]
    if len(ids) < 3: raise ValueError('At least three households are required')
    # Verify every input, including clients not selected in the first round.
    for item in manifest['clients']:
        if hashlib.sha256((Path(config['data'])/item['file']).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Dataset checksum mismatch')
    if store.path.exists() and not resume:
        raise ValueError('Output already contains a run; use --resume or a new output path')
    if resume and not store.path.exists(): raise ValueError('No checkpoint to resume')
    calibration_file = output/'calibration.json'
    if calibration is None:
        if resume:
            calibration = json.loads(calibration_file.read_text(encoding='utf-8'))
        else:
            subprocess.run([sys.executable, '-m', 'energy_fl.calibrate', '--output', str(calibration_file)],
                           check=True, stdout=subprocess.DEVNULL)
            calibration = json.loads(calibration_file.read_text(encoding='utf-8'))
    if calibration['backend'] != 'numpy':
        raise ValueError('Scheduling uses an isolated NumPy CPU reference, even for GPU execution')
    public_config = {k: v for k, v in config.items() if k != 'data'}
    run_hash = fingerprint({**public_config, 'calibration': calibration}, manifest)
    if resume:
        weights, state = store.load(run_hash)
    else:
        weights = flatten(initial_model(config['seed']))
        state = {'fingerprint': run_hash, 'round': 0, 'version': 0, 'privacy_releases': 0,
                 'counts': [0]*len(ids), 'last_seen': [0]*len(ids), 'history': [],
                 'simulated_seconds': 0., 'wire_bytes': 0, 'target_seconds': None}
        atomic_json(calibration_file, calibration)
        store.save(weights, state)
    atomic_json(output/'config.json', {**public_config, 'households': len(ids),
                'data_manifest_sha256': hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()})

    def publish(status, extra=None):
        summary = {'status': status, 'round': state['round'], 'rounds': config['rounds'],
                   'version': state['version'], 'households': len(ids), 'backend': config['backend'],
                   'policy': config['policy'], 'secure': config['secure'], 'private': config['sigma'] > 0,
                   'epsilon': spent(state['privacy_releases'], config['sigma'], config['delta']) if config['sigma'] else None,
                   'delta': config['delta'] if config['sigma'] else None,
                   'privacy_releases': state['privacy_releases'], 'epsilon_budget': config['epsilon_budget'],
                   'simulated_seconds': state['simulated_seconds'], 'wire_bytes': state['wire_bytes'],
                   'target_seconds': state['target_seconds'], 'fairness': fairness(state['counts']),
                   'history': state['history'], 'updated_unix': time.time(),
                   'scope': 'Same-host Flower process simulation; edge timing is modeled; metrics use public Ausgrid data.'}
        if extra: summary.update(extra)
        atomic_json(output/'progress.json', summary)
        return summary

    publish('initializing')
    end = min(config['rounds'], stop_after) if stop_after else config['rounds']
    status = 'completed'
    try:
        with ProcessGrid(config, ids) as grid:
            for number in range(state['round']+1, end+1):
                if config['sigma'] and spent(state['privacy_releases']+1, config['sigma'], config['delta']) > config['epsilon_budget']:
                    status = 'privacy_budget_exhausted'; break
                schedule = plan(number, ids, state['counts'], state['last_seen'], config, calibration)
                publish('training', {'active_round': number, 'selected_count': len(schedule['selected'])})
                grid.failed = {i+1 for i in schedule['failed']}; grid.stats = []
                start = time.perf_counter()
                average, survivors = (None, [])
                if len(schedule['selected']) >= 3:
                    average, survivors = aggregate_round(grid, weights, schedule['selected'], schedule['epochs'],
                                                         number, state['version'], config)
                training_stats = list(grid.stats)
                committed = average is not None
                if committed:
                    if config['sigma']:
                        # Reserve BEFORE constructing/releasing a noisy model. An interrupted
                        # reservation is conservatively charged on resume, never reused.
                        count = state['privacy_releases']+1
                        store.reserve(count, run_hash)
                        average, _ = privatize(average, len(survivors), config['clip'], config['sigma'])
                        state['privacy_releases'] = count
                    weights = (weights.astype('f8')+average).astype('f4')
                    state['version'] += 1
                    for i in survivors:
                        state['counts'][i] += 1; state['last_seen'][i] = number
                elapsed = time.perf_counter()-start
                # Measured protobuf bytes, modeled profile bandwidth, sequential protocol stages.
                network = sum(max(((p['down']+p['up'])*8/(PROFILES[(int(nid)-1)%3]['mbps']*1e6)
                                    for nid, p in s['traffic'].items()), default=0.) for s in training_stats)
                compute = max((epoch_cost(calibration,config,i)*schedule['epochs'][i]/PROFILES[i%3]['compute']
                               for i in schedule['selected'] if i not in schedule['failed']), default=0.)
                modeled = network + (max(compute, config['deadline']) if schedule['failed'] else compute)
                state['simulated_seconds'] += modeled
                record = {'round': number, 'committed': committed, 'version': state['version'],
                          'selected': len(schedule['selected']), 'survivors': len(survivors),
                          'simulated_dropouts': len(schedule['failed']), 'modeled_seconds': modeled,
                          'wall_seconds': elapsed if not config['sigma'] else None,
                          'training_wire_bytes': sum(s['up_bytes']+s['down_bytes'] for s in training_stats),
                          'stages': training_stats, 'validation': None}
                if not config['sigma']:
                    record['validation'] = evaluate(grid, weights, 'val')
                    if state['target_seconds'] is None and record['validation']['mae'] <= config['validation_target']:
                        state['target_seconds'] = state['simulated_seconds']
                record['evaluation_wire_bytes'] = sum(s['up_bytes']+s['down_bytes'] for s in grid.stats[len(training_stats):])
                state['wire_bytes'] += record['training_wire_bytes']+record['evaluation_wire_bytes']
                state['round'] = number; state['history'].append(record)
                store.save(weights, state)
                publish('running')
                print(f"round={number} version={state['version']} survivors={len(survivors)} committed={committed}", flush=True)
            extra = {}
            if not config['sigma']:
                grid.stats = []
                extra['test'] = evaluate(grid, weights, 'test')
                # This diagnostic is not part of the round state or time-to-target.
                extra['final_test_wire_bytes'] = sum(s['up_bytes']+s['down_bytes'] for s in grid.stats)
            if end < config['rounds'] and status == 'completed': status = 'paused_at_boundary'
            store.save_model(weights)
            summary = publish(status, extra)
            atomic_json(output/'summary.json', summary)
            return summary
    except BaseException:
        # Show only the last committed state, never an uncheckpointed release.
        weights, state = store.load(run_hash)
        publish('interrupted_resume_available')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data30')
    parser.add_argument('--output', default='system_runs/demo')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--calibration', help='Shared public CPU calibration JSON for paired experiments')
    for key in ('rounds','clients_per_round','epochs','batch','seed','workers'):
        parser.add_argument('--'+key.replace('_','-'), type=int, default=DEFAULTS[key])
    for key in ('lr','dropout','deadline','clip','sigma','delta','epsilon_budget','threshold','validation_target'):
        parser.add_argument('--'+key.replace('_','-'), type=float, default=DEFAULTS[key])
    parser.add_argument('--policy', choices=['uniform','resource'], default='resource')
    parser.add_argument('--backend', choices=['numpy','torch-cpu','cuda'], default='numpy')
    parser.add_argument('--adaptive-epochs', action='store_true')
    parser.add_argument('--adaptive-batch', action='store_true')
    parser.add_argument('--plain', action='store_true', help='Disable SecAgg+ for public-data ablation only')
    args = vars(parser.parse_args())
    output, resume, stop = args.pop('output'), args.pop('resume'), args.pop('stop_after')
    calibration = args.pop('calibration')
    args['secure'] = not args.pop('plain')
    result = run(args, output, resume, stop,
                 json.loads(Path(calibration).read_text(encoding='utf-8')) if calibration else None)
    print(json.dumps({k:v for k,v in result.items() if k != 'history'}, indent=2))


if __name__ == '__main__': main()
