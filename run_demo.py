"""Run real NumPy training and export a self-contained, offline result dashboard."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import argparse
import csv
import json
from pathlib import Path
import platform
import sys
import time
import webbrowser
import numpy as np
from engine import load_clients, baseline_models, run_fl, export_quantized, predictions, PARAMETERS

ROOT = Path(__file__).resolve().parent


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, default=ROOT/'data')
    p.add_argument('--output', type=Path, default=ROOT/'results')
    p.add_argument('--rounds', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--epochs', type=int, default=2)
    p.add_argument('--clients-per-round', type=int, default=5)
    p.add_argument('--lr', type=float, default=.03)
    p.add_argument('--deadline', type=float, default=5.)
    p.add_argument('--privacy-demo', action='store_true')
    p.add_argument('--sigma', type=float, default=2.)
    p.add_argument('--clip', type=float, default=.05)
    p.add_argument('--epsilon-budget', type=float, default=10.)
    p.add_argument('--open', action='store_true')
    a = p.parse_args()
    if not 1 <= a.rounds <= 300 or a.epochs < 1 or a.clients_per_round < 2 or min(a.lr,a.deadline,a.sigma,a.clip,a.epsilon_budget) <= 0:
        p.error('Use 1..300 rounds, positive numeric settings and at least two selected clients.')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(a).items() if k not in ('data','output','open')}
    clients, manifest = load_clients(a.data)
    config['clients_per_round'] = min(a.clients_per_round, len(clients))
    start = time.perf_counter()
    print(f'Real Ausgrid data: {len(clients)} homes; {PARAMETERS} MLP parameters. Training baselines...', flush=True)
    baseline = baseline_models(clients, config)
    experiments, models = {}, {}
    for name in ['uniform', 'resource'] + (['privacy_demo'] if a.privacy_demo else []):
        def callback(row):
            write_json(out/'progress.json', {'experiment': name, 'latest': row})
        models[name], experiments[name] = run_fl(clients, config, 'resource' if name != 'uniform' else 'uniform',
                                                 out/name, callback, private=name == 'privacy_demo')
    compression = export_quantized(models['resource'], clients, out/'resource')
    forecast = []
    for c in clients:
        d = c.splits['test']
        forecast.append({'id': c.id, 'profile': c.profile['name'], 'date': d['dates'][0],
                         'actual': (d['y'][0]*c.scale+c.mean).tolist(), 'persistence': d['persistence'][0].tolist(),
                         'uniform': predictions(models['uniform'],c,'test')[0].tolist(),
                         'resource': predictions(models['resource'],c,'test')[0].tolist()})
    result = {'config': config, 'manifest': manifest, 'baselines': baseline, 'experiments': experiments,
              'compression': compression, 'forecasts': forecast, 'parameters': PARAMETERS,
              'total_wall_seconds': time.perf_counter()-start,
              'environment': {'python': sys.version.split()[0], 'numpy': np.__version__, 'platform': platform.platform()},
              'splits': [{'id': c.id, 'train':len(c.splits['train']['x']), 'val':len(c.splits['val']['x']),
                          'test':len(c.splits['test']['x']), 'boundaries_hours': c.boundaries,
                          'train_target_end_exclusive': c.splits['val']['dates'][0],
                          'test_start': c.splits['test']['dates'][0]} for c in clients],
              'limitations': ['Single-process simulation; no network FL, TLS or SecAgg+.',
                             'Only public GC load from selected Australian solar homes; not total household demand or representative population.',
                             'Simulation seconds and communication tensor bytes are estimates, not measured device time or protocol bytes.',
                             'INT8 weights are decoded to FP32 for inference; no integer-kernel speedup is claimed.',
                             'Optional Gaussian/RDP mode is educational: public seeded noise, raw evaluation outputs and local data are not a protected release.',
                             'One seed and one split; no statistical or generalization claim. Checkpoints are weight snapshots, not resumable orchestration.']}
    write_json(out/'results.json', result)
    with (out/'metrics.csv').open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['method','test_macro_MAE_kWh','test_macro_RMSE_kWh','worst_quintile_MAE_kWh'])
        for name, s in list(baseline.items())+[(k,v['test']) for k,v in experiments.items()]:
            w.writerow([name,s['mae'],s['rmse'],s['worst_quintile_mae']])
    for name, exp in experiments.items():
        with (out/f'{name}_rounds.csv').open('w',newline='',encoding='utf-8-sig') as f:
            if exp['history']:
                w = csv.DictWriter(f, fieldnames=exp['history'][0].keys())
                w.writeheader()
                w.writerows(exp['history'])
    template = (ROOT/'dashboard_template.html').read_text(encoding='utf-8')
    embedded = json.dumps(result,ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    (out/'dashboard.html').write_text(template.replace('__RESULT_JSON__',embedded),encoding='utf-8')
    print('\nTEST METRICS (kWh/hour; equal-household macro average)')
    for name, s in list(baseline.items())+[(k,v['test']) for k,v in experiments.items()]:
        print(f'{name:18s} MAE={s["mae"]:.6f} RMSE={s["rmse"]:.6f}')
    print(f'Completed in {result["total_wall_seconds"]:.2f}s. Open {out / "dashboard.html"}')
    if a.open:
        webbrowser.open((out/'dashboard.html').as_uri())


if __name__ == '__main__':
    main()
