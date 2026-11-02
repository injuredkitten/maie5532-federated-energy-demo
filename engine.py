"""Small synchronous FL simulator. No sockets, encryption or hardware claims."""
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

SHAPES = [(52, 32), (32,), (32, 24), (24,)]
PARAMETERS = sum(math.prod(s) for s in SHAPES)
PROFILES = [
    {'name': 'Low', 'compute': .5, 'memory_mb': 256, 'mbps': 1., 'online': .60},
    {'name': 'Medium', 'compute': 1., 'memory_mb': 1024, 'mbps': 5., 'online': .80},
    {'name': 'High', 'compute': 2., 'memory_mb': 4096, 'mbps': 20., 'online': .95},
]


def initial_model(seed):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, .1, SHAPES[0]).astype('f4'), np.zeros(32, 'f4'),
            rng.normal(0, .03, SHAPES[2]).astype('f4'), np.zeros(24, 'f4')]


def flatten(weights):
    return np.concatenate([a.ravel() for a in weights]).astype('f4')


def unflatten(vector):
    result, offset = [], 0
    for shape in SHAPES:
        n = math.prod(shape)
        result.append(vector[offset:offset+n].reshape(shape).copy())
        offset += n
    if offset != len(vector):
        raise ValueError('Wrong model size.')
    return result


def predict(weights, x):
    return np.maximum(x @ weights[0] + weights[1], 0) @ weights[2] + weights[3]


def gradients(weights, x, y):
    hidden = np.maximum(x @ weights[0] + weights[1], 0)
    error = hidden @ weights[2] + weights[3] - y
    g = (2. / error.size) * error
    gh = (g @ weights[2].T) * (hidden > 0)
    return [x.T @ gh, gh.sum(axis=0), hidden.T @ g, g.sum(axis=0)]


def train(weights, x, y, epochs, batch, lr, seed):
    w = [a.copy() for a in weights]
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for start in range(0, len(x), batch):
            ids = order[start:start+batch]
            grads = gradients(w, x[ids], y[ids])
            norm = np.linalg.norm(flatten(grads))
            factor = min(1., 10. / max(float(norm), 1e-12))
            w = [(a - lr * factor * b).astype('f4') for a, b in zip(w, grads)]
    return w


@dataclass
class Client:
    id: int
    profile: dict
    mean: float
    scale: float
    splits: dict
    boundaries: tuple


def load_clients(data_dir):
    data_dir = Path(data_dir)
    manifest = json.loads((data_dir / 'manifest.json').read_text(encoding='utf-8'))
    clients = []
    for i, info in enumerate(manifest['clients']):
        path = data_dir / info['file']
        if hashlib.sha256(path.read_bytes()).hexdigest() != info['sha256']:
            raise ValueError(f'Data integrity check failed: {path}')
        with np.load(path, allow_pickle=False) as data:
            series = data['kwh'].copy()
            start = datetime.fromisoformat(str(data['start']))
        days = len(series) // 24
        b1, b2 = int(days * .70) * 24, int(days * .85) * 24
        # Strictly training-only, local scalers never enter aggregation.
        mean = float(series[:b1].mean())
        scale = max(float(series[:b1].std()), .05)
        rows = {'train': [], 'val': [], 'test': []}
        for t in range(48, len(series)-23, 24):
            split = 'train' if t+24 <= b1 else 'val' if t >= b1 and t+24 <= b2 else 'test' if t >= b2 else None
            if split is None:
                continue
            dt = start + timedelta(hours=t)
            dow, doy = dt.weekday(), dt.timetuple().tm_yday
            calendar = [math.sin(2*math.pi*dow/7), math.cos(2*math.pi*dow/7),
                        math.sin(2*math.pi*doy/366), math.cos(2*math.pi*doy/366)]
            x = np.r_[(series[t-48:t]-mean)/scale, calendar].astype('f4')
            y = ((series[t:t+24]-mean)/scale).astype('f4')
            rows[split].append((x, y, series[t-24:t], t, dt.isoformat()))
        splits = {}
        for name, r in rows.items():
            if not r:
                raise ValueError(f'Empty {name} split for {path}')
            splits[name] = {'x': np.stack([v[0] for v in r]), 'y': np.stack([v[1] for v in r]),
                            'persistence': np.stack([v[2] for v in r]),
                            'origins': np.array([v[3] for v in r]), 'dates': [v[4] for v in r]}
        clients.append(Client(info['id'], PROFILES[i % 3], mean, scale, splits, (b1, b2)))
    return clients, manifest


def predictions(weights, client, split):
    return np.maximum(0, predict(weights, client.splits[split]['x']) * client.scale + client.mean)


def score(clients, weights=None, split='test', persistence=False, local=None):
    metrics = []
    for i, client in enumerate(clients):
        d = client.splits[split]
        y = d['y'] * client.scale + client.mean
        pred = d['persistence'] if persistence else predictions(local[i] if local is not None else weights, client, split)
        error = pred-y
        metrics.append({'id': client.id, 'mae': float(np.abs(error).mean()),
                        'rmse': float(np.sqrt(np.mean(error**2)))})
    return {'mae': float(np.mean([r['mae'] for r in metrics])),
            'rmse': float(np.mean([r['rmse'] for r in metrics])),
            'worst_quintile_mae': float(np.mean(sorted(r['mae'] for r in metrics)[-max(1, math.ceil(len(clients)*.2)): ])),
            'households': metrics}


def clip_update(vector, bound):
    return vector * min(1., bound / max(float(np.linalg.norm(vector)), 1e-12))


def epsilon(releases, sigma, delta=1e-5):
    """Gaussian RDP, no subsampling amplification; sigma relative to sensitivity."""
    if releases == 0:
        return 0.
    if sigma <= 0:
        return math.inf
    orders = np.array([1.25, 1.5, 1.75, 2, 3, 4, 5, 8, 10, 16, 32, 64, 128, 256])
    return float(np.min(releases*orders/(2*sigma*sigma) + np.log(1/delta)/(orders-1)))


def aggregate(base, updates, versions, version, minimum=2, clip=None, sigma=0., rng=None):
    if any(v != version for v in versions):
        raise ValueError('Stale model update rejected.')
    if len(updates) != len(versions):
        raise ValueError('Missing update version.')
    if len(updates) < minimum:
        return base.copy(), False
    if sigma > 0 and (clip is None or clip <= 0):
        raise ValueError('Gaussian mode requires positive client-update clipping.')
    u = [clip_update(v, clip) if clip is not None else v for v in updates]
    delta = np.mean(u, axis=0)  # Equal-household weighting, as in proposal.
    if sigma > 0:
        # Replacement adjacency: one bounded update can change by at most 2C.
        delta = delta + rng.normal(0, sigma*2*clip/len(u), delta.shape)
    return (base + delta).astype('f4'), True


def resource_trace(clients, rounds, seed):
    rng = np.random.default_rng(seed+10000)
    n = len(clients)
    return {'online': rng.random((rounds, n)) < np.array([c.profile['online'] for c in clients]),
            'drop': rng.random((rounds, n)) < .08,
            'jitter': rng.uniform(.8, 1.2, (rounds, n))}


def select_clients(online, counts, costs, k, strategy, rng):
    candidates = np.flatnonzero(online).tolist()
    if strategy == 'uniform':
        return rng.permutation(candidates)[:k].tolist()
    if not candidates:
        return []
    # Reserve one slot for least-served available device, then minimize estimated cost.
    first = min(candidates, key=lambda j: (counts[j], costs[j], j))
    others = sorted((j for j in candidates if j != first), key=lambda j: (costs[j], counts[j], j))
    return [first] + others[:max(0, k-1)]


def run_fl(clients, config, strategy, output, callback=None, private=False):
    rounds, seed = config['rounds'], config['seed']
    trace = resource_trace(clients, rounds, seed)
    weights = initial_model(seed)
    rng = np.random.default_rng(seed+200)
    noise_rng = np.random.default_rng(seed+500)  # Demonstration only, public reproducibility seed.
    counts = np.zeros(len(clients), dtype=int)
    history, version, releases, simulated, payload = [], 0, 0, 0., 0
    best, best_mae, best_round = weights, math.inf, 0
    stopped = False
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start_wall = time.perf_counter()
    for r in range(rounds):
        if private and epsilon(releases+1, config['sigma']) > config['epsilon_budget']:
            stopped = True
            break
        start_round = time.perf_counter()
        # Explicit synthetic reference: 0.30 seconds per 100 training examples per epoch.
        epochs = [1 if strategy == 'resource' and c.profile['name'] == 'Low' else config['epochs'] for c in clients]
        costs = [(.30 * len(c.splits['train']['x']) / 100 * epochs[j] / c.profile['compute']
                  + 2*PARAMETERS*4*8/(c.profile['mbps']*1e6)) * trace['jitter'][r, j]
                 for j, c in enumerate(clients)]
        selected = select_clients(trace['online'][r], counts, costs, config['clients_per_round'], strategy, rng)
        survivors = [j for j in selected if not trace['drop'][r,j] and costs[j] <= config['deadline']]
        base, updates = flatten(weights), []
        for j in survivors:
            c = clients[j]
            d = c.splits['train']
            local = train(weights, d['x'], d['y'], epochs[j], 16 if c.profile['name'] == 'Low' else 32,
                          config['lr'], seed+r*1000+j)
            updates.append(flatten(local)-base)
        next_flat, success = aggregate(base, updates, [version]*len(updates), version,
                                      clip=config['clip'] if private else None,
                                      sigma=config['sigma'] if private else 0, rng=noise_rng)
        if success:
            weights = unflatten(next_flat)
            version += 1
            releases += int(private)
            counts[survivors] += 1
        duration = max((config['deadline'] if trace['drop'][r,j] else min(costs[j], config['deadline']) for j in selected), default=0.)
        simulated += duration
        # Model tensors only; no cryptographic/protocol/TLS overhead is claimed.
        payload += (len(selected)+len(survivors))*PARAMETERS*4
        val = score(clients, weights, 'val')['mae']
        if val < best_mae:
            best, best_mae, best_round = [a.copy() for a in weights], val, r+1
        row = {'round': r+1, 'status': 'aggregated' if success else 'aborted', 'version': version,
               'selected': [clients[j].id for j in selected], 'survivors': [clients[j].id for j in survivors],
               'validation_mae': val, 'simulated_seconds': simulated, 'round_simulated_seconds': duration,
               'measured_wall_seconds': time.perf_counter()-start_round, 'payload_bytes': payload,
               'coverage': float(np.mean(counts > 0)), 'counts': counts.tolist(),
               'epsilon_demo': epsilon(releases, config['sigma']) if private else None}
        history.append(row)
        np.savez(output / 'latest_checkpoint.npz', **{f'w{i}': a for i,a in enumerate(weights)}, version=version, round=r+1)
        if callback:
            callback(row)
        print(f'{strategy}{"+noise" if private else "":8s} round {r+1:02d} val MAE={val:.4f} survivors={len(survivors)} {row["status"]}', flush=True)
    # Private demonstration uses the final release, not data-dependent validation selection.
    chosen = weights if private else best
    np.savez(output / 'model_fp32.npz', **{f'w{i}': a for i,a in enumerate(chosen)})
    return chosen, {'test': score(clients, chosen), 'history': history,
                    'selected_round': history[-1]['round'] if private and history else best_round,
                    'model_selection': 'final release' if private else 'minimum validation MAE',
                    'wall_seconds': time.perf_counter()-start_wall, 'counts': counts.tolist(),
                    'privacy_budget_stopped': stopped, 'releases': releases,
                    'epsilon_demo': epsilon(releases, config['sigma']) if private else None}


def baseline_models(clients, config):
    """Same maximum epochs for local and centralized; not equal-total-FLOP comparisons."""
    seed = config['seed']
    init = initial_model(seed)
    local = []
    epochs = config['rounds']*config['epochs']
    for c in clients:
        w, best, best_v = init, init, math.inf
        d = c.splits['train']
        for epoch in range(epochs):
            w = train(w, d['x'], d['y'], 1, 32, config['lr'], seed+epoch)
            v = score([c], w, 'val')['mae']
            if v < best_v:
                best, best_v = [a.copy() for a in w], v
        local.append(best)
    x = np.concatenate([c.splits['train']['x'] for c in clients])
    y = np.concatenate([c.splits['train']['y'] for c in clients])
    w, best, best_v = init, init, math.inf
    for epoch in range(epochs):
        w = train(w, x, y, 1, 32, config['lr'], seed+epoch)
        v = score(clients, w, 'val')['mae']
        if v < best_v:
            best, best_v = [a.copy() for a in w], v
    return {'persistence': score(clients, persistence=True), 'local_only': score(clients, local=local),
            'centralized': score(clients, best)}


def export_quantized(weights, clients, output):
    """INT8 weight STORAGE, decoded for FP32 compute; not an integer inference runtime."""
    output = Path(output)
    arrays, reconstructed = {}, []
    for i, a in enumerate(weights):
        scale = np.float32(max(float(np.abs(a).max())/127., 1e-10))
        q = np.clip(np.rint(a/scale), -127, 127).astype('i1')
        arrays[f'q{i}'], arrays[f's{i}'] = q, scale
        reconstructed.append(q.astype('f4')*scale)
    path = output / 'model_int8_storage.npz'
    np.savez(path, **arrays)
    # Verify serialization round trip using the actual saved artifact.
    with np.load(path, allow_pickle=False) as f:
        restored = [f[f'q{i}'].astype('f4')*f[f's{i}'] for i in range(4)]
    latency = {}
    sample = clients[0].splits['test']['x'][:1]
    for label, model in [('fp32', weights), ('int8_decoded_fp32', restored)]:
        for _ in range(30):
            predict(model, sample)
        times = []
        for _ in range(300):
            start = time.perf_counter_ns()
            predict(model, sample)
            times.append((time.perf_counter_ns()-start)/1e6)
        latency[label] = {'median_ms': float(np.median(times)), 'p95_ms': float(np.percentile(times,95))}
    return {'fp32_bytes': (output / 'model_fp32.npz').stat().st_size, 'int8_storage_bytes': path.stat().st_size,
            'fp32_test': score(clients, weights), 'int8_storage_test': score(clients, restored),
            'latency': latency, 'compute': 'Both execute FP32 NumPy; INT8 is weight storage only.'}
