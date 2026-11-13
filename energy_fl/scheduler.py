"""Scheduling based on public resource traces, with fair participation slots."""
import numpy as np
from engine import PROFILES, PARAMETERS


def trace(round_number, n, seed, dropout):
    rng=np.random.default_rng(np.random.SeedSequence([seed,round_number,991]))
    online=rng.random(n)<np.array([PROFILES[i%3]['online'] for i in range(n)])
    drops=rng.random(n)<dropout
    jitter=rng.uniform(.8,1.2,n)
    return online,drops,jitter


def plan(round_number, ids, counts, last_seen, config, calibration):
    n=len(ids);online,drops,jitter=trace(round_number,n,config['seed'],config['dropout'])
    epochs=np.full(n,config['epochs'],dtype=int)
    if config['adaptive_epochs']:
        epochs[np.arange(n)%3==0]=1
    # Runtime memory from a public calibration; no household contents or measured
    # private-data training duration enters online selection.
    memory=max(calibration['worker_rss_mb'],calibration['tensor_working_set_mb'])*1.15
    feasible=np.array([memory<=PROFILES[i%3]['memory_mb'] for i in range(n)])
    costs=np.array([(epoch_cost(calibration,config,i)*epochs[i]/PROFILES[i%3]['compute']
        +2*PARAMETERS*4*8/(PROFILES[i%3]['mbps']*1e6))*jitter[i] for i in range(n)])
    available=np.flatnonzero(online & feasible).tolist()
    k=min(config['clients_per_round'],len(available))
    rng=np.random.default_rng(np.random.SeedSequence([config['seed'],round_number,772]))
    if config['policy'] in ('cohort','cohort_matched_uniform'):
        # Rotate high/medium/low cohorts. A slow link then delays its own cohort,
        # rather than every round. Round/seed and public device metadata only.
        tier=2-((round_number-1)%3)
        cohort=sorted((i for i in available if i%3==tier),
                      key=lambda i:(counts[i],last_seen[i],i))[:k]
        # At least three clients for SecAgg+. Fill only when the tier is sparse.
        if len(cohort)<min(3,k):
            rest=sorted((i for i in available if i not in cohort),
                        key=lambda i:(abs(i%3-tier),counts[i],costs[i],i))
            cohort += rest[:min(3,k)-len(cohort)]
        selected=cohort if config['policy']=='cohort' else rng.permutation(available)[:len(cohort)].tolist()
    elif config['policy']=='uniform':
        selected=rng.permutation(available)[:k].tolist()
    else:
        # Fair slots: prioritize longest since success, then least successful.
        fair=sorted(available,key=lambda i:(last_seen[i],counts[i],i))[:max(1,k//3)] if k else []
        rest=sorted((i for i in available if i not in fair),key=lambda i:(costs[i],counts[i],i))
        selected=fair+rest[:k-len(fair)]
    late={i for i in selected if costs[i]>config['deadline']}
    failed={i for i in selected if drops[i]}|late
    simulated=max((config['deadline'] if i in failed else costs[i] for i in selected),default=0.)
    return {'selected':selected,'failed':failed,'epochs':epochs.tolist(),'costs':costs.tolist(),
            'online':online.tolist(),'feasible':feasible.tolist(),'simulated_seconds':float(simulated),
            'memory_estimate_mb':memory}


def epoch_cost(calibration, config, index):
    batch=16 if config.get('adaptive_batch') and index%3==0 else config['batch']
    return calibration.get('epoch_seconds_by_batch',{}).get(str(batch),calibration['epoch_seconds'])


def fairness(counts):
    x=np.array(counts,dtype=float)
    return {'coverage':float(np.mean(x>0)), 'jain_index':float(x.sum()**2/(len(x)*np.sum(x*x))) if np.any(x) else 0.,
            'minimum_contributions':int(x.min()),'maximum_contributions':int(x.max())}
