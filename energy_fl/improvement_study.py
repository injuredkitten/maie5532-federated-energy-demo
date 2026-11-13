"""Validation-only development followed by a frozen, public held-out comparison."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from engine import load_clients,score,unflatten
from .benchmark import execute,row
from .runner import DEFAULTS
from .state import atomic_json


SCHEDULERS=['uniform','resource','cohort_matched_uniform','cohort']
PRIVATE_CANDIDATES={
    'cold_dp005': {'clip':.05,'initial_model':None},
    'warm_dp005': {'clip':.05,'initial_model':'artifacts/public_init.npz'},
    'warm_dp001': {'clip':.01,'initial_model':'artifacts/public_init.npz'},
    'warm_dp0003': {'clip':.003,'initial_model':'artifacts/public_init.npz'},
    'warm_dp0001': {'clip':.001,'initial_model':'artifacts/public_init.npz'},
}


def public_utility(folder,clients,split):
    with np.load(folder/'model_fp32.npz',allow_pickle=False) as artifact:
        return score(clients,unflatten(artifact['weights']),split)


def development(seed=101,confirm=False):
    out=Path('evidence_v3');out.mkdir(exist_ok=True)
    calibration=json.loads(Path('evidence/cpu_calibration.json').read_text(encoding='utf-8'))
    clients,_=load_clients('data30')
    candidates={k:v for k,v in PRIVATE_CANDIDATES.items() if k in ('cold_dp005','warm_dp0001')} if confirm else PRIVATE_CANDIDATES
    design={'phase':'validation_only_confirmation' if confirm else 'validation_only_development','seed':seed,'rounds':30,
            'schedulers':SCHEDULERS,'private_candidates':candidates,
            'sigma':4.,'delta':1e-5,'epsilon_budget':8.,'validation_target':.29,
            'private_metric_scope':'unprotected offline public-data validation; not a deployable DP selection mechanism',
            'test_access':'none; final held-out evaluation uses fresh seeds 201,202,203 after configuration freeze'}
    atomic_json(out/f'development_design_s{seed}.json',design)
    cfg={**DEFAULTS,'data':'data30','workers':1,'seed':seed,'evaluation':'validation'}
    results=[]
    for policy in SCHEDULERS:
        folder=Path(f'benchmark_runs/v3_dev_s{seed}_{policy}')
        summary=execute(folder,{**cfg,'policy':policy},calibration)
        results.append({'method':policy,'validation':summary['history'][-1]['validation'],
                        'target_seconds':summary['target_seconds'],'modeled_seconds':summary['simulated_seconds'],
                        'fairness':summary['fairness']})
        print(results[-1],flush=True)
        atomic_json(out/f'development_results_s{seed}.json',results)
    for name,parameters in candidates.items():
        folder=Path(f'benchmark_runs/v3_dev_s{seed}_{name}')
        summary=execute(folder,{**cfg,'sigma':4.,'policy':'resource',**parameters},calibration)
        results.append({'method':name,'validation':public_utility(folder,clients,'val'),
                        'epsilon':summary['epsilon'],'releases':summary['privacy_releases']})
        print(results[-1],flush=True)
        atomic_json(out/f'development_results_s{seed}.json',results)
    with np.load('artifacts/public_init.npz',allow_pickle=False) as artifact:
        baseline=score(clients,unflatten(artifact['weights']),'val')
    results.append({'method':'public_only','validation':baseline})
    atomic_json(out/f'development_results_s{seed}.json',results)
    print(results[-1],flush=True)


def evaluation():
    out=Path('evidence_v3');out.mkdir(exist_ok=True)
    calibration=json.loads(Path('evidence/cpu_calibration.json').read_text(encoding='utf-8'))
    variants={
        'uniform':{'policy':'uniform'},
        'legacy_resource':{'policy':'resource'},
        'matched_uniform':{'policy':'cohort_matched_uniform'},
        'cohort':{'policy':'cohort'},
    }
    private={
        'cold_dp005':{'policy':'resource','sigma':4.,'clip':.05},
        'cold_dp0001':{'policy':'resource','sigma':4.,'clip':.001},
        'warm_no_dp0001':{'policy':'resource','clip':.001,'initial_model':'artifacts/public_init.npz'},
        'warm_dp0001':{'policy':'resource','sigma':4.,'clip':.001,'initial_model':'artifacts/public_init.npz'},
        'warm_cohort_dp0001':{'policy':'cohort','sigma':4.,'clip':.001,'initial_model':'artifacts/public_init.npz'},
    }
    design={'phase':'frozen_evaluation','seeds':[201,202,203],'rounds':30,'sizes':[10,30],
            'validation_targets':{'10':.26,'30':.29},'schedulers':variants,'privacy_variants':private,
            'chosen_from':'validation-only development seed 101; confirmation seed 102',
            'selection':'cohort grouping fixed before development; C=.001 selected by validation MAE',
            'privacy_comparison':'sigma=4, delta=1e-5, epsilon budget=8; actual releases reported; no sampling amplification',
            'test_policy':'Evaluate after settings freeze; no test-driven changes or seed exclusion',
            'public_model_sha256':json.loads(Path('artifacts/public_init.json').read_text())['model_sha256'],
            'dp_scope':'All data are public benchmark data. Offline validation/test utility is not a DP output.'}
    path=out/'evaluation_design.json'
    if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=design:
        raise ValueError('Evaluation design is frozen; use a new study to change it')
    atomic_json(path,design)
    rows=[];curves=[]
    for n in (10,30):
        data='data' if n==10 else 'data30'
        clients,_=load_clients(data)
        for seed in design['seeds']:
            base={**DEFAULTS,'data':data,'seed':seed,'workers':1,'clients_per_round':5 if n==10 else 10,
                  'validation_target':design['validation_targets'][str(n)]}
            for name,settings in {**variants,**(private if n==30 else {})}.items():
                folder=Path(f'benchmark_runs/v3_eval_n{n}_s{seed}_{name}')
                start=time.perf_counter()
                summary=execute(folder,{**base,**settings},calibration)
                result=row(summary,name,n,seed,time.perf_counter()-start)
                result['minimum_contributions']=summary['fairness']['minimum_contributions']
                result['maximum_contributions']=summary['fairness']['maximum_contributions']
                result['mean_selected']=float(np.mean([r['selected'] for r in summary['history']]))
                result['mean_survivors']=float(np.mean([r['survivors'] for r in summary['history']]))
                elapsed=0.;streak=0;stable_target=None
                for r in summary['history']:
                    elapsed+=r['modeled_seconds']
                    streak=streak+1 if r['validation'] and r['validation']['mae']<=base['validation_target'] else 0
                    if streak>=3 and stable_target is None: stable_target=elapsed
                result['three_round_target_seconds']=stable_target
                if settings.get('sigma'):
                    utility=public_utility(folder,clients,'test')
                    result.update(test_mae=utility['mae'],test_rmse=utility['rmse'],
                                  worst_quintile_mae=utility['worst_quintile_mae'])
                rows.append(result)
                curves.append({'households':n,'seed':seed,'method':name,
                    'rounds':[{k:r[k] for k in ('round','committed','survivors','validation','modeled_seconds','training_wire_bytes')}
                              for r in summary['history']]})
                atomic_json(out/'evaluation_rows.json',rows)
                print(f'{n=} {seed=} {name}: MAE={result["test_mae"]:.6f}, target={result["target_seconds"]}',flush=True)
    clients,_=load_clients('data30')
    with np.load('artifacts/public_init.npz',allow_pickle=False) as artifact:
        utility=score(clients,unflatten(artifact['weights']),'test')
    rows.append({'households':30,'method':'public_only','seed':2026,
                 'test_mae':utility['mae'],'test_rmse':utility['rmse'],
                 'worst_quintile_mae':utility['worst_quintile_mae'],
                 'scope':'one fixed public-only model; no federated training or DP releases'})
    atomic_json(out/'evaluation_rows.json',rows);atomic_json(out/'learning_curves.json',curves)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (out/'evaluation.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)
    print(f'Saved {len(rows)} frozen evaluation rows',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--development-seed',type=int,default=101)
    p.add_argument('--confirm',action='store_true')
    p.add_argument('--evaluate',action='store_true')
    args=p.parse_args()
    if args.evaluate: evaluation()
    else: development(args.development_seed,args.confirm)
