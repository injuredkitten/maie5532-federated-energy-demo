"""Paired public-data experiments. Never use this evaluator on private deployments."""
import argparse
from contextlib import redirect_stdout
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from engine import load_clients, baseline_models, score, unflatten
from .runner import run, DEFAULTS
from .state import atomic_json, RunStore, fingerprint

SEEDS=[42,43,44]
VARIANTS={
    'plain_uniform': {'secure':False,'policy':'uniform'},
    'secure_uniform': {'secure':True,'policy':'uniform'},
    'secure_resource': {'secure':True,'policy':'resource'},
    'secure_adaptive': {'secure':True,'policy':'resource','adaptive_epochs':True,'adaptive_batch':True},
}


def execute(folder, cfg, calibration):
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists():
        saved=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        if saved['status'] in ('completed','privacy_budget_exhausted'):
            manifest=json.loads((Path(cfg['data'])/'manifest.json').read_text(encoding='utf-8'))
            for item in manifest['clients']:
                if hashlib.sha256((Path(cfg['data'])/item['file']).read_bytes()).hexdigest()!=item['sha256']:
                    raise ValueError('Cached run dataset checksum mismatch')
            public_config={k:v for k,v in {**DEFAULTS,**cfg}.items() if k not in ('data','initial_model')}
            public_config['initial_model_sha256']=None
            if cfg.get('initial_model'):
                from .initialization import load_public_model
                _,info=load_public_model(cfg['initial_model'],manifest)
                public_config['initial_model_sha256']=info['model_sha256']
            expected=fingerprint({**public_config,'calibration':calibration},manifest)
            RunStore(folder).load(expected)  # Never relabel cached runs with changed settings.
            return saved
    with (folder/'console.log').open('a',encoding='utf-8') as log, redirect_stdout(log):
        return run(cfg,folder,resume=(folder/'checkpoint.npz').exists(),calibration=calibration)


def row(summary, name, n, seed, wall):
    history=summary['history']
    return {'households':n,'seed':seed,'method':name,'rounds':summary['round'],
            'committed_rounds':summary['version'],
            'test_mae':summary.get('test',{}).get('mae'),
            'test_rmse':summary.get('test',{}).get('rmse'),
            'worst_quintile_mae':summary.get('test',{}).get('worst_quintile_mae'),
            'target_seconds':summary['target_seconds'],
            'modeled_training_seconds':summary['simulated_seconds'],
            'measured_training_seconds':sum(r['wall_seconds'] or 0 for r in history) if not summary['private'] else None,
            'script_seconds':wall,
            'training_wire_bytes':sum(r['training_wire_bytes'] for r in history),
            'evaluation_wire_bytes':sum(r['evaluation_wire_bytes'] for r in history)+summary.get('final_test_wire_bytes',0),
            'coverage':summary['fairness']['coverage'],'jain_index':summary['fairness']['jain_index'],
            'epsilon':summary['epsilon'],'privacy_releases':summary['privacy_releases'],
            'scope':'public benchmark; DP model accounting excludes post-hoc utility metrics'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='benchmark_runs/v2')
    p.add_argument('--evidence',default='evidence')
    p.add_argument('--calibration',default='evidence/cpu_calibration.json')
    p.add_argument('--rounds',type=int,default=30)
    p.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    p.add_argument('--skip-baselines',action='store_true')
    args=p.parse_args(); root=Path(args.output); evidence=Path(args.evidence)
    evidence.mkdir(parents=True,exist_ok=True)
    calibration=json.loads(Path(args.calibration).read_text(encoding='utf-8'))
    # Freeze the design before any test-set evaluation. Thresholds chosen from
    # early validation-only pilot runs; same threshold for every method/seed.
    design={'seeds':args.seeds,'rounds':args.rounds,'sizes':[10,30],
            'validation_targets':{'10':.26,'30':.29},'variants':VARIANTS,
            'dp_sigma':[2.,4.,8.],'dp_clip':.05,'epsilon_budget':8.,'delta':1e-5,
            'comparison':'paired public traces; final-round FL model; baselines select validation-best epoch',
            'limitations':'3 seeds are exploratory; edge time is modeled, not a hardware/network benchmark'}
    atomic_json(evidence/'experiment_design.json',design)
    rows=[]; curves=[]
    for n in (10,30):
        data='data' if n==10 else 'data30'
        clients,_=load_clients(data)
        for seed in args.seeds:
            cfg={**DEFAULTS,'data':data,'rounds':args.rounds,'seed':seed,'workers':1,
                 'clients_per_round':5 if n==10 else 10,'validation_target':design['validation_targets'][str(n)]}
            for name,settings in VARIANTS.items():
                start=time.perf_counter()
                summary=execute(root/f'n{n}_s{seed}_{name}',{**cfg,**settings},calibration)
                rows.append(row(summary,name,n,seed,time.perf_counter()-start))
                curves.append({'households':n,'seed':seed,'method':name,
                    'rounds':[{k:r[k] for k in ('round','committed','survivors','validation','modeled_seconds','training_wire_bytes')}
                              for r in summary['history']]})
                print(f'finished n={n} seed={seed} {name} MAE={rows[-1]["test_mae"]:.6f}',flush=True)
                atomic_json(evidence/'benchmark_rows.json',rows)
            if n==30:
                for sigma in (0.,2.,4.,8.):
                    name='secure_clip005' if sigma==0 else f'private_sigma{int(sigma)}'
                    start=time.perf_counter()
                    folder=root/f'n{n}_s{seed}_{name}'
                    summary=execute(folder,{**cfg,'clip':.05,'sigma':sigma},calibration)
                    result=row(summary,name,n,seed,time.perf_counter()-start)
                    if sigma:
                        # Separate post-hoc public-data analysis: this output is not DP.
                        with np.load(folder/'model_fp32.npz',allow_pickle=False) as artifact:
                            utility=score(clients,unflatten(artifact['weights']))
                        result.update(test_mae=utility['mae'],test_rmse=utility['rmse'],
                                      worst_quintile_mae=utility['worst_quintile_mae'])
                    rows.append(result)
                    print(f'finished n={n} seed={seed} {name} MAE={result["test_mae"]:.6f}',flush=True)
                    atomic_json(evidence/'benchmark_rows.json',rows)
            if not args.skip_baselines:
                baseline_file=root/f'n{n}_s{seed}_baselines.json'
                if baseline_file.exists(): baselines=json.loads(baseline_file.read_text(encoding='utf-8'))
                else:
                    baselines=baseline_models(clients,cfg);atomic_json(baseline_file,baselines)
                for name,metrics in baselines.items():
                    rows.append({'households':n,'seed':seed,'method':name,'test_mae':metrics['mae'],
                                 'test_rmse':metrics['rmse'],'worst_quintile_mae':metrics['worst_quintile_mae'],
                                 'scope':'public reference; centralized/local up to 60 epochs; not FLOP matched'})
                print(f'finished n={n} seed={seed} reference baselines',flush=True)
    atomic_json(evidence/'benchmark_rows.json',rows);atomic_json(evidence/'learning_curves.json',curves)
    fields=list(dict.fromkeys(key for r in rows for key in r))
    with (evidence/'benchmark.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fields);writer.writeheader();writer.writerows(rows)
    print(f'Saved {len(rows)} rows to {evidence}',flush=True)


if __name__=='__main__': main()
