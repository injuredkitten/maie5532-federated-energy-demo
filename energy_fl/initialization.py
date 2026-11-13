"""Auditable initialization from public households disjoint from the federation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from engine import PARAMETERS, flatten, initial_model, load_clients, train, score
from .state import atomic_json


def load_public_model(path, federation_manifest):
    path=Path(path)
    info=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
    if info.get('purpose')!='public_disjoint_pretraining':
        raise ValueError('Initializer must have public pretraining provenance')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=info['model_sha256']:
        raise ValueError('Public initializer checksum mismatch')
    private_ids={c['id'] for c in federation_manifest['clients']}
    public_ids=set(info['household_ids'])
    if not public_ids or private_ids & public_ids:
        raise ValueError('Public pretraining and federated households overlap')
    if info['dataset']!=federation_manifest['dataset']:
        raise ValueError('Household namespaces differ; inspect initializer provenance')
    with np.load(path,allow_pickle=False) as artifact:
        vector=artifact['weights'].copy()
    if vector.shape!=(PARAMETERS,) or not np.isfinite(vector).all():
        raise ValueError('Invalid public model shape/values')
    return vector.astype('f4'),info


def pretrain(data, exclude_manifest, output, epochs=60, seed=2026):
    clients,manifest=load_clients(data)
    excluded=json.loads(Path(exclude_manifest).read_text(encoding='utf-8'))
    if {c.id for c in clients}&{c['id'] for c in excluded['clients']}:
        raise ValueError('Public training households overlap with federation')
    x=np.concatenate([c.splits['train']['x'] for c in clients])
    y=np.concatenate([c.splits['train']['y'] for c in clients])
    w=initial_model(seed);best=[a.copy() for a in w]
    best_val=score(clients,w,'val')['mae'];best_epoch=0
    for epoch in range(epochs):
        w=train(w,x,y,1,32,.03,seed+epoch)
        val=score(clients,w,'val')['mae']
        if val<best_val:
            best,best_val,best_epoch=[a.copy() for a in w],val,epoch+1
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(path,weights=flatten(best))
    info={'purpose':'public_disjoint_pretraining','dataset':manifest['dataset'],
          'household_ids':[c.id for c in clients],
          'public_manifest_sha256':hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
          'source_archive_sha256':manifest['archive_sha256'],
          'seed':seed,'max_epochs':epochs,'selected_epoch':best_epoch,
          'selection_metric':'public-household validation MAE only; no federation or test data',
          'public_validation_mae':best_val,'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'privacy_scope':'No DP claim for these public pretraining households; DP applies only to disjoint federation.'}
    atomic_json(path.with_suffix('.json'),info)
    print(json.dumps(info,indent=2))
    return info


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data_public')
    p.add_argument('--exclude-manifest',default='data30/manifest.json')
    p.add_argument('--output',default='artifacts/public_init.npz')
    p.add_argument('--epochs',type=int,default=60)
    args=p.parse_args();pretrain(args.data,args.exclude_manifest,args.output,args.epochs)
