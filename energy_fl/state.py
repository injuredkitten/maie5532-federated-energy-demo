"""Atomic round-boundary checkpoints and pre-release privacy reservation journal."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
        f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def fingerprint(config, manifest):
    encoded=json.dumps({'config':config,'manifest':manifest},sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(encoded).hexdigest()


class RunStore:
    def __init__(self, folder):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.path=self.folder/'checkpoint.npz'
        self.journal=self.folder/'privacy_reservation.json'

    def save(self, weights, state):
        temp=self.path.with_suffix('.tmp')
        with temp.open('wb') as f:
            np.savez(f,weights=weights,state=np.array(json.dumps(state,allow_nan=False)))
            f.flush();os.fsync(f.fileno())
        os.replace(temp,self.path)

    def load(self, expected_fingerprint):
        with np.load(self.path,allow_pickle=False) as f:
            weights=f['weights'].copy();state=json.loads(str(f['state']))
        if state['fingerprint']!=expected_fingerprint:
            raise ValueError('Checkpoint config/data differs; use the same settings or a new output directory.')
        if self.journal.exists():
            reservation=json.loads(self.journal.read_text(encoding='utf-8'))
            if reservation['fingerprint']!=expected_fingerprint: raise ValueError('Privacy journal mismatch.')
            state['privacy_releases']=max(state['privacy_releases'],reservation['reserved_count'])
        return weights,state

    def reserve(self, count, run_fingerprint):
        atomic_json(self.journal,{'fingerprint':run_fingerprint,'reserved_count':count})

    def save_model(self, weights):
        temp=self.folder/'model_fp32.tmp'
        with temp.open('wb') as f:
            np.savez(f,weights=weights);f.flush();os.fsync(f.fileno())
        os.replace(temp,self.folder/'model_fp32.npz')
