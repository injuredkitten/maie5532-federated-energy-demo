"""Client code executes inside worker processes; no local data enters the Grid."""
import hashlib
import numpy as np
from flwr.client import NumPyClient, ClientApp
from flwr.client.mod import secaggplus_mod
from flwr.common import ConfigRecord
from engine import load_clients, flatten, unflatten, clip_update, score
from .model import fit


def model_digest(vector):
    return hashlib.sha256(np.asarray(vector, dtype='f4').tobytes()).hexdigest()


class EnergyClient(NumPyClient):
    def __init__(self, context, config, cache):
        self.context, self.config = context, config
        household = context.node_config['household']
        if household not in cache:
            cache[household] = load_clients(config['data'], [household])[0][0]
        self.data = cache[household]

    def fit(self, parameters, config):
        vector = parameters[0]
        state = self.context.state.config_records.setdefault('training', ConfigRecord())
        if int(config['round']) <= state.get('last_round', -1):
            raise ValueError('Stale/replayed training round rejected')
        if config['digest'] != model_digest(vector):
            raise ValueError('Model digest mismatch')
        if int(config['version']) < state.get('version', -1):
            raise ValueError('Stale model version rejected')
        train = self.data.splits['train']
        seed = self.config['seed'] + 1009*int(config['round']) + 7919*self.data.id
        local = fit(unflatten(vector), train['x'], train['y'], int(config['epochs']),
                    int(config['batch']), float(config['lr']), seed, self.config['backend'])
        delta = clip_update(flatten(local).astype('f8')-vector, float(config['clip']))
        state['last_round'], state['version'] = int(config['round']), int(config['version'])
        # Reproducible stochastic rounding, independent of cryptographic entropy.
        # Privacy sensitivity includes its worst-case error, rather than relying on randomness.
        np.random.seed(seed % (2**32))
        return [delta], 1, {'round': int(config['round']), 'version': int(config['version']),
                            'digest': config['digest']}

    def evaluate(self, parameters, config):
        if self.config['sigma'] > 0:
            raise ValueError('Private mode forbids unprotected evaluation metrics')
        result = score([self.data], unflatten(parameters[0]), split=config['split'])
        return result['mae'], 1, {'mae': result['mae'], 'rmse': result['rmse']}


def make_apps(config):
    cache = {}
    def client_fn(context):
        return EnergyClient(context, config, cache).to_client()
    return (ClientApp(client_fn=client_fn, mods=[secaggplus_mod]),
            ClientApp(client_fn=client_fn))
