"""Equal-household clipped-delta FedAvg through Flower's SecAgg+ protocol."""
from types import SimpleNamespace
import numpy as np
from flwr.common import (Context, RecordDict, ConfigRecord, Message, FitIns, EvaluateIns,
                         ndarrays_to_parameters, parameters_to_ndarrays)
from flwr.common import recorddict_compat as compat
from flwr.server.strategy import FedAvg
from flwr.server.compat.legacy_context import LegacyContext
from flwr.server.workflow import SecAggPlusWorkflow
from .client import model_digest


class DeltaStrategy(FedAvg):
    def __init__(self, base, selected, epochs, round_number, version, config):
        super().__init__()
        self.base, self.selected, self.epochs = base, selected, epochs
        self.round_number, self.version, self.config = round_number, version, config
        self.average, self.survivors = None, []

    def configure_fit(self, server_round, parameters, client_manager):
        return [(SimpleNamespace(node_id=i+1, cid=str(i+1)), FitIns(parameters,
                 {'round': server_round, 'version': self.version, 'digest': model_digest(self.base),
                  'epochs': self.epochs[i],
                  'batch': 16 if self.config.get('adaptive_batch') and i%3==0 else self.config['batch'],
                  'lr': self.config['lr'], 'clip': self.config['clip']})) for i in self.selected]

    def aggregate_fit(self, server_round, results, failures):
        if len(results) < 2:
            return None, {}
        for proxy, result in results:
            if (result.metrics.get('round') != self.round_number or
                result.metrics.get('version') != self.version or
                result.metrics.get('digest') != model_digest(self.base)):
                raise ValueError('Stale or mismatched update rejected by server')
            if result.num_examples != 1:
                raise ValueError('Expected equal household weight')
        arrays = [parameters_to_ndarrays(result.parameters)[0] for _, result in results]
        # Flower replaces each FitRes with the same recovered aggregate in secure mode.
        self.average = arrays[0] if self.config['secure'] else np.mean(arrays, axis=0)
        if self.average.shape != self.base.shape or not np.isfinite(self.average).all():
            raise ValueError('Invalid aggregate')
        self.survivors = [proxy.node_id-1 for proxy, _ in results]
        return ndarrays_to_parameters([self.average]), {}


def aggregate_round(grid, base, selected, epochs, round_number, version, config):
    strategy = DeltaStrategy(base, selected, epochs, round_number, version, config)
    parameters = ndarrays_to_parameters([base])
    if config['secure']:
        state = RecordDict({'config': ConfigRecord({'current_round': round_number}),
                            'parameters': compat.parameters_to_arrayrecord(parameters, True)})
        context = LegacyContext(Context(1, 0, {}, state, {}), strategy=strategy,
                                client_manager=SimpleNamespace(num_available=lambda: len(grid.ids)))
        workflow = SecAggPlusWorkflow(num_shares=1.0, reconstruction_threshold=config['threshold'],
                                      max_weight=1., clipping_range=config['clip'])
        workflow(grid, context)
    else:
        pairs = strategy.configure_fit(round_number, parameters, None)
        proxies = {proxy.node_id: proxy for proxy, _ in pairs}
        messages = [Message(compat.fitins_to_recorddict(ins, True), dst_node_id=proxy.node_id,
                            message_type='train', group_id=str(round_number)) for proxy, ins in pairs]
        replies = grid.send_and_receive(messages)
        results = [(proxies[msg.metadata.src_node_id], compat.recorddict_to_fitres(msg.content, True))
                   for msg in replies]
        strategy.aggregate_fit(round_number, results, [])
    return strategy.average, strategy.survivors


def evaluate(grid, weights, split):
    if grid.config['sigma'] > 0:
        raise ValueError('Unprotected evaluation forbidden in DP mode')
    ins = EvaluateIns(ndarrays_to_parameters([weights]), {'split': split})
    replies = grid.send_and_receive([
        Message(compat.evaluateins_to_recorddict(ins, True), dst_node_id=nid, message_type='evaluate')
        for nid in grid.get_node_ids()])
    metrics = [compat.recorddict_to_evaluateres(msg.content).metrics for msg in replies]
    maes = [r['mae'] for r in metrics]
    return {'mae': float(np.mean(maes)), 'rmse': float(np.mean([r['rmse'] for r in metrics])),
            'worst_quintile_mae': float(np.mean(sorted(maes)[-max(1, int(np.ceil(len(maes)*.2))):]))}
