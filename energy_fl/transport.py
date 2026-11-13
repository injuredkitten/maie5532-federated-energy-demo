"""Bounded local process Grid for the official Flower SecAgg+ workflow.

Only serialized Flower messages cross process boundaries. This is a same-host
simulation, not SuperLink/TLS deployment or an adversarial OS security boundary.
"""
import multiprocessing as mp
import time
import uuid
from collections import defaultdict
from flwr.common import Context, Message, RecordDict
from flwr.common.serde import message_to_proto, message_from_proto
from flwr.proto.message_pb2 import Message as ProtoMessage
from flwr.common.secure_aggregation.secaggplus_constants import RECORD_KEY_CONFIGS, Key


def encode(message):
    return message_to_proto(message).SerializeToString()


def decode(payload):
    return message_from_proto(ProtoMessage.FromString(payload))


def _worker(connection, config, households):
    import logging
    logging.getLogger('flwr').setLevel(logging.ERROR)
    from .client import make_apps
    secure_app, plain_app = make_apps(config)
    contexts = {nid: Context(1, nid, {'household': household}, RecordDict(), {})
                for nid, household in households.items()}
    connection.send(('ready', None))
    while True:
        batch = connection.recv()
        if batch is None:
            break
        replies = []
        for payload in batch:
            msg = decode(payload)
            try:
                app = secure_app if RECORD_KEY_CONFIGS in msg.content.config_records else plain_app
                reply = app(msg, contexts[msg.metadata.dst_node_id])
                replies.append(encode(reply))
            except Exception as exc:
                # Fail closed. Do not include input arrays or client exception locals in logs.
                connection.send(('error', f'{type(exc).__name__}: {exc}'))
                return
        connection.send(('ok', replies))
    connection.close()


class ProcessGrid:
    """Implements the synchronous Grid interface used by SecAggPlusWorkflow."""
    def __init__(self, config, ids):
        self.config, self.ids = config, ids
        self.connections, self.processes = [], []
        self.failed = set()
        self.drop_stage = 'collect_masked_vectors'
        self.stats = []
        self.audit_hook = None  # tests may inspect wire records; never persisted
        count = min(config['workers'], len(ids))
        ctx = mp.get_context('spawn')
        for index in range(count):
            parent, child = ctx.Pipe()
            owned = {i+1: household for i, household in enumerate(ids) if i % count == index}
            process = ctx.Process(target=_worker, args=(child, config, owned), daemon=True)
            process.start(); child.close()
            self.connections.append(parent); self.processes.append(process)
        try:
            for connection in self.connections:
                if not connection.poll(120) or connection.recv()[0] != 'ready':
                    raise TimeoutError('Client worker failed to initialize')
        except BaseException:
            self.close(); raise

    def get_node_ids(self):
        return range(1, len(self.ids)+1)

    def create_message(self, content, message_type, dst_node_id, group_id, ttl=None):
        return Message(content, dst_node_id=dst_node_id, message_type=message_type,
                       group_id=group_id, ttl=ttl)

    def send_and_receive(self, messages, *, timeout=None):
        batches = defaultdict(list)
        sent = received = skipped = 0
        traffic = defaultdict(lambda: {'down': 0, 'up': 0})
        stage = 'plain'
        for message in messages:
            record = message.content.config_records.get(RECORD_KEY_CONFIGS)
            stage = record[Key.STAGE] if record is not None else message.metadata.message_type
            proto = message_to_proto(message)
            proto.metadata.message_id = uuid.uuid4().hex
            proto.metadata.run_id = 1
            payload = proto.SerializeToString()
            sent += len(payload)
            traffic[message.metadata.dst_node_id]['down'] += len(payload)
            # Emulate dropout after key exchange, exercising actual share recovery.
            if stage in (self.drop_stage, 'train') and message.metadata.dst_node_id in self.failed:
                skipped += 1; continue
            index = (message.metadata.dst_node_id-1) % len(self.connections)
            batches[index].append(payload)
        for index, batch in batches.items():
            self.connections[index].send(batch)
        result = []
        expires = time.monotonic() + (timeout or 120)
        for index in batches:
            connection = self.connections[index]
            if not connection.poll(max(0., expires-time.monotonic())):
                # A hung process is fatal, not reused with ambiguous protocol state.
                raise TimeoutError('Worker deadline exceeded; restart from committed checkpoint')
            status, payloads = connection.recv()
            if status != 'ok':
                raise RuntimeError(payloads)
            for payload in payloads:
                received += len(payload)
                reply = decode(payload)
                traffic[reply.metadata.src_node_id]['up'] += len(payload)
                if self.audit_hook is not None:
                    self.audit_hook(stage, reply)
                result.append(reply)
        self.stats.append({'stage': stage, 'down_bytes': sent, 'up_bytes': received,
                           'replies': len(result), 'simulated_drops': skipped,
                           'traffic': dict(traffic)})
        return result

    def close(self):
        for connection, process in zip(self.connections, self.processes):
            if process.is_alive():
                try: connection.send(None)
                except (BrokenPipeError, EOFError, OSError): pass
                process.join(timeout=2)
                if process.is_alive(): process.terminate(); process.join(timeout=2)
            connection.close()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
