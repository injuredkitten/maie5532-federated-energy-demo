"""Integration checks for actual SecAgg+, accounting and interrupted training."""
import json
import logging
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from flwr.common import Context, RecordDict
from engine import flatten, initial_model, PARAMETERS, clip_update
from energy_fl.client import EnergyClient, model_digest
from energy_fl.model import fit
from energy_fl.privacy import spent, effective_bound, privatize
from energy_fl.protocol import aggregate_round, evaluate
from energy_fl.runner import DEFAULTS, run, exclusive_run
from energy_fl.scheduler import plan
from energy_fl.state import RunStore, fingerprint
from energy_fl.transport import ProcessGrid

ROOT = Path(__file__).resolve().parents[1]
CAL = {'backend':'numpy','epoch_seconds':.001,'worker_rss_mb':100.,'tensor_working_set_mb':.13}


def config(**overrides):
    return {**DEFAULTS, 'data': str(ROOT/'data'), 'workers':1, 'clients_per_round':5,
            'rounds':3, 'dropout':0., **overrides}


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.getLogger('flwr').setLevel(logging.CRITICAL)
        cls.cfg = config()
        cls.ids = [1,3,4,5,6]
        cls.grid = ProcessGrid(cls.cfg, cls.ids)
        cls.base = flatten(initial_model(42))
        cls.epochs = [2]*5
    @classmethod
    def tearDownClass(cls): cls.grid.close()

    def expected(self, round_number, selected, clip=.5):
        updates = []
        for i in selected:
            client = EnergyClient(Context(1,i+1,{'household':self.ids[i]},RecordDict(),{}), self.cfg, {})
            vector, _, _ = client.fit([self.base], {'round':round_number,'version':0,
                'digest':model_digest(self.base),'epochs':2,'batch':32,'lr':.03,'clip':clip})
            updates.append(vector[0])
        return np.mean(updates, axis=0)

    def test_01_secure_sum_matches_plain_clipped_updates_and_wire_is_masked(self):
        seen = []
        def audit(stage, message):
            if stage == 'collect_masked_vectors':
                self.assertTrue(all(len(record)==0 for record in message.content.array_records.values()))
                self.assertIn('masked_params', message.content.config_records['secaggplus_configs'])
                # Public metadata only, no local loss/scaler/sample count.
                self.assertNotIn('loss', str(message.content.metrics_records))
                seen.append(message.metadata.src_node_id)
        self.grid.audit_hook = audit
        average, survivors = aggregate_round(self.grid,self.base,list(range(5)),self.epochs,1,0,self.cfg)
        self.grid.audit_hook = None
        np.testing.assert_allclose(average,self.expected(1,range(5)),atol=2*.5/2**22,rtol=0)
        self.assertEqual(len(survivors),5); self.assertEqual(len(seen),5)
        self.assertEqual([s['stage'] for s in self.grid.stats[-4:]],
                         ['setup','share_keys','collect_masked_vectors','unmask'])

    def test_02_dropped_client_after_key_sharing_is_recovered(self):
        self.grid.failed = {5}
        average, survivors = aggregate_round(self.grid,self.base,list(range(5)),self.epochs,2,0,self.cfg)
        np.testing.assert_allclose(average,self.expected(2,range(4)),atol=2*.5/2**22,rtol=0)
        self.assertEqual(set(survivors),{0,1,2,3})

    def test_03_insufficient_shares_abort_without_aggregate(self):
        self.grid.failed = {3,4,5}
        average, survivors = aggregate_round(self.grid,self.base,list(range(5)),self.epochs,3,0,self.cfg)
        self.assertIsNone(average); self.assertEqual(survivors,[])

    def test_04_drop_during_unmask_preserves_already_uploaded_contribution(self):
        self.grid.failed={5}; self.grid.drop_stage='unmask'
        average, survivors=aggregate_round(self.grid,self.base,list(range(5)),self.epochs,4,0,self.cfg)
        np.testing.assert_allclose(average,self.expected(4,range(5)),atol=2*.5/2**22,rtol=0)
        self.assertEqual(len(survivors),5)
        self.grid.failed=set(); self.grid.drop_stage='collect_masked_vectors'

    def test_05_replay_is_rejected_locally(self):
        client=EnergyClient(Context(1,1,{'household':1},RecordDict(),{}),self.cfg,{})
        cfg={'round':1,'version':0,'digest':model_digest(self.base),'epochs':1,'batch':32,'lr':.03,'clip':.5}
        client.fit([self.base],cfg)
        with self.assertRaisesRegex(ValueError,'Stale/replayed'): client.fit([self.base],cfg)
        with self.assertRaisesRegex(ValueError,'digest'):
            client.fit([self.base],{**cfg,'round':2,'digest':'wrong'})

    def test_06_small_clip_secure_aggregate_retains_precision(self):
        cfg={**self.cfg,'clip':.001}
        average,survivors=aggregate_round(self.grid,self.base,list(range(5)),self.epochs,5,0,cfg)
        np.testing.assert_allclose(average,self.expected(5,range(5),.001),atol=.002/2**22,rtol=0)
        self.assertLessEqual(np.linalg.norm(average),effective_bound(.001,PARAMETERS))


class ImprovementTests(unittest.TestCase):
    def test_cohorts_rotate_device_tiers_and_matched_control_matches_size(self):
        from energy_fl.scheduler import trace
        counts=[0]*30;last=[0]*30
        public=(np.ones(30,dtype=bool),np.zeros(30,dtype=bool),np.ones(30))
        with patch('energy_fl.scheduler.trace',return_value=public):
            for number,tier in ((1,2),(2,1),(3,0)):
                scheduled=plan(number,list(range(30)),counts,last,config(policy='cohort',clients_per_round=10),CAL)
                control=plan(number,list(range(30)),counts,last,config(policy='cohort_matched_uniform',clients_per_round=10),CAL)
                self.assertEqual(len(scheduled['selected']),10)
                self.assertEqual(len(control['selected']),len(scheduled['selected']))
                self.assertEqual({i%3 for i in scheduled['selected']},{tier})

    def test_cohort_selection_does_not_see_future_dropouts(self):
        a=plan(1,list(range(30)),[0]*30,[0]*30,config(policy='cohort',dropout=0),CAL)
        b=plan(1,list(range(30)),[0]*30,[0]*30,config(policy='cohort',dropout=1),CAL)
        self.assertEqual(a['selected'],b['selected'])
        self.assertNotEqual(a['failed'],b['failed'])

    def test_sparse_tier_fills_to_secure_minimum(self):
        online=np.array([True,True,True,True,False,False])
        with patch('energy_fl.scheduler.trace',return_value=(online,np.zeros(6,dtype=bool),np.ones(6))):
            scheduled=plan(1,list(range(6)),[0]*6,[0]*6,config(policy='cohort'),CAL)
        self.assertEqual(len(scheduled['selected']),3)
        self.assertEqual(len(set(scheduled['selected'])),3)
        self.assertTrue(all(online[i] for i in scheduled['selected']))

    def test_public_initializer_disjointness_and_integrity(self):
        from energy_fl.initialization import load_public_model
        import shutil
        manifest=json.loads((ROOT/'data30/manifest.json').read_text(encoding='utf-8'))
        vector,info=load_public_model(ROOT/'artifacts/public_init.npz',manifest)
        self.assertEqual(vector.shape,(PARAMETERS,))
        self.assertFalse(set(info['household_ids'])&{c['id'] for c in manifest['clients']})
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'init.npz'
            shutil.copyfile(ROOT/'artifacts/public_init.npz',path)
            path.with_suffix('.json').write_text(json.dumps({**info,'household_ids':[1]}),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'overlap'): load_public_model(path,manifest)
            path.with_suffix('.json').write_text(json.dumps(info),encoding='utf-8')
            path.write_bytes(path.read_bytes()+b'tampered')
            with self.assertRaisesRegex(ValueError,'checksum'): load_public_model(path,manifest)

    def test_validation_only_run_does_not_touch_test_split(self):
        original=evaluate;seen=[]
        def validation_only(grid,weights,split):
            self.assertEqual(split,'val');seen.append(split)
            return original(grid,weights,split)
        with tempfile.TemporaryDirectory() as folder,patch('energy_fl.protocol.evaluate',side_effect=validation_only):
            summary=run(config(rounds=1,evaluation='validation',
                               initial_model=str(ROOT/'artifacts/public_init.npz')),
                        folder,calibration=CAL)
        self.assertTrue(seen);self.assertNotIn('test',summary)
        self.assertEqual(summary['initialization'],'disjoint_public')

    def test_public_initialization_is_not_reapplied_on_resume(self):
        cfg=config(initial_model=str(ROOT/'artifacts/public_init.npz'),clip=.001,evaluation='none')
        with tempfile.TemporaryDirectory() as folder:
            direct,split=Path(folder)/'direct',Path(folder)/'split'
            run(cfg,direct,calibration=CAL)
            run(cfg,split,stop_after=1,calibration=CAL)
            resumed=run(cfg,split,resume=True,calibration=CAL)
            with np.load(direct/'model_fp32.npz') as a,np.load(split/'model_fp32.npz') as b:
                np.testing.assert_array_equal(a['weights'],b['weights'])
            self.assertEqual(resumed['version'],3)
            with self.assertRaises(ValueError):
                run({**cfg,'initial_model':None},split,resume=True,calibration=CAL)


class PrivacyAndStateTests(unittest.TestCase):
    def test_accountant_monotone_and_below_basic_rdp_bound(self):
        from energy_fl.privacy import ORDERS
        values=[spent(k,4.) for k in range(4)]
        self.assertEqual(values[0],0); self.assertTrue(all(a<b for a,b in zip(values,values[1:])))
        bound=min(3*a/(2*4**2)+math.log(1e5)/(a-1) for a in ORDERS)
        self.assertLessEqual(values[-1],bound)

    def test_quantization_bound_and_noise_not_seeded(self):
        self.assertGreater(effective_bound(.5,PARAMETERS),.5)
        np.random.seed(42); a,std=privatize(np.zeros(PARAMETERS),5,.5,4)
        np.random.seed(42); b,_=privatize(np.zeros(PARAMETERS),5,.5,4)
        self.assertFalse(np.array_equal(a,b))
        self.assertAlmostEqual(std,4*2*effective_bound(.5,PARAMETERS)/5)
        with self.assertRaises(ValueError): privatize(np.zeros(10),1,.5,4)

    def test_uncommitted_reservation_is_charged_after_crash(self):
        with tempfile.TemporaryDirectory() as folder:
            store=RunStore(folder); w=np.ones(10)
            store.save(w,{'fingerprint':'abc','privacy_releases':2})
            store.reserve(3,'abc')
            got,state=store.load('abc')
            np.testing.assert_array_equal(w,got); self.assertEqual(state['privacy_releases'],3)
            with self.assertRaises(ValueError): store.load('changed')

    def test_output_lock_prevents_two_controllers(self):
        with tempfile.TemporaryDirectory() as folder:
            with exclusive_run(folder):
                with self.assertRaises((RuntimeError,BlockingIOError)):
                    with exclusive_run(folder): pass

    def test_private_evaluation_is_rejected_before_network(self):
        from types import SimpleNamespace
        with self.assertRaises(ValueError): evaluate(SimpleNamespace(config={'sigma':4}),np.zeros(10),'val')

    def test_cpu_reference_memory_excludes_infeasible_clients(self):
        result=plan(1,list(range(30)),[0]*30,[0]*30,config(),{**CAL,'worker_rss_mb':300.})
        self.assertTrue(all(i%3!=0 for i in result['selected']))

    def test_round_boundary_resume_matches_uninterrupted_model(self):
        with tempfile.TemporaryDirectory() as folder:
            direct, split=Path(folder)/'direct',Path(folder)/'split'
            a=run(config(),direct,calibration=CAL)
            run(config(),split,stop_after=1,calibration=CAL)
            b=run(config(),split,resume=True,calibration=CAL)
            with np.load(direct/'checkpoint.npz') as fa,np.load(split/'checkpoint.npz') as fb:
                np.testing.assert_array_equal(fa['weights'],fb['weights'])
            self.assertEqual(a['version'],b['version']); self.assertEqual(a['fairness'],b['fairness'])
            with self.assertRaises(ValueError): run(config(lr=.2),split,resume=True,calibration=CAL)
            from energy_fl.benchmark import execute
            with self.assertRaises(ValueError): execute(direct,config(rounds=9),CAL)

    def test_budget_stop_hides_metrics_and_counts_only_releases(self):
        with tempfile.TemporaryDirectory() as folder:
            cfg=config(sigma=4.,epsilon_budget=(spent(1,4)+spent(2,4))/2)
            summary=run(cfg,folder,calibration=CAL)
            self.assertEqual(summary['status'],'privacy_budget_exhausted')
            self.assertEqual(summary['privacy_releases'],1); self.assertEqual(summary['version'],1)
            self.assertNotIn('test',summary)
            self.assertIsNone(summary['history'][0]['validation'])
            self.assertIsNone(summary['history'][0]['wall_seconds'])

    def test_aborted_secure_round_does_not_spend_budget(self):
        with tempfile.TemporaryDirectory() as folder:
            summary=run(config(sigma=4.,dropout=1.,rounds=1),folder,calibration=CAL)
            self.assertEqual(summary['privacy_releases'],0);self.assertEqual(summary['version'],0)

    def test_monitor_updates_status_but_does_not_serve_checkpoint(self):
        from energy_fl.monitor import make_server
        from energy_fl.state import atomic_json
        from threading import Thread
        from urllib.request import urlopen
        from urllib.error import HTTPError
        with tempfile.TemporaryDirectory() as folder:
            server=make_server(folder,0)
            thread=Thread(target=server.serve_forever,daemon=True);thread.start()
            url=f'http://127.0.0.1:{server.server_port}'
            try:
                with urlopen(url+'/api/status') as res:
                    self.assertEqual(json.load(res)['status'],'waiting_for_training')
                atomic_json(Path(folder)/'progress.json',{'round':7,'status':'running'})
                with urlopen(url+'/api/status') as res: self.assertEqual(json.load(res)['round'],7)
                (Path(folder)/'checkpoint.npz').write_bytes(b'not for serving')
                for route in ('/checkpoint.npz','/../checkpoint.npz','/config.json'):
                    with self.assertRaises(HTTPError) as error: urlopen(url+route)
                    self.assertEqual(error.exception.code,404)
            finally:
                server.shutdown();server.server_close();thread.join(timeout=2)


if __name__ == '__main__': unittest.main()
