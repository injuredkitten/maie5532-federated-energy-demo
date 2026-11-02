import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tempfile
import unittest
import numpy as np
from engine import (load_clients, initial_model, flatten, unflatten, gradients, predict,
                    clip_update, aggregate, epsilon, resource_trace, run_fl, export_quantized, PARAMETERS)

ROOT = Path(__file__).resolve().parents[1]


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clients, cls.manifest = load_clients(ROOT/'data')

    def test_data_target_splits_and_scalers(self):
        for c, entry in zip(self.clients,self.manifest['clients']):
            tr, va, te = (c.splits[k]['origins'] for k in ['train','val','test'])
            self.assertLessEqual(tr.max()+24,va.min())
            self.assertLessEqual(va.max()+24,te.min())
            with np.load(ROOT/'data'/entry['file']) as f:
                raw = f['kwh']
            self.assertAlmostEqual(c.mean, float(raw[:c.boundaries[0]].mean()),places=6)
            np.testing.assert_allclose(c.splits['test']['y'][0]*c.scale+c.mean,raw[te[0]:te[0]+24],atol=1e-6)
            self.assertEqual(c.splits['train']['x'].shape[1],52)

    def test_known_hourly_energy_sum(self):
        # First two raw GC entries of household 1 on 2012-07-01: .855 and .786 kWh.
        with np.load(ROOT/'data'/'home_001.npz') as f:
            self.assertAlmostEqual(float(f['kwh'][0]), .855+.786, places=6)

    def test_backprop_against_finite_difference(self):
        rng=np.random.default_rng(7)
        w=[a.astype('f8') for a in initial_model(7)]
        x=rng.normal(size=(3,52)); y=rng.normal(size=(3,24))
        g=gradients(w,x,y)
        for k,idx in [(0,(3,4)),(1,(4,)),(2,(4,3)),(3,(3,))]:
            orig=w[k][idx]; step=1e-5
            w[k][idx]=orig+step; plus=np.mean((predict(w,x)-y)**2)
            w[k][idx]=orig-step; minus=np.mean((predict(w,x)-y)**2)
            w[k][idx]=orig
            self.assertAlmostEqual(g[k][idx],(plus-minus)/(2*step),places=7)

    def test_equal_household_mean_and_abort(self):
        base=np.zeros(3,dtype='f4')
        result,ok=aggregate(base,[np.array([1,2,3]),np.array([3,4,5])],[0,0],0)
        self.assertTrue(ok); np.testing.assert_array_equal(result,[2,3,4])
        result,ok=aggregate(base,[np.ones(3)],[0],0)
        self.assertFalse(ok); np.testing.assert_array_equal(result,base)

    def test_stale_updates_rejected(self):
        with self.assertRaises(ValueError):
            aggregate(np.zeros(3),[np.ones(3),np.ones(3)],[1,0],1)

    def test_clipping_and_gaussian_placement(self):
        np.testing.assert_allclose(clip_update(np.array([3.,4.]),2),[1.2,1.6])
        u=[np.array([3.,4.]),np.array([4.,3.])]
        a,_=aggregate(np.zeros(2),u,[0,0],0,clip=2,sigma=1,rng=np.random.default_rng(3))
        expected=np.array([1.4,1.4])+np.random.default_rng(3).normal(0,2,2)
        np.testing.assert_allclose(a,expected,rtol=1e-6)

    def test_accountant_and_budget_stop_before_release(self):
        self.assertEqual(epsilon(0,2),0)
        self.assertGreater(epsilon(2,2),epsilon(1,2))
        self.assertGreater(epsilon(1,1),epsilon(1,2))
        cfg=dict(rounds=3,seed=42,epochs=1,clients_per_round=5,lr=.03,deadline=5,clip=.05,sigma=2,epsilon_budget=.01)
        with tempfile.TemporaryDirectory() as folder:
            w,r=run_fl(self.clients,cfg,'resource',folder,private=True)
        self.assertTrue(r['privacy_budget_stopped']); self.assertEqual(r['releases'],0)
        np.testing.assert_array_equal(flatten(w),flatten(initial_model(42)))

    def test_common_resource_traces(self):
        a=resource_trace(self.clients,4,42); b=resource_trace(self.clients,4,42)
        for key in a: np.testing.assert_array_equal(a[key],b[key])

    def test_weight_serialization_and_quantization(self):
        w=initial_model(42)
        np.testing.assert_array_equal(flatten(unflatten(flatten(w))),flatten(w))
        self.assertEqual(len(flatten(w)),PARAMETERS)
        with tempfile.TemporaryDirectory() as folder:
            np.savez(Path(folder)/'model_fp32.npz',**{f'w{i}':a for i,a in enumerate(w)})
            result=export_quantized(w,self.clients,folder)
            self.assertLess(result['int8_storage_bytes'],result['fp32_bytes']/2)
            self.assertTrue(np.isfinite(result['int8_storage_test']['mae']))


if __name__=='__main__':
    unittest.main()
