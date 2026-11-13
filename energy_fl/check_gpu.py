"""Optional CUDA correctness check; writes compact, non-sensitive evidence."""
import numpy as np
from engine import initial_model, flatten
from .model import fit
from .state import atomic_json


if __name__=='__main__':
    import torch
    rng=np.random.default_rng(42)
    x=rng.normal(size=(253,52)).astype('f4');y=rng.normal(size=(253,24)).astype('f4')
    w=initial_model(42)
    cpu=flatten(fit(w,x,y,2,32,.03,42,'numpy'))
    gpu=flatten(fit(w,x,y,2,32,.03,42,'cuda'))
    np.testing.assert_allclose(cpu,gpu,atol=2e-6,rtol=2e-5)
    again=flatten(fit(w,x,y,2,32,.03,42,'cuda'))
    np.testing.assert_array_equal(gpu,again)
    result={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
            'cuda_available':torch.cuda.is_available(),'cpu_gpu_max_abs_difference':float(np.max(np.abs(cpu-gpu))),
            'same_seed_cuda_repeat_exact':True,'atol':2e-6,'rtol':2e-5,
            'scope':'public synthetic tensors; SGD compatibility and local determinism'}
    atomic_json('evidence/gpu_correctness.json',result)
    print(result)
