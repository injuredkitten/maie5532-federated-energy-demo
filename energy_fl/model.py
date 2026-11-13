"""Compatible NumPy/PyTorch training backends; common FP32 wire architecture."""
import time
import numpy as np
from engine import train, initial_model


def fit(weights, x, y, epochs, batch, lr, seed, backend='numpy'):
    if backend == 'numpy':
        return train(weights, x, y, epochs, batch, lr, seed)
    import torch
    device = 'cuda' if backend == 'cuda' else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable. Install torch with CUDA 12.8 or select numpy.')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    w = [torch.tensor(a, device=device, requires_grad=True) for a in weights]
    tx, ty = torch.tensor(x, device=device), torch.tensor(y, device=device)
    optimizer = torch.optim.SGD(w, lr=lr)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(x))
        for pos in range(0,len(x),batch):
            ids = torch.tensor(order[pos:pos+batch],device=device)
            pred = torch.relu(tx[ids]@w[0]+w[1])@w[2]+w[3]
            loss = ((pred-ty[ids])**2).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(w,10.)
            optimizer.step()
    return [a.detach().cpu().numpy().copy() for a in w]


def calibrate(backend='numpy', samples=253, repeats=5):
    """Public synthetic workload: resource selection must not depend on private loads."""
    import psutil
    rng=np.random.default_rng(1234)
    x=rng.normal(size=(samples,52)).astype('f4'); y=rng.normal(size=(samples,24)).astype('f4')
    w=initial_model(1234)
    fit(w,x,y,1,32,.03,1,backend)  # import/CUDA initialization excluded from timing
    timings=[]; by_batch={}
    for batch in (16,32):
        timings=[]
        for i in range(repeats):
            start=time.perf_counter()
            fit(w,x,y,1,batch,.03,i,backend)
            timings.append(time.perf_counter()-start)
        by_batch[str(batch)]=float(np.median(timings))
    rss=psutil.Process().memory_info().rss/2**20
    result={'backend':backend,'samples':samples,'batch':32,'repeats':repeats,
            'epoch_seconds':float(np.median(timings)), 'p95_epoch_seconds':float(np.percentile(timings,95)),
            'epoch_seconds_by_batch':by_batch,
            'worker_rss_mb':rss, 'workload':'public synthetic tensors; excludes initialization',
            'tensor_working_set_mb':float((sum(a.nbytes for a in w)*5+x.nbytes+y.nbytes)/2**20)}
    if backend!='numpy':
        import torch
        result['torch']=torch.__version__
        if backend=='cuda':
            result['gpu']=torch.cuda.get_device_name(0)
            result['gpu_peak_allocated_mb']=torch.cuda.max_memory_allocated()/2**20
    return result
