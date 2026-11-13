"""Conservative replacement-household Gaussian RDP accounting after quantization."""
import math
import random
import numpy as np
from dp_accounting import GaussianDpEvent
from dp_accounting.rdp import RdpAccountant

ORDERS=[1.25,1.5,1.75,2.,3.,4.,5.,8.,10.,16.,32.,64.,128.,256.]


def spent(count, sigma, delta=1e-5):
    if count==0: return 0.
    if sigma<=0: raise ValueError('Noise multiplier must be positive.')
    accountant=RdpAccountant(orders=ORDERS)
    accountant.compose(GaussianDpEvent(sigma),count=count)
    return float(accountant.get_epsilon(delta))


def effective_bound(clip, dimension, quantization_range=2**22):
    # Stochastic rounding changes each coordinate by at most a full grid step.
    # SecAgg+ max_weight=1 and all num_examples=1 make weighting exact.
    return clip+math.sqrt(dimension)*(2*clip/quantization_range)


def privatize(average, survivors, clip, sigma):
    if survivors<2: raise ValueError('At least two household contributions required.')
    sensitivity=2*effective_bound(clip,len(average))/survivors
    # OS-backed randomness. Never seed with the experiment seed or serialize RNG state.
    rng=random.SystemRandom()
    noise=np.fromiter((rng.normalvariate(0,sigma*sensitivity) for _ in average),dtype=np.float64,count=len(average))
    return average.astype(np.float64)+noise, sigma*sensitivity
