#!/usr/bin/env python
import numpy as np
import cupy as cp


l2norm_kernel = cp.ReductionKernel(
     'T x',  # input params
     'T y',  # output params
     'x * x',  # map
     'a + b',  # reduce
     'y = sqrt(a)',  # post-reduction map
     '0',  # identity value
     'l2norm'  # kernel name
)
x = cp.arange(2, dtype=np.float32).reshape(2, 1)
print l2norm_kernel(x, axis=1)
