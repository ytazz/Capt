#!/usr/bin/env python
import numpy as np
import cupy as cp


squared_diff = cp.ElementwiseKernel(
    'float32 x, float32 y',
    'float32 z',
    'z = (x - y) * (x - y)',
    'squared_diff')

squared_diff_generic = cp.ElementwiseKernel(
     'T x, T y',
     'T z',
     '''
         T diff = x - y;
         z = diff * diff;
     ''',
     'squared_diff_generic')

add_reverse = cp.ElementwiseKernel(
    'T x, raw T y',
    'T z',
    'z = x + y[_ind.size() - i - 1]',
    'add_reverse')


x = cp.arange(10, dtype=np.float32).reshape(2, 5)
y = cp.arange(10, dtype=np.float32).reshape(2, 5)

print x
print y
print add_reverse(x, y)
