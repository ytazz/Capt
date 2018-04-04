#!/usr/bin/env python
import datetime
import numpy as np
import cupy as cp

def test(xp):
    a = xp.arange(1000000).reshape(1000, -1)
    return a.T * 2

# test(np)
t1 = datetime.datetime.now()
for i in range(1000):
    test(np)
t2 = datetime.datetime.now()
print(t2 -t1)

cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
# test(cp)
t1 = datetime.datetime.now()
for i in range(1000):
    test(cp)
t2 = datetime.datetime.now()
print(t2 -t1)
