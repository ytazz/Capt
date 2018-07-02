#!/usr/bin/env python
import numpy as np
import math

height = 0.225
gravity = 9.81
omega0 = math.sqrt(gravity/height)

lmax = 0.15
footsize = 0.045

deltaTs = 0.4

def calc_d(prev_d):
    dn = (lmax - footsize + prev_d)*math.exp(-omega0*deltaTs) + footsize
    return dn

d0 = footsize
d1 = calc_d(d0)
d2 = calc_d(d1)

print d0, d1, d2

r_ic_0 = 0.053158
def r_ic(deltaT):
    return (r_ic_0 - footsize)*math.exp(omega0*deltaT) + footsize

for i in range(10):
    deltaT = 0.1 * i
    r_ic_Dt = r_ic(deltaT)
    print r_ic_Dt
