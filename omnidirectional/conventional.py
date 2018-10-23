#!/usr/bin/env python
import numpy as np
import math

height = 0.3
gravity = 9.81
omega0 = math.sqrt(gravity/height)
PI = 3.141592

vmax = 1.0
r = 0.22
footsize = 0.040


cp_0 = np.matrix([[0.056842], [2.645053]])

# th1 = PI*3.0/2.0 - cp_0[1] - (PI*20/180.0)
# lmax = math.sin(th1) * 0.22
lmax = 0.22

# deltaTs = 0.1 + 0.42/vmax
deltaTs = 0.1
# deltaTs = 0.1

print deltaTs
print lmax

def calc_d(prev_d):
    dn = (lmax - footsize + prev_d)*math.exp(-omega0*deltaTs) + footsize
    return dn

def r_ic(deltaT):
    return (cp_0[0] - footsize)*math.exp(omega0*deltaT) + footsize


r_ic_Dt = r_ic(deltaTs)
print r_ic_Dt

d0 = footsize
d1 = calc_d(d0)
d2 = calc_d(d1)
d3 = calc_d(d2)

print d0, d1, d2, d3

captureRegion = np.matrix([r_ic_Dt, cp_0[1], d0])
captureRegion = np.vstack((captureRegion, np.matrix([r_ic_Dt, cp_0[1], d1])))
captureRegion = np.vstack((captureRegion, np.matrix([r_ic_Dt, cp_0[1], d2])))
captureRegion = np.vstack((captureRegion, np.matrix([r_ic_Dt, cp_0[1], d3])))

np.savetxt('data/conventional.csv', captureRegion, fmt = '%.6f %.6f %.6f', delimiter=',')











# end
