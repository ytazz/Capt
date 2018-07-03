#!/usr/bin/env python
import numpy as np
import math

height = 0.225
gravity = 9.81
omega0 = math.sqrt(gravity/height)
PI = 3.141592

lmax = 0.15
footsize = 0.045

deltaTs = 0.4
deltaT = 0.2

r_ic_0 = 0.053158
th_ic_0 = 1.322526
d_cp_0 = np.matrix([r_ic_0, th_ic_0])


def calc_d(prev_d):
    dn = (lmax - footsize + prev_d)*math.exp(-omega0*deltaTs) + footsize
    return dn

def r_ic(dT):
    return (r_ic_0 - footsize)*math.exp(omega0*dT) + footsize


r_ic_Dt = r_ic(deltaT)
print r_ic_Dt

d0 = footsize
d1 = calc_d(d0)
d2 = calc_d(d1)
d3 = calc_d(d2)

print d0, d1, d2, d3

captureRegion = np.matrix([r_ic_Dt, th_ic_0, d0])
captureRegion = np.vstack((captureRegion, np.matrix([r_ic_Dt, th_ic_0, d1])))
captureRegion = np.vstack((captureRegion, np.matrix([r_ic_Dt, th_ic_0, d2])))


np.savetxt('koolen_cp.csv', captureRegion, fmt = '%.6f %.6f %.6f', delimiter=',')












# end
