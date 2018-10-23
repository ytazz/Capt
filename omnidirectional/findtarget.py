#!/usr/bin/env python
import numpy as np
import math

data = np.loadtxt('build/data.csv', dtype='float', delimiter=',')

deleted_data = np.delete(data, [4, 5, 6, 7], 1)

target = np.array([0.056842, 2.645053, 0.158421, 1.120474])

ind = np.where(np.all(deleted_data == target, axis = 1))

sorted_data = data[ind]

step_1 = np.empty((0,8), float)
step_2 = np.empty((0,8), float)
step_3 = np.empty((0,8), float)

for i in range(sorted_data.shape[0]):
    if sorted_data[i][7] == 1:
        step_1 = np.vstack((step_1, sorted_data[i]))
    if sorted_data[i][7] == 2:
        step_2 = np.vstack((step_2, sorted_data[i]))
    if sorted_data[i][7] == 3:
        step_3 = np.vstack((step_3, sorted_data[i]))



np.savetxt('data/1step.csv', step_1, fmt = '%.6f %.6f %.6f %.6f %d %.6f %.6f %d', delimiter=',')
np.savetxt('data/2step.csv', step_2, fmt = '%.6f %.6f %.6f %.6f %d %.6f %.6f %d', delimiter=',')
np.savetxt('data/3step.csv', step_3, fmt = '%.6f %.6f %.6f %.6f %d %.6f %.6f %d', delimiter=',')
