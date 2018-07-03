#!/usr/bin/env python
import numpy as np
import math

data = np.loadtxt('data.csv', dtype='float', delimiter=',')

deleted_data = np.delete(data, [4, 5, 6, 7], 1)

target = np.array([0.061316, 1.322526, 0.137368, 1.983789])

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



np.savetxt('1step.csv', step_1, fmt = '%.6f %.6f %.6f %.6f %d %.6f %.6f %d', delimiter=',')
np.savetxt('2step.csv', step_2, fmt = '%.6f %.6f %.6f %.6f %d %.6f %.6f %d', delimiter=',')
np.savetxt('3step.csv', step_3, fmt = '%.6f %.6f %.6f %.6f %d %.6f %.6f %d', delimiter=',')
