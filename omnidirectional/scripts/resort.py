#!/usr/bin/env python
import numpy as np
import math

data = np.loadtxt('entire_LAST.csv', dtype='float', delimiter=',')

deleted_data = np.delete(data, [0, 1, 4], 1)

target = np.array([0.389474, 0.330632])

ind = np.where(np.all(deleted_data == target, axis = 1))

sorted_data = data[ind]

step_0 = np.empty((0,5), float)
step_1 = np.empty((0,5), float)
step_2 = np.empty((0,5), float)

for i in range(sorted_data.shape[0]):
    if sorted_data[i][4] == 0:
        step_0 = np.vstack((step_0, sorted_data[i]))
    if sorted_data[i][4] == 1:
        step_1 = np.vstack((step_1, sorted_data[i]))
    if sorted_data[i][4] == 2:
        step_2 = np.vstack((step_2, sorted_data[i]))



np.savetxt('0step.csv', step_0, fmt = '%.6f %.6f %.6f %.6f %d', delimiter=',')
np.savetxt('1step.csv', step_1, fmt = '%.6f %.6f %.6f %.6f %d', delimiter=',')
np.savetxt('2step.csv', step_2, fmt = '%.6f %.6f %.6f %.6f %d', delimiter=',')
