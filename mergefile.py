#!/usr/bin/env python
import numpy as np
import math

def lexsort_based(data):#get array removed duplicate rows
    sorted_data =  data[np.lexsort(data.T),:]
    row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
    return sorted_data[row_mask]

# def mergefile(filename, minnum, maxnum):
#     for i in range(minnum, maxnum+1):
#         arrays = [np.loadtxt("%s_%d.csv" % (filename, i), delimiter=",")]
#     return arrays
#
# data = mergefile("Biped", 1, 4)
# print data[0][1]

data_1 = np.loadtxt("Biped_1.csv",delimiter=",")
data_2 = np.loadtxt("Biped_2.csv",delimiter=",")
data_3 = np.loadtxt("Biped_3.csv",delimiter=",")
data_4 = np.loadtxt("Biped_4.csv",delimiter=",")

data_1_o = np.empty((0,4), float)
data_2_o = np.empty((0,4), float)
data_3_o = np.empty((0,4), float)
data_4_o = np.empty((0,4), float)

data_1_o = np.hstack((data_1, np.ones((len(data_1),1))))

for i in range(len(data_2)):
    if not any((np.equal(data_1, data_2[i])).all(1)):
        data_2_o = np.vstack((data_2_o, np.append(data_2[i], 2)))

for i in range(len(data_3)):
    if not any((np.equal(data_2, data_3[i])).all(1)):
        data_3_o = np.vstack((data_3_o, np.append(data_3[i], 3)))

for i in range(len(data_4)):
    if not any((np.equal(data_3, data_4[i])).all(1)):
        data_4_o = np.vstack((data_4_o, np.append(data_4[i], 4)))

merged_data = np.vstack((data_1_o, data_2_o, data_3_o, data_4_o))

np.savetxt('temp.csv', merged_data, delimiter=',')
