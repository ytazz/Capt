#!/usr/bin/env python
import numpy as np
import math

data_1 = np.loadtxt("Biped_lataral_1.csv",delimiter=",")
# data_2 = np.loadtxt("Biped_2.csv",delimiter=",")
# data_3 = np.loadtxt("Biped_3.csv",delimiter=",")
# data_4 = np.loadtxt("Biped_4.csv",delimiter=",")

data_1_o = np.empty((0,4), float)
# data_2_o = np.empty((0,4), float)
# data_3_o = np.empty((0,4), float)
# data_4_o = np.empty((0,4), float)

data_1_o = np.hstack((data_1, np.ones((len(data_1),1))))
#
# for i in range(len(data_2)):
#     if not any((np.equal(data_1, data_2[i])).all(1)):
#         data_2_o = np.vstack((data_2_o, np.append(data_2[i], 2)))
#
# for i in range(len(data_3)):
#     if not any((np.equal(data_2, data_3[i])).all(1)):
#         data_3_o = np.vstack((data_3_o, np.append(data_3[i], 3)))
#
# for i in range(len(data_4)):
#     if not any((np.equal(data_3, data_4[i])).all(1)):
#         data_4_o = np.vstack((data_4_o, np.append(data_4[i], 4)))

merged_data = data_1_o
# # sorted_data = merged_data[merged_data[:,0].argsort(), :]
ind = np.lexsort((merged_data[:,2], merged_data[:,1], merged_data[:,0]))
sorted_data = merged_data[ind]

np.savetxt('Biped_lataral.csv', sorted_data, delimiter=',')
