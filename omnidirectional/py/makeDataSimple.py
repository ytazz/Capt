#!/usr/bin/env python
import numpy as np
import math

def lexsort_based(data):#get array removed duplicate rows
    sorted_data =  data[np.lexsort(data.T),:]
    row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
    return sorted_data[row_mask]

data = np.loadtxt("1step_capturability.csv",delimiter=",")
sortedData = lexsort_based(np.delete(data, [2, 3], 1))
np.savetxt('1step_capturability_sorted.csv', sortedData, delimiter=',')
