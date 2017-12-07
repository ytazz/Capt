#!/usr/bin/env python
import numpy as np
import math

def lexsort_based(data):#get array removed duplicate rows
    sorted_data =  data[np.lexsort(data.T),:]
    row_mask = np.append([True],np.any(np.diff(sorted_data,axis=0),1))
    return sorted_data[row_mask]

class Lateral_Capturability(object):
    def __init__(self, text, delta_t=0):
        self.FOOTSIZE = 0.05
        self.OMEGA = math.sqrt(9.81/0.50)
        self.VELOCITY_OF_FOOT = 1.0
        self.MINIMUM_DELTA_T = 0.05
        self.CONSTANT_DELTA_T = delta_t
        self.GRID = 100
        self.count = 0
        self.L_STEP_MIN = -0.35
        self.L_STEP_MAX = 0.35
        self.L_CP_MIN = -0.5
        self.L_CP_MAX = 0.5
        self.string = text

        L_STEP_T1 = np.linspace(self.L_STEP_MIN, -(2*self.FOOTSIZE), self.GRID/2)
        L_STEP_T2 = np.linspace(2*self.FOOTSIZE, self.L_STEP_MAX, self.GRID/2)
        self.L_STEP = np.concatenate([L_STEP_T1, L_STEP_T2], axis=0)
        self.L_CP = np.linspace(self.L_CP_MIN, self.L_CP_MAX, self.GRID)

    def take_a_step(self, l_step):
        if l_step > 0:
            L_STRIDE_MIN = -l_step + 2*self.FOOTSIZE
            L_STRIDE_MAX = -l_step + self.L_STEP_MAX
        elif l_step < 0:
            L_STRIDE_MIN = 0
            L_STRIDE_MAX = -l_step -2*self.FOOTSIZE
        self.L_STRIDE = np.linspace(L_STRIDE_MIN, L_STRIDE_MAX, 50)

    def set_squareframe(self, point):
        if point[0] < self.L_STEP_MIN or point[0] > self.L_STEP_MAX or \
        point[1] < self.L_CP_MIN or point[1] > self.L_CP_MAX:
            return -1
        else :
            if any(point[0] == self.L_STEP):
                A = point[0]
                bA = point[0]
            else:
                for i in range(len(self.L_STEP)):
                    if point[0] < self.L_STEP[i]:
                        A = self.L_STEP[i]
                        bA = self.L_STEP[i-1]
                        break
            if any(point[1] == self.L_CP):
                B = point[1]
                bB = point[1]
            else :
                for j in range(len(self.L_CP)):
                    if point[1] < self.L_CP[j]:
                        B = self.L_CP[j]
                        bB = self.L_CP[j-1]
                        break
            self.squarepoint = np.array([[A  ,  B],
                                         [bA ,  B],
                                         [A  , bB],
                                         [bA , bB]])
            return 1

    def check_frame(self, states_set):
        if any((np.equal(states_set, self.squarepoint[0])).all(1)) \
        and any((np.equal(states_set, self.squarepoint[1])).all(1)) \
        and any((np.equal(states_set, self.squarepoint[2])).all(1)) \
        and any((np.equal(states_set, self.squarepoint[3])).all(1)):
            return True
        else:
            return False

    def set_F(self):#Field
        LL, RR = np.meshgrid(self.L_STEP, self.L_CP)
        self.F = np.c_[LL.ravel(), RR.ravel()]

    def set_O(self):
        self.set_F()
        self.zero_step_capture()
        self.O = self.F
        self.writefile(self.O)

    def zero_step_capture(self):
        self.P0 = np.empty((0,2), float)
        for i in range(len(self.F)):
            if self.F[i][1] <= self.FOOTSIZE and self.F[i][1] >= -self.FOOTSIZE:
                self.P0 = np.append(self.P0, np.array([self.F[i]]), axis = 0)

    def set_P(self):
        self.set_F()
        self.zero_step_capture()
        self.P = self.P0
        self.writefile(self.P)

    def function(self, step, cp, j):
        self.L_CP_NEXT = (cp - self.FOOTSIZE) \
        * math.exp(self.OMEGA*(self.L_STRIDE[j]/self.VELOCITY_OF_FOOT+self.MINIMUM_DELTA_T)) \
        - (step + self.L_STRIDE[j]) + self.FOOTSIZE
        self.L_STEP_NEXT = -(step + self.L_STRIDE[j])

    def cal_inner(self):
        TEMP = np.empty((0,3), float)

        for i in range(len(self.F)):
            self.take_a_step(self.F[i][0])
            for j in range(len(self.L_STRIDE)):
                self.function(step=self.F[i][0], cp=self.F[i][1], j=j)
                F_NEXT = np.array([self.L_STEP_NEXT, self.L_CP_NEXT])
                flag = self.set_squareframe(F_NEXT)
                if flag == 1:
                    if self.check_frame(self.P):
                        row = np.append(self.F[i], self.L_STRIDE[j])
                        TEMP = np.append(TEMP,np.array([row]) , axis = 0)

        P_TEMP = np.vstack((self.P, np.delete(TEMP, 2, 1)))
        self.P = lexsort_based(P_TEMP)
        self.writefile(TEMP)

    def cal_outer(self):
        TEMP = np.empty((0,3), float)

        for i in range(len(self.O)):
            if any((np.equal(self.P0, self.O[i])).all(1)):
                O_NEXT = np.array([0.0 , self.O[i][1]])
                flag = self.set_squareframe(O_NEXT)
                if self.check_frame(self.O):
                    row = np.append(self.O[i], self.O[i][0])
                    TEMP = np.append(TEMP,np.array([row]) , axis = 0)
            else:
                self.take_a_step(self.O[i][0])
                for j in range(len(self.L_STRIDE)):
                    self.function(step=self.O[i][0], cp=self.O[i][1], j=j)
                    O_NEXT = np.array([self.L_STEP_NEXT, self.L_CP_NEXT])
                    flag = self.set_squareframe(O_NEXT)
                    if flag == 1:
                        if self.check_frame(self.O):
                            row = np.append(self.O[i], self.L_STRIDE[j])
                            TEMP = np.append(TEMP,np.array([row]) , axis = 0)

        TEMP = np.delete(TEMP, 2, 1)
        self.O = lexsort_based(TEMP)
        self.writefile(self.O)


    def writefile(self, data):
        np.savetxt('%s_%d.csv' % (self.string, self.count), data, delimiter=',')
        self.count += 1
