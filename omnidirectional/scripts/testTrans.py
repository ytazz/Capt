#!/usr/bin/env python

import numpy as np


def calcTheta(x, y):
    if x == 0 and y == 0:
        return 0
    else:
        r = np.sqrt(x * x + y * y)
        if x < 0:
            theta = np.arccos(y / r)
        else:
            theta = -np.pi + np.arccos(-y / r)
        return theta

def R(theta):
    return np.matrix([[np.cos(theta), np.sin(theta)],
                     [-np.sin(theta), np.cos(theta)]])


if __name__ == '__main__':

    theta = calcTheta(5, 4)
    print theta*180/np.pi
    hatXd = 5.0
    hatYd = 2.0

    T = np.matrix([[np.cos(theta), -np.sin(theta), hatXd],
                    [np.sin(theta), np.cos(theta), hatYd],
                    [0.0,   0.0,              1.0]])

    T_inv = np.matrix([[np.cos(theta), np.sin(theta), -hatXd*np.cos(theta) - hatYd*np.sin(theta) ],
                       [-np.sin(theta), np.cos(theta), hatXd*np.sin(theta) - hatYd*np.cos(theta) ],
                       [0.0,   0.0,              1.0]])

    print np.linalg.inv(T)*np.matrix([[0], [0], [1]])
    print T_inv*np.matrix([[0], [0], [1]])
