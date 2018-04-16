#ifndef __robotParam_H__
#define __robotParam_H__

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>

const int numGrid = 50;

const float heightOfCOM = 0.50;
const float omega = sqrt(9.81/heightOfCOM);

const float step_max_x = 0.3;
const float step_min_x = -0.3;
const float step_max_y = 0.3;
const float step_min_y = -0.3;

const float cp_max_x = 0.2;
const float cp_min_x = -0.2;
const float cp_max_y = 0.2;
const float cp_min_y = -0.2;

#else
#endif
