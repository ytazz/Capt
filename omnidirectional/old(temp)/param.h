#ifndef __param_H__
#define __param_H__

#include <stdio.h>
#include <math.h>

#define G 9.81
#define HEIGHTOFCOM 0.50
#define OMEGA sqrt(G/HEIGHTOFCOM)
#define FOOTVEL 1.4
#define XFOOTSIZE 0.05
#define YFOOTSIZE 0.05
#define MINIMUM_STEPPING_TIME 0.05

#define c_step_x = 0.1
#define c_step_y = 0.2

const int numGrid = 50;
const long int N = (long int)numGrid*numGrid*numGrid*numGrid;
const int M = 4;

const int threadsPerBlock = 512;
const int blocksPerGrid = 128;

const float cp_max_x = 0.3;
const float cp_min_x = -0.3;
const float cp_max_y = 0.3;
const float cp_min_y = -0.3;

const float step_max_x = 0.3;
const float step_min_x = -0.3;
const float step_max_y = 0.3;
const float step_min_y = 0.1;

#else
#endif
