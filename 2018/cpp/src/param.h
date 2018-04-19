#ifndef __param_H__
#define __param_H__

#include <stdio.h>
#include <math.h>

#define G 9.81
#define heightOfCOM 0.50
#define omega sqrt(G/heightOfCOM)

const int numGrid = 50;
const long int N = (long int)numGrid*numGrid*numGrid*numGrid;
const int M = 4;

const int threadsPerBlock = 512;
const int blocksPerGrid = 128;

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
