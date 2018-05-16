#ifndef __param_H__
#define __param_H__

#include <stdio.h>
#include <math.h>

struct TwoDim {
  float x, y;
};

#define G 9.81
#define HEIGHTOFCOM 0.50
#define OMEGA sqrt(G/HEIGHTOFCOM)
#define FOOTVEL 1.4
#define MINIMUM_STEPPING_TIME 0.05
#define PI 3.141592

const int numGrid = 50;
const long int N = (long int)numGrid*numGrid*numGrid*numGrid;
const int M = 4;

const int threadsPerBlock = 512;
const int blocksPerGrid = 128;

const TwoDim CP_MAX = {0.3, 0.3};
const TwoDim CP_MIN = {-0.3, -0.3};
const TwoDim FOOT_MAX = {0.3, 0.3};
const TwoDim FOOT_MIN = {0.1, 0.1};


#else
#endif
