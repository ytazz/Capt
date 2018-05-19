#ifndef __capturability_H__
#define __capturability_H__

#include "common/nvidia.h"
#include <cuda.h>
#include <vector>
#include <math.h>

using namespace std;

struct TwoDim {
  float x, y;
};

struct States {
  TwoDim step;
  TwoDim cp;
};

//////////////////////////////// parameter ////////////////////////////////////
////////////  [m]  ////////////
#define HEIGHTOFCOM 0.50
#define FOOTVEL 1.4
#define FOOTSIZE 0.1
////////////  [s]  ////////////
#define MINIMUM_STEPPING_TIME 0.05
///////////////////////////////
#define OMEGA sqrt(9.81/HEIGHTOFCOM)
#define PI 3.141592



const int numGrid = 10;
const long int N = (long int)numGrid*numGrid*numGrid*numGrid;

const int threadsPerBlock = 1024;
const int blocksPerGrid = 1024;

const TwoDim CP_MAX = {0.3, 0.3};
const TwoDim CP_MIN = {-0.3, -0.3};
const TwoDim FOOT_MAX = {0.3, 0.3};
const TwoDim FOOT_MIN = {0.1, 0.1};
////////////////////////////////////////////////////////////////////////////////

void linspace(float result[], float min, float max);

void makeStatesSpace(States* result,
                        float stepX[], float stepY[], float cpX[], float cpY[] );

void writeFile(States* data);

__device__ TwoDim rotation_inv(TwoDim in, float theta);

__device__ float calcTheta (TwoDim p);

__global__ void transf(States *hat_set, States *set );
#else
#endif
