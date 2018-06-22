#ifndef __capturability_H__
#define __capturability_H__

#include "common/nvidia.h"
#include <cuda.h>
#include <vector>
#include <math.h>
#include <iostream>
#include <string>

using namespace std;

#define sq(x) ((x) * (x))

const int TPB = 256;
const int BPG = 256;

struct PolarCoord {
    float r, th;
};

struct States {
    PolarCoord cp;
    PolarCoord step;
    int c;
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
#define PI 3.141

#define FAILED 100.0

const int N_CP_R = 20;
const int N_CP_TH = 20;
const int N_FOOT_R = 20;
const int N_FOOT_TH = 20;
const int N_INPUT = N_FOOT_R*N_FOOT_TH;
const long int N_ENTIRE = (long int)N_CP_R * N_CP_TH * N_FOOT_R * N_FOOT_TH;

#define CP_MIN_R 0.0
#define CP_MAX_R 0.5

#define CP_MIN_TH 0.0
#define CP_MAX_TH 2*PI

#define FOOT_MIN_R 2*FOOTSIZE
#define FOOT_MAX_R 0.5

#define FOOT_MIN_TH 0.0
#define FOOT_MAX_TH PI
////////////////////////////////////////////////////////////////////////////////

void linspace(float result[], float min, float max, int n);
void makeStatesSpace(States* result, float cpR[], float cpTh[], float stepR[], float stepTh[] );
void makeInputSpace(PolarCoord* result, float stepR[], float stepTh[]);
void writeFile(std::vector<States> data, std::string str);

__global__ void step_0(States *statesSpace);
__global__ void step_N(States *statesSpace, PolarCoord input, int n_step,
                       float *cpR, float *cpTh, float *stepR, float *stepTh);
__device__ States stepping (States p0, PolarCoord u);
__device__ float distanceTwoPolar (PolarCoord a, PolarCoord b);
__device__ bool isZeroStepCapt(States p);
__device__ bool isInPrevSet(States *statesSpace, States p, int n_step,
                            float cpR[], float cpTh[], float stepR[], float stepTh[]);
__device__ void setBound(States *bound, States p,
                         float cpR[], float cpTh[], float stepR[], float stepTh[]);

#else
#endif
