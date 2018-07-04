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

const int TPB = 1024;
const int BPG = 1024;

//////////////////////////////// parameter ////////////////////////////////////
#define HEIGHTOFCOM 0.225            //[m]
#define FOOTVEL 1.0                 //[m/s]
#define FOOTSIZE 0.045                //[m]
#define MINIMUM_STEPPING_TIME 0.1  //[s]
#define OMEGA sqrt(9.81/HEIGHTOFCOM)
#define PI 3.141

#define FAILED 100.0

const int N_CP_R = 20;
const int N_CP_TH = 20;
const int N_FOOT_R = 20;
const int N_FOOT_TH = 20;
const int N_INPUT = N_FOOT_R*N_FOOT_TH;
const long int N_STATE = (long int)N_CP_R * N_CP_TH * N_FOOT_R * N_FOOT_TH;

#define CP_MIN_R FOOTSIZE
#define CP_MAX_R 0.2

#define CP_MIN_TH 0.0
#define CP_MAX_TH 2*PI

#define FOOT_MIN_R 0.1
#define FOOT_MAX_R 0.2

#define FOOT_MIN_TH 0.0
#define FOOT_MAX_TH PI
////////////////////////////////////////////////////////////////////////////////
struct PolarCoord {
    float r, th;
};

struct State {
  PolarCoord icp;
  PolarCoord sfPos;
};

struct Input {
  PolarCoord step;
  int c_r;
};

struct Data {
    State state;
    Input input[N_INPUT];
    int n;
};
////////////////////////////////////////////////////////////////////////////////

void linspace(float result[], float min, float max, int n);
void initializing(Data *dataSet,
                  float cpR[], float cpTh[], float stepR[], float stepTh[]);
void writeData(Data* data, std::string str);


__global__ void step_1(Data *dataSet);
__global__ void step_N(Data *dataSet, int n_step,
                       float *cpR, float *cpTh, float *stepR, float *stepTh);
__device__ State stepping (State p0, PolarCoord u);
__device__ bool isZeroStepCapt(State p);
__device__ float distanceTwoPolar (PolarCoord a, PolarCoord b);
__device__ bool isInPrevSet(Data *dataSet, State p, int n_step,
                            float cpR[], float cpTh[],
                            float stepR[], float stepTh[]);
__device__ void setBound(State *bound, State p,
                         float cpR[], float cpTh[],
                         float stepR[], float stepTh[]);


#else
#endif
