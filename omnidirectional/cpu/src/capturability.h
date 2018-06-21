#ifndef __capturability_H__
#define __capturability_H__

#include <stdio.h>
#include <vector>
#include <math.h>

using namespace std;

#define sq(x) ((x) * (x))

struct PolarCoord {
  float r, th;
};

struct States {
  PolarCoord step;
  PolarCoord cp;
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

const int numGrid = 50;
const long int N = (long int)numGrid*numGrid*numGrid*numGrid;

const int threadsPerBlock = 1024;
const int blocksPerGrid = 1024;

const PolarCoord CP_MAX = {0.5, 2*PI};
const PolarCoord CP_MIN = {0.0, 0.0};
const PolarCoord FOOT_MAX = {0.5, PI};
const PolarCoord FOOT_MIN = {2*FOOTSIZE, 0};
////////////////////////////////////////////////////////////////////////////////

void linspace(float result[], float min, float max);

void makeStatesSpace(States* result,
                        float cpR[], float cpTh[], float stepR[], float stepTh[]);

void makeInputSpace(PolarCoord* result, float stepR[], float stepTh[]);

void writeFile(States* data);

void step_1(States *result_set, States *set, PolarCoord *input );

States oneStepAfter (States p0, PolarCoord u);

bool isZeroStepCapt(States p);

float distanceTwoPolar (PolarCoord a, PolarCoord b);

#else
#endif
