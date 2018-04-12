/**
   \author GWANWOO KIM
 */
#include "capturability.h"
#include "tools.h"

using namespace std;

// struct RobotsParam {
//     const float heightOfCOM = 0.50;
//     const float omega = sqrt(9.81/heightOfCOM);
//
//     const float step_max[2] = {0.3, 0.3};
//     const float step_min[2] = {-0.3, -0.3};
//
//     const float cp_max[2] = {0.2, 0.2};
//     const float cp_min[2] = {-0.2, -0.2};
//
//     const int numGrid = 100;
//
//     __device__ float take_a_step( float *currentSwFtPos, float *desiredSwFtPos) {
//     }
// };
//
// __global__ void kernel
//
//
// void setStateSpace( float *stateSpace, float *cp_max,
//                     float *cp_min, float *step_max, float *step_min)



int main (void) {
  vector<double> test;
  test = linspace(0.0, 2.0, 100);
  for(int i = 0; i< 100; i++){
    printf("%lf",test[i]);
  }
  return 0;
}
