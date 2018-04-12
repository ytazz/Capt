/**
   \author GWANWOO KIM
 */
#include "robotParam.h"
#include "capturability.h"
#include "tools.h"

const int numGrid = 100;

using namespace std;

struct RobotsParam {
    const float heightOfCOM = 0.50;
    const float omega = sqrt(9.81/heightOfCOM);

    const float step_max[2] = {0.3, 0.3};
    const float step_min[2] = {-0.3, -0.3};

    const float cp_max[2] = {0.2, 0.2};
    const float cp_min[2] = {-0.2, -0.2};
};

void setStateSpace( float *stateSpace, float *cp_max,
                    float *cp_min, float *step_max, float *step_min) {
    vector<float>
}



int main (void) {
    return 0;
}
