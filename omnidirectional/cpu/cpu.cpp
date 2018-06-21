#include "src/capturability.h"

int main(void)
{
    float cpR[numGrid], cpTh[numGrid], stepR[numGrid], stepTh[numGrid];
    linspace(cpR, CP_MIN.r, CP_MAX.r);
    linspace(cpTh, CP_MIN.th, CP_MAX.th);
    linspace(stepR, FOOT_MIN.r, FOOT_MAX.r);
    linspace(stepTh, FOOT_MIN.th, FOOT_MAX.th);

    States *statesSpace = new States[N];
    makeStatesSpace(statesSpace, cpR, cpTh, stepR, stepTh);

    PolarCoord *inputSpace = new PolarCoord[numGrid*numGrid];
    makeInputSpace(inputSpace, stepR, stepTh);

    States temp;
    std::vector<States> oneStep;
    for (size_t i = 0; i < N; i++) {
      for (size_t j = 0; j < numGrid * numGrid; j++) {
        temp = oneStepAfter(statesSpace[i], inputSpace[j]);
      }
    }




    delete [] statesSpace;
    delete [] inputSpace;

    return 0;

}
