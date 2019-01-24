/*
    author G. KIM
*/


#ifndef STEP_MODIFIER_H
#define STEP_MODIFIER_H

#include <BalanceMonitor.h>
#include <CA.h>

#include "Gnuplot.h"

#include <algorithm>
#include <ctime>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>


class CRplot {

public:
    CRplot();
    ~CRplot();
    void plot(CAstate current_state, std::vector<CAinput> captureRegion);

private:
    void drawReachableRegion();
    gnuplot p;
};

#endif  // STEP_MODIFIER_H
