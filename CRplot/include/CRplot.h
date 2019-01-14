/*
    author G. KIM
*/


#ifndef STEP_MODIFIER_H
#define STEP_MODIFIER_H

#include <BalanceMonitor.h>
#include <CA.h>

#include "Gnuplot.h"

#include <string>
#include <vector>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <unistd.h>


class CRplot {
    gnuplot p;

public:
    CRplot();
    ~CRplot();
    void plot(CAstate current_state, std::vector<CAinput> captureRegion);
};

#endif  // STEP_MODIFIER_H
