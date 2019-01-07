/*
    author G. KIM
*/


#ifndef STEP_MODIFIER_H
#define STEP_MODIFIER_H

#include <DataStruct.h>
#include <vector>
#include <string>
#include <algorithm>

#include "Gnuplot.h"

class CRplot {
    gnuplot       p;

public:
    CRplot();
    ~CRplot();
    void plot(nkk::State current_state, std::vector<nkk::Input> captureRegion);
};

#endif  // STEP_MODIFIER_H
