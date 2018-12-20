#ifndef STEP_MODIFIER_H
#define STEP_MODIFIER_H

#include <stdlib.h>
#include <algorithm>
#include <math.h>
#include <string>
#include "../include/data_struct.h"
#include "../include/gnuplot.h"


using namespace std;

class StepModifier
{
  State current_state;
  PolarCoord current_swft_position;
  vector<Input> captureRegion;
  gnuplot p;

public:
  StepModifier();
  ~StepModifier();
  void initPlotting(string output = "file", string supportfoot = "right");
  void plotCR();
  void setCurrent(State state, PolarCoord swft_position);
  void readTable();
  State closestGridfrom(State a);
  float closest(std::vector<float> const& vec, float val);
  void findCaptureRegion(vector<Data> *d);
  float distTwoPolar(const PolarCoord &a, const PolarCoord &b);
  Input modifier();

private:
  std::vector<float> cp_r;
  std::vector<float> cp_th;
  std::vector<float> foot_r;
  std::vector<float> foot_th;
};

#endif // STEP_MODIFIER_H
