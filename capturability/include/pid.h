#ifndef __PID_H__
#define __PID_H__

#include <cnoid/BodyItem>
#include <iostream>
#include <stdio.h>
#include <vector>

class PID {
public:
  PID();
  ~PID();

  void setTimestep(float dt);
  void setGain(float k_p, float k_i, float k_d);
  void setLimit(float u_min, float u_max);
  void init();

  void showParam();
  void control(cnoid::BodyPtr ioBody, std::vector<float> qref);

private:
  float dt, t;
  float k_p, k_i, k_d;
  float u_min, u_max;

  cnoid::BodyPtr ioBody;
  int num_joint;

  std::vector<float> qold;
  std::vector<float> qrefold;
  std::vector<float> integral;
};

#endif // __PID_H__
