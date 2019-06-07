#ifndef __COM_ESTIMATOR_H__
#define __COM_ESTIMATOR_H__

#include <iostream>

class ComEstimator {
public:
  ComEstimator(Model model);
  ~ComEstimator();

  std::vector<float> getCom(std::vector<float> joint_angle);

private:
  Model model;
};

#endif // __COM_ESTIMATOR_H__