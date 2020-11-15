#pragma once

#include "loader.h"
#include <math.h>

namespace CaptEnum {

enum ParamElement {
  SWING,
  EXCEPT,
  COP,
  ICP,
  GRID,
};

enum Axis { AXIS_X, AXIS_Y, AXIS_Z, AXIS_T };

} // namespace CaptEnum

namespace Capt {

class Param : public Loader {

public:
  explicit Param(const std::string &name = "");
  virtual ~Param();

  void callbackElement(const std::string &name, const bool is_start) override;
  void callbackAttribute(const std::string &name, const std::string &value) override;

  void read(float *val, const std::string &name);

  void print();

private:
  CaptEnum::ParamElement element;
  CaptEnum::Axis         axis;

  float swf_x_min, swf_x_max;
  float swf_y_min, swf_y_max;
  float swf_z_min, swf_z_max;
  float exc_x_min, exc_x_max;
  float exc_y_min, exc_y_max;
  float cop_x_min, cop_x_max;
  float cop_y_min, cop_y_max;
  float icp_x_min, icp_x_max;
  float icp_y_min, icp_y_max;

  float grid_x_min, grid_x_max, grid_x_stp;
  float grid_y_min, grid_y_max, grid_y_stp;
  float grid_z_min, grid_z_max, grid_z_stp;
  float grid_t_min, grid_t_max, grid_t_stp;
};

} // namespace Capt
