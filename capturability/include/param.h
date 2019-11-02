#ifndef __PARAM_H__
#define __PARAM_H__

#include "loader.h"
#include <math.h>

namespace CaptEnum {

enum ParamElement {
  PARAM_ELE_NONE,
  COORDINATE,
  UNIT,
  ICP,
  SWING,
  MAP
};

enum Coordinate { COORD_NONE, COORD_CARTESIAN };

enum Axis { AXIS_NONE, AXIS_X, AXIS_Y };

} // namespace CaptEnum

namespace Capt {

class Param : public Loader {

public:
  explicit Param(const std::string &name = "");
  ~Param();

  void callbackElement(const std::string &name, const bool is_start) override;
  void callbackAttribute(const std::string &name, const std::string &value) override;

  void read(std::string *val, const std::string &name);
  void read(int *val, const std::string &name);
  void read(double *val, const std::string &name);

  void print();

private:
  void calcNum();

  CaptEnum::ParamElement element;
  CaptEnum::Coordinate   coordinate;
  CaptEnum::Axis         axis;

  // unit
  double unit_length;
  // number
  int icp_x_num;
  int icp_y_num;
  int swf_x_num;
  int swf_y_num;
  int map_x_num;
  int map_y_num;
  // cartesian
  double icp_x_min, icp_x_max, icp_x_stp;
  double icp_y_min, icp_y_max, icp_y_stp;
  double swf_x_min, swf_x_max, swf_x_stp;
  double swf_y_min, swf_y_max, swf_y_stp;
  double map_x_min, map_x_max, map_x_stp;
  double map_y_min, map_y_max, map_y_stp;
};

} // namespace Capt

#endif // __PARAM_H__