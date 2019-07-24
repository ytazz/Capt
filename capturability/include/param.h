#ifndef __PARAM_H__
#define __PARAM_H__

#include "loader.h"
#include <math.h>

namespace CA {

namespace Pa {
enum ParamElement {
  NOELEMENT,
  COORDINATE,
  UNIT,
  ICP,
  SWING,
};

enum Coordinate { NOCOORD, POLAR, CARTESIAN };

enum Axis { NOAXIS, RADIUS, ANGLE, X, Y };
} // namespace Pa

class Param : public Loader {

public:
  explicit Param(const std::string &name = "");
  ~Param();

  void callbackElement(const std::string &name, const bool is_start) override;
  void callbackAttribute(const std::string &name,
                         const std::string &value) override;

  float getVal(const char *element_name, const char *attribute_name);
  std::string getStr(const char *element_name, const char *attribute_name);
  void print();

private:
  int round(float value);
  void calcNum();

  Pa::ParamElement element;
  Pa::Coordinate coordinate;
  Pa::Axis axis;

  // unit
  float unit_length, unit_angle;
  // number
  int icp_r_num;
  int icp_th_num;
  int swft_r_num;
  int swft_th_num;
  // polar
  float icp_r_min, icp_r_max, icp_r_step;
  float icp_th_min, icp_th_max, icp_th_step;
  float swft_r_min, swft_r_max, swft_r_step;
  float swft_th_min, swft_th_max, swft_th_step;
  // cartesian
  float icp_x_min, icp_x_max, icp_x_step;
  float icp_y_min, icp_y_max, icp_y_step;
  float swft_x_min, swft_x_max, swft_x_step;
  float swft_y_min, swft_y_max, swft_y_step;

  float pi;
};

} // namespace CA

#endif // __PARAM_H__