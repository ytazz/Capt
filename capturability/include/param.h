#ifndef __PARAM_H__
#define __PARAM_H__

#include "loader.h"
#include <math.h>

namespace Capt {

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

  float       getVal(const char *element_name, const char *attribute_name);
  std::string getStr(const char *element_name, const char *attribute_name);
  void        print();

private:
  int  round(float value);
  void calcNum();

  Pa::ParamElement element;
  Pa::Coordinate   coordinate;
  Pa::Axis         axis;

  // unit
  float unit_length, unit_angle;
  // number
  int icp_x_num;
  int icp_y_num;
  int swf_x_num;
  int swf_y_num;
  int icp_r_num;
  int icp_th_num;
  int swf_r_num;
  int swf_th_num;
  // polar
  float icp_r_min, icp_r_max, icp_r_step;
  float icp_th_min, icp_th_max, icp_th_step;
  float swf_r_min, swf_r_max, swf_r_step;
  float swf_th_min, swf_th_max, swf_th_step;
  // cartesian
  float icp_x_min, icp_x_max, icp_x_step;
  float icp_y_min, icp_y_max, icp_y_step;
  float swf_x_min, swf_x_max, swf_x_step;
  float swf_y_min, swf_y_max, swf_y_step;

  float pi;
};

} // namespace Capt

#endif // __PARAM_H__