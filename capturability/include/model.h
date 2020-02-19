#ifndef __MODEL_H__
#define __MODEL_H__

#include "loader.h"
#include "polygon.h"
#include "base.h"
#include <iostream>
#include <math.h>
#include <string.h>
#include <vector>

namespace Capt {

enum ModelElement {
  MODEL_ELE_NONE,
  MODEL_ELE_ROBOT,
  MODEL_ELE_UNIT,
  MODEL_ELE_PHYSICS,
  MODEL_ELE_ENVIRONMENT,
  MODEL_ELE_FOOT,
  MODEL_ELE_SHAPE
};

enum ShapeElement {
  SHAPE_NONE,
  SHAPE_POLYGON,
  SHAPE_REVERSE
};

class Model : public Loader {

public:
  explicit Model(const std::string &name);
  virtual ~Model();

  void callbackElement(const std::string &name, const bool is_start) override;
  void callbackAttribute(const std::string &name, const std::string &value) override;

  void read(std::string *val, const std::string &name);
  void read(int *val, const std::string &name);
  void read(double *val, const std::string &name);
  void read(arr2_t *val, const std::string &name);
  void read(arr2_t *val, const std::string &name, vec2_t offset);

  void print();

private:
  ModelElement element;
  ShapeElement shape;
  Foot         foot;

  Polygon polygon;

  std::string robot_name;
  double      unit_length, unit_mass, unit_time;
  double      mass, com_height, step_time_min, foot_vel_max, step_height;
  double      gravity, friction;

  arr2_t foot_r, foot_l;
  arr2_t foot_r_convex, foot_l_convex;
};

} // namespace Capt

#endif // __MODEL_H__
