#ifndef __MODEL_H__
#define __MODEL_H__

#include "loader.h"
#include "vector.h"
#include <iostream>
#include <math.h>
#include <string.h>
#include <vector>

namespace CA {

namespace Mo {
enum ModelElement { NOELEMENT, ROBOT, UNIT, PHYSICS, LINK, SHAPE };

enum Foot { NOFOOT, RFOOT, LFOOT };

enum Shape { NOSHAPE, BOX, POLYGON, CIRCLE, REVERSE };
} // namespace Mo

class Model : public Loader {

public:
  explicit Model(const std::string &name);
  ~Model();

  void callbackElement(const std::string &name, const bool is_start) override;
  void callbackAttribute(const std::string &name,
                         const std::string &value) override;

  std::vector<Vector2> reverseShape(std::vector<Vector2> points);

  float getVal(const char *element_name, const char *attribute_name);
  std::string getStr(const char *element_name, const char *attribute_name);
  std::vector<Vector2> getVec(const char *element_name,
                              const char *attribute_name);
  void print();

private:
  Mo::ModelElement element;
  Mo::Foot foot;
  Mo::Shape shape;

  const float pi;

  std::string robot_name;
  float unit_length, unit_mass, unit_time, unit_angle;
  float mass, com_height, step_time_min, foot_vel_max;

  std::vector<Vector2> foot_r, foot_l;

  Vector2 trn;
  float rot;
};

} // namespace CA

#endif // __MODEL_H__
