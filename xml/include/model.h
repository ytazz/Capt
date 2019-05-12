#ifndef __MODEL__
#define __MODEL__

#include "loader.h"
#include "vector.h"
#include <iostream>
#include <math.h>
#include <string.h>
#include <vector>

namespace CA {

enum Element { NOELEMENT, ROBOT, UNIT, PHYSICS, LINK, SHAPE };

enum Foot { NOFOOT, RFOOT, LFOOT };

enum Shape { NOSHAPE, BOX, POLYGON, CIRCLE, REVERSE };

class Model : public Loader {

public:
  explicit Model(const std::string &name);
  ~Model();

  void callbackElement(const std::string &name, const bool is_start) override;
  void callbackAttribute(const std::string &name,
                         const std::string &value) override;

  bool equalStr(const char *chr1, const char *chr2);
  bool equalStr(const std::string &str1, const char *chr2);
  bool equalStr(const char *chr1, const std::string &str2);
  bool equalStr(const std::string &str1, const std::string &str2);
  Vector2 convertStrToVec(const std::string &str);
  std::vector<Vector2> reverseShape(std::vector<Vector2> points);

  void print();

private:
  Element element;
  Foot foot;
  Shape shape;

  const float pi;

  std::string robot_name;
  float unit_length, unit_mass, unit_time, unit_angle;
  float mass, com_height, step_time_min, foot_vel_max;

  std::vector<Vector2> foot_r, foot_l;

  Vector2 trn;
  float rot;
};

} // namespace CA

#endif // __MODEL__
