#ifndef __CONFIG_H__
#define __CONFIG_H__

#include "loader.h"
#include <math.h>

namespace ConfigEnum {

enum Element {
  CONFIG_ELE_NONE,
  UNIT,
  SIMULATION,
  CONTROL
};

}

namespace Capt {

class Config : public Loader {

public:
  explicit Config(const std::string &name = "");
  virtual ~Config();

  void callbackElement(const std::string &name, const bool is_start) override;
  void callbackAttribute(const std::string &name, const std::string &value) override;

  void read(int *val, const std::string &name);
  void read(double *val, const std::string &name);

  void print();

private:
  ConfigEnum::Element element;

  // unit
  double unit_length; // [m]
  double unit_mass;   // [kg]
  double unit_time;   // [s]
  double unit_angle;  // [rad]

  // parameter
  double timestep;
};

} // namespace Capt

#endif // __CONFIG_H__