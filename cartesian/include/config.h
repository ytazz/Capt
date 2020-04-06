#ifndef __CONFIG_H__
#define __CONFIG_H__

#include "loader.h"
#include <math.h>

namespace ConfigEnum {

enum Element {
  CONFIG_ELE_NONE,
  UNIT,
  SIMULATION,
  LOCAL,
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
  void read(float *val, const std::string &name);

  void print();

private:
  ConfigEnum::Element element;

  // unit
  float unit_length; // [m]
  float unit_mass;   // [kg]
  float unit_time;   // [s]
  float unit_angle;  // [rad]

  // parameter
  float  timestep;
  int    preview;
};

} // namespace Capt

#endif // __CONFIG_H__