#include "config.h"

namespace Capt {

Config::Config(const std::string &name) : Loader(name) {
  printf("\x1b[36mConfig File: %s\x1b[39m\n", name.c_str() );

  element = ConfigEnum::CONFIG_ELE_NONE;

  unit_length = 1.0;
  unit_mass   = 1.0;
  unit_time   = 1.0;
  unit_angle  = 1.0;

  parse();
}

Config::~Config() {
}

void Config::callbackElement(const std::string &name, const bool is_start) {
  if (is_start) {
    switch (element) {
    case ConfigEnum::CONFIG_ELE_NONE:
      if (equalStr(name, "unit") )
        element = ConfigEnum::UNIT;
      if (equalStr(name, "simulation") )
        element = ConfigEnum::SIMULATION;
      if (equalStr(name, "local") )
        element = ConfigEnum::LOCAL;
      if (equalStr(name, "control") )
        element = ConfigEnum::CONTROL;
      break;
    default:
      break;
    }
  } else {
    switch (element) {
    case ConfigEnum::UNIT:
    case ConfigEnum::SIMULATION:
    case ConfigEnum::LOCAL:
    case ConfigEnum::CONTROL:
      element = ConfigEnum::CONFIG_ELE_NONE;
      break;
    default:
      break;
    }
  }
}

void Config::callbackAttribute(const std::string &name,
                               const std::string &value) {
  switch (element) {
  case ConfigEnum::UNIT:
    if (equalStr(name, "length") ) {
      if (equalStr(value, "m") )
        unit_length = 1.0;
      if (equalStr(value, "cm") )
        unit_length = 1.0 / 100.0;
      if (equalStr(value, "mm") )
        unit_length = 1.0 / 1000.0;
    }
    if (equalStr(name, "mass") ) {
      if (equalStr(value, "kg") )
        unit_mass = 1.0;
      if (equalStr(value, "g") )
        unit_mass = 1.0 / 1000.0;
    }
    if (equalStr(name, "time") ) {
      if (equalStr(value, "s") )
        unit_time = 1.0;
      if (equalStr(value, "ms") )
        unit_time = 1.0 / 1000.0;
    }
    if (equalStr(name, "angle") ) {
      if (equalStr(value, "rad") )
        unit_angle = 1.0;
      if (equalStr(value, "deg") )
        unit_angle = M_PI / 180.0;
    }
    break;
  case ConfigEnum::SIMULATION:
    if (equalStr(name, "timestep") )
      timestep = std::stof(value) * unit_time;
    break;
  case ConfigEnum::LOCAL:
    if (equalStr(name, "preview") )
      preview = std::stof(value) * unit_length;
    break;
  default:
    break;
  }
}

void Config::read(int *val, const std::string &name) {
}

void Config::read(double *val, const std::string &name){
  if (equalStr(name, "timestep") )
    *val = timestep;
  if (equalStr(name, "preview") )
    *val = preview;
}

void Config::print() {
  printf("-------------------------------------------\n");

  printf("unit:\n");
  printf("\tlength: %lf [m]\n", unit_length);
  printf("\tmass  : %lf [kg]\n", unit_mass);
  printf("\ttime  : %lf [s]\n", unit_time);
  printf("\tangle : %lf [rad]\n", unit_angle);

  printf("simulation:\n");
  printf("\ttimestep: %lf\n", timestep);

  printf("local:\n");
  printf("\tpreview : %lf\n", preview );

  printf("-------------------------------------------\n");
}

} // namespace Capt