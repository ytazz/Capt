#include "param.h"

namespace Capt {

Param::Param(const std::string &name) : Loader(name) {
  printf("\x1b[36mParam File: %s\x1b[39m\n", name.c_str() );

  swf_x_min = swf_x_max = 0.0;
  swf_y_min = swf_y_max = 0.0;
  swf_z_min = swf_z_max = 0.0;

  exc_x_min = exc_x_max = 0.0;
  exc_y_min = exc_y_max = 0.0;

  cop_x_min = cop_x_max = 0.0;
  cop_y_min = cop_y_max = 0.0;

  grid_x_min = grid_x_max = grid_x_stp = 0.0;
  grid_y_min = grid_y_max = grid_y_stp = 0.0;
  grid_z_min = grid_z_max = grid_z_stp = 0.0;
  grid_t_min = grid_t_max = grid_t_stp = 0.0;

  parse();
}

Param::~Param() {
}

void Param::callbackElement(const std::string &name, const bool is_start) {
  //printf("callback %d, %s\n", (int)is_start, name.c_str());
  using namespace CaptEnum;
  if (is_start) {
    if (equalStr(name, "swing" )) element = SWING;
    if (equalStr(name, "except")) element = EXCEPT;
    if (equalStr(name, "cop"   )) element = COP;
    if (equalStr(name, "grid"  )) element = GRID;
    if (equalStr(name, "x"     )) axis    = AXIS_X;
    if (equalStr(name, "y"     )) axis    = AXIS_Y;
    if (equalStr(name, "z"     )) axis    = AXIS_Z;
    if (equalStr(name, "t"     )) axis    = AXIS_T;
  }
}

void Param::callbackAttribute(const std::string &name, const std::string &value) {
  //printf("callback %s, %s\n", name.c_str(), value.c_str());
  using namespace CaptEnum;
  switch (element) {
    case SWING:
      if (axis == AXIS_X) {
        if (equalStr(name, "min") ) swf_x_min = std::stof(value);
        if (equalStr(name, "max") ) swf_x_max = std::stof(value);
      }
      if (axis == AXIS_Y) {
        if (equalStr(name, "min") ) swf_y_min = std::stof(value);
        if (equalStr(name, "max") ) swf_y_max = std::stof(value);
      }
      if (axis == AXIS_Z) {
        if (equalStr(name, "min") ) swf_z_min = std::stof(value);
        if (equalStr(name, "max") ) swf_z_max = std::stof(value);
      }
      break;
    case EXCEPT:
      if (axis == AXIS_X) {
        if (equalStr(name, "min") ) exc_x_min = std::stof(value);
        if (equalStr(name, "max") ) exc_x_max = std::stof(value);
      }
      if (axis == AXIS_Y) {
        if (equalStr(name, "min") ) exc_y_min = std::stof(value);
        if (equalStr(name, "max") ) exc_y_max = std::stof(value);
      }
      break;
    case COP:
      if (axis == AXIS_X) {
        if (equalStr(name, "min") ) cop_x_min = std::stof(value);
        if (equalStr(name, "max") ) cop_x_max = std::stof(value);
      }
      if (axis == AXIS_Y) {
        if (equalStr(name, "min") ) cop_y_min = std::stof(value);
        if (equalStr(name, "max") ) cop_y_max = std::stof(value);
      }
      break;
    case GRID:
      if (axis == AXIS_X) {
        if (equalStr(name, "min") ) grid_x_min = std::stof(value);
        if (equalStr(name, "max") ) grid_x_max = std::stof(value);
        if (equalStr(name, "stp") ) grid_x_stp = std::stof(value);
      }
      if (axis == AXIS_Y) {
        if (equalStr(name, "min") ) grid_y_min = std::stof(value);
        if (equalStr(name, "max") ) grid_y_max = std::stof(value);
        if (equalStr(name, "stp") ) grid_y_stp = std::stof(value);
      }
      if (axis == AXIS_Z) {
        if (equalStr(name, "min") ) grid_z_min = std::stof(value);
        if (equalStr(name, "max") ) grid_z_max = std::stof(value);
        if (equalStr(name, "stp") ) grid_z_stp = std::stof(value);
      }
      if (axis == AXIS_T) {
        if (equalStr(name, "min") ) grid_t_min = std::stof(value);
        if (equalStr(name, "max") ) grid_t_max = std::stof(value);
        if (equalStr(name, "stp") ) grid_t_stp = std::stof(value);
      }
      break;
    default:
      break;
  }
}

void Param::read(float *val, const std::string &name){
  if (equalStr(name, "swf_x_min") ) *val = swf_x_min;
  if (equalStr(name, "swf_x_max") ) *val = swf_x_max;
  if (equalStr(name, "swf_y_min") ) *val = swf_y_min;
  if (equalStr(name, "swf_y_max") ) *val = swf_y_max;
  if (equalStr(name, "swf_z_min") ) *val = swf_z_min;
  if (equalStr(name, "swf_z_max") ) *val = swf_z_max;

  if (equalStr(name, "exc_x_min") ) *val = exc_x_min;
  if (equalStr(name, "exc_x_max") ) *val = exc_x_max;
  if (equalStr(name, "exc_y_min") ) *val = exc_y_min;
  if (equalStr(name, "exc_y_max") ) *val = exc_y_max;

  if (equalStr(name, "cop_x_min") ) *val = cop_x_min;
  if (equalStr(name, "cop_x_max") ) *val = cop_x_max;
  if (equalStr(name, "cop_y_min") ) *val = cop_y_min;
  if (equalStr(name, "cop_y_max") ) *val = cop_y_max;

  if (equalStr(name, "grid_x_min") ) *val = grid_x_min;
  if (equalStr(name, "grid_x_max") ) *val = grid_x_max;
  if (equalStr(name, "grid_x_stp") ) *val = grid_x_stp;
  if (equalStr(name, "grid_y_min") ) *val = grid_y_min;
  if (equalStr(name, "grid_y_max") ) *val = grid_y_max;
  if (equalStr(name, "grid_y_stp") ) *val = grid_y_stp;
  if (equalStr(name, "grid_z_min") ) *val = grid_z_min;
  if (equalStr(name, "grid_z_max") ) *val = grid_z_max;
  if (equalStr(name, "grid_z_stp") ) *val = grid_z_stp;
  if (equalStr(name, "grid_t_min") ) *val = grid_t_min;
  if (equalStr(name, "grid_t_max") ) *val = grid_t_max;
  if (equalStr(name, "grid_t_stp") ) *val = grid_t_stp;
}

} // namespace Capt