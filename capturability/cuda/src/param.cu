#include "param.cuh"

namespace GPGPU {

__device__ Param::Param(const std::string &name) : Loader(name), pi(M_PI) {
  element = Pa::NOELEMENT;
  coordinate = Pa::NOCOORD;
  axis = Pa::NOAXIS;

  unit_length = 0.0;
  unit_angle = 0.0;

  icp_r_min = 0.0;
  icp_r_max = 0.0;
  icp_r_step = 0.0;
  icp_th_min = 0.0;
  icp_th_max = 0.0;
  icp_th_step = 0.0;
  swft_r_min = 0.0;
  swft_r_max = 0.0;
  swft_r_step = 0.0;
  swft_th_min = 0.0;
  swft_th_max = 0.0;
  swft_th_step = 0.0;

  icp_x_min = 0.0;
  icp_x_max = 0.0;
  icp_x_step = 0.0;
  icp_y_min = 0.0;
  icp_y_max = 0.0;
  icp_y_step = 0.0;
  swft_x_min = 0.0;
  swft_x_max = 0.0;
  swft_x_step = 0.0;
  swft_y_min = 0.0;
  swft_y_max = 0.0;
  swft_y_step = 0.0;

  parse();
}

__device__ Param::~Param() {}

__device__ void Param::callbackElement(const std::string &name,
                                       const bool is_start) {
  using namespace Pa;
  if (is_start) {
    switch (element) {
    case NOELEMENT:
      if (equalStr(name, "coordinate"))
        element = COORDINATE;
      break;
    case COORDINATE:
      if (equalStr(name, "unit"))
        element = UNIT;
      if (equalStr(name, "icp"))
        element = ICP;
      if (equalStr(name, "swing"))
        element = SWING;
      break;
    case ICP:
      if (equalStr(name, "radius"))
        axis = RADIUS;
      if (equalStr(name, "angle"))
        axis = ANGLE;
      break;
    case SWING:
      if (equalStr(name, "radius"))
        axis = RADIUS;
      if (equalStr(name, "angle"))
        axis = ANGLE;
      break;
    default:
      break;
    }
  } else {
    switch (element) {
    case COORDINATE:
      element = NOELEMENT;
      break;
    case UNIT:
      element = COORDINATE;
      axis = NOAXIS;
      break;
    case ICP:
    case SWING:
      if (axis != NOAXIS)
        axis = NOAXIS;
      else
        element = COORDINATE;
      break;
    default:
      break;
    }
  }
}

__device__ void Param::callbackAttribute(const std::string &name,
                                         const std::string &value) {
  using namespace Pa;
  switch (element) {
  case COORDINATE:
    if (equalStr(name, "type")) {
      if (equalStr(value, "polar"))
        coordinate = POLAR;
      if (equalStr(value, "cartesian"))
        coordinate = CARTESIAN;
    }
    break;
  case UNIT:
    if (equalStr(name, "length")) {
      if (equalStr(value, "m"))
        unit_length = 1.0;
      if (equalStr(value, "cm"))
        unit_length = 1.0 / 100.0;
      if (equalStr(value, "mm"))
        unit_length = 1.0 / 1000.0;
    }
    if (equalStr(name, "angle")) {
      if (equalStr(value, "rad"))
        unit_angle = 1.0;
      if (equalStr(value, "deg"))
        unit_angle = pi / 180.0;
    }
    break;
  case ICP:
    if (coordinate == POLAR) {
      if (axis == RADIUS) {
        if (equalStr(name, "min"))
          icp_r_min = std::stof(value) * unit_length;
        if (equalStr(name, "max"))
          icp_r_max = std::stof(value) * unit_length;
        if (equalStr(name, "step"))
          icp_r_step = std::stof(value) * unit_length;
      }
      if (axis == ANGLE) {
        if (equalStr(name, "min"))
          icp_th_min = std::stof(value) * unit_angle;
        if (equalStr(name, "max"))
          icp_th_max = std::stof(value) * unit_angle;
        if (equalStr(name, "step"))
          icp_th_step = std::stof(value) * unit_angle;
      }
    }
    break;
  case SWING:
    if (coordinate == POLAR) {
      if (axis == RADIUS) {
        if (equalStr(name, "min"))
          swft_r_min = std::stof(value) * unit_length;
        if (equalStr(name, "max"))
          swft_r_max = std::stof(value) * unit_length;
        if (equalStr(name, "step"))
          swft_r_step = std::stof(value) * unit_length;
      }
      if (axis == ANGLE) {
        if (equalStr(name, "min"))
          swft_th_min = std::stof(value) * unit_angle;
        if (equalStr(name, "max"))
          swft_th_max = std::stof(value) * unit_angle;
        if (equalStr(name, "step"))
          swft_th_step = std::stof(value) * unit_angle;
      }
    }
    break;
  default:
    break;
  }
}

__device__ std::string Param::getStr(const char *element_name,
                                     const char *attribute_name) {
  std::string str;
  using namespace Pa;
  if (equalStr(element_name, "coordinate")) {
    if (equalStr(attribute_name, "type")) {
      if (coordinate == POLAR)
        str = "polar";
      if (coordinate == CARTESIAN)
        str = "cartesian";
    }
  }
  return str;
}

__device__ float Param::getVal(const char *element_name,
                               const char *attribute_name) {
  float val = 0.0;
  if (equalStr(element_name, "icp_r")) {
    if (equalStr(attribute_name, "min"))
      val = icp_r_min;
    if (equalStr(attribute_name, "max"))
      val = icp_r_max;
    if (equalStr(attribute_name, "step"))
      val = icp_r_step;
  }
  if (equalStr(element_name, "icp_th")) {
    if (equalStr(attribute_name, "min"))
      val = icp_th_min;
    if (equalStr(attribute_name, "max"))
      val = icp_th_max;
    if (equalStr(attribute_name, "step"))
      val = icp_th_step;
  }
  if (equalStr(element_name, "swft_r")) {
    if (equalStr(attribute_name, "min"))
      val = swft_r_min;
    if (equalStr(attribute_name, "max"))
      val = swft_r_max;
    if (equalStr(attribute_name, "step"))
      val = swft_r_step;
  }
  if (equalStr(element_name, "swft_th")) {
    if (equalStr(attribute_name, "min"))
      val = swft_th_min;
    if (equalStr(attribute_name, "max"))
      val = swft_th_max;
    if (equalStr(attribute_name, "step"))
      val = swft_th_step;
  }
  return val;
}

} // namespace GPGPU