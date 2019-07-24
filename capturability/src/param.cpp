#include "param.h"

namespace CA {

Param::Param(const std::string &name) : Loader(name), pi(M_PI) {
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

Param::~Param() {}

void Param::callbackElement(const std::string &name, const bool is_start) {
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
      calcNum();
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

void Param::callbackAttribute(const std::string &name,
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

std::string Param::getStr(const char *element_name,
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

float Param::getVal(const char *element_name, const char *attribute_name) {
  float val = 0.0;
  if (equalStr(element_name, "icp_r")) {
    if (equalStr(attribute_name, "min"))
      val = icp_r_min;
    if (equalStr(attribute_name, "max"))
      val = icp_r_max;
    if (equalStr(attribute_name, "step"))
      val = icp_r_step;
    if (equalStr(attribute_name, "num"))
      val = icp_r_num;
  }
  if (equalStr(element_name, "icp_th")) {
    if (equalStr(attribute_name, "min"))
      val = icp_th_min;
    if (equalStr(attribute_name, "max"))
      val = icp_th_max;
    if (equalStr(attribute_name, "step"))
      val = icp_th_step;
    if (equalStr(attribute_name, "num"))
      val = icp_th_num;
  }
  if (equalStr(element_name, "swft_r")) {
    if (equalStr(attribute_name, "min"))
      val = swft_r_min;
    if (equalStr(attribute_name, "max"))
      val = swft_r_max;
    if (equalStr(attribute_name, "step"))
      val = swft_r_step;
    if (equalStr(attribute_name, "num"))
      val = swft_r_num;
  }
  if (equalStr(element_name, "swft_th")) {
    if (equalStr(attribute_name, "min"))
      val = swft_th_min;
    if (equalStr(attribute_name, "max"))
      val = swft_th_max;
    if (equalStr(attribute_name, "step"))
      val = swft_th_step;
    if (equalStr(attribute_name, "num"))
      val = swft_th_num;
  }
  return val;
}

int Param::round(float value) {
  int result = (int)value;

  float decimal = value - (int)value;
  if (decimal >= 0.5) {
    result += 1;
  }

  return result;
}

void Param::calcNum() {
  icp_r_num = round((icp_r_max - icp_r_min) / icp_r_step) + 1;
  icp_th_num = round((icp_th_max - icp_th_min) / icp_th_step) + 1;
  swft_r_num = round((swft_r_max - swft_r_min) / swft_r_step) + 1;
  swft_th_num = round((swft_th_max - swft_th_min) / swft_th_step) + 1;
}

void Param::print() {
  using namespace Pa;
  printf("-------------------------------------------\n");
  printf("coordinate:\n");
  if (coordinate == POLAR)
    printf("\ttype: %s\n", "polar");
  if (coordinate == CARTESIAN)
    printf("\ttype: %s\n", "cartesian");

  printf("unit:\n");
  printf("\tlength: %lf\n", unit_length);
  printf("\tangle : %lf\n", unit_angle);

  printf("icp:\n");
  printf("\tradius:\n");
  printf("\t\tmin : %lf\n", icp_r_min);
  printf("\t\tmax : %lf\n", icp_r_max);
  printf("\t\tstep: %lf\n", icp_r_step);
  printf("\t\tnum : %d\n", icp_r_num);
  printf("\tangle:\n");
  printf("\t\tmin : %lf\n", icp_th_min);
  printf("\t\tmax : %lf\n", icp_th_max);
  printf("\t\tstep: %lf\n", icp_th_step);
  printf("\t\tnum : %d\n", icp_th_num);

  printf("swing:\n");
  printf("\tradius:\n");
  printf("\t\tmin : %lf\n", swft_r_min);
  printf("\t\tmax : %lf\n", swft_r_max);
  printf("\t\tstep: %lf\n", swft_r_step);
  printf("\t\tnum : %d\n", swft_r_num);
  printf("\tangle:\n");
  printf("\t\tmin : %lf\n", swft_th_min);
  printf("\t\tmax : %lf\n", swft_th_max);
  printf("\t\tstep: %lf\n", swft_th_step);
  printf("\t\tnum : %d\n", swft_th_num);
  printf("-------------------------------------------\n");
}

} // namespace CA