#include "param.h"

namespace Capt {

Param::Param(const std::string &name) : Loader(name), pi(M_PI) {
  element    = Pa::NOELEMENT;
  coordinate = Pa::NOCOORD;
  axis       = Pa::NOAXIS;

  unit_length = 0.0;
  unit_angle  = 0.0;

  icp_r_min   = 0.0;
  icp_r_max   = 0.0;
  icp_r_step  = 0.0;
  icp_th_min  = 0.0;
  icp_th_max  = 0.0;
  icp_th_step = 0.0;
  swf_r_min   = 0.0;
  swf_r_max   = 0.0;
  swf_r_step  = 0.0;
  swf_th_min  = 0.0;
  swf_th_max  = 0.0;
  swf_th_step = 0.0;

  icp_x_min  = 0.0;
  icp_x_max  = 0.0;
  icp_x_step = 0.0;
  icp_y_min  = 0.0;
  icp_y_max  = 0.0;
  icp_y_step = 0.0;
  swf_x_min  = 0.0;
  swf_x_max  = 0.0;
  swf_x_step = 0.0;
  swf_y_min  = 0.0;
  swf_y_max  = 0.0;
  swf_y_step = 0.0;

  parse();
}

Param::~Param() {
}

void Param::callbackElement(const std::string &name, const bool is_start) {
  using namespace Pa;
  if (is_start) {
  switch (element) {
  case NOELEMENT:
    if (equalStr(name, "coordinate") )
      element = COORDINATE;
    break;
  case COORDINATE:
    if (equalStr(name, "unit") )
      element = UNIT;
    if (equalStr(name, "icp") )
      element = ICP;
    if (equalStr(name, "swing") )
      element = SWING;
    break;
  case ICP:
    if (equalStr(name, "x") )
      axis = X;
    if (equalStr(name, "y") )
      axis = Y;
    if (equalStr(name, "radius") )
      axis = RADIUS;
    if (equalStr(name, "angle") )
      axis = ANGLE;
    break;
  case SWING:
    if (equalStr(name, "x") )
      axis = X;
    if (equalStr(name, "y") )
      axis = Y;
    if (equalStr(name, "radius") )
      axis = RADIUS;
    if (equalStr(name, "angle") )
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
      axis    = NOAXIS;
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
  if (equalStr(name, "type") ) {
    if (equalStr(value, "polar") )
      coordinate = POLAR;
    if (equalStr(value, "cartesian") )
      coordinate = CARTESIAN;
  }
  break;
case UNIT:
  if (equalStr(name, "length") ) {
    if (equalStr(value, "m") )
      unit_length = 1.0;
    if (equalStr(value, "cm") )
      unit_length = 1.0 / 100.0;
    if (equalStr(value, "mm") )
      unit_length = 1.0 / 1000.0;
  }
  if (equalStr(name, "angle") ) {
    if (equalStr(value, "rad") )
      unit_angle = 1.0;
    if (equalStr(value, "deg") )
      unit_angle = pi / 180.0;
  }
  break;
case ICP:
  if (coordinate == CARTESIAN) {
    if (axis == X) {
      if (equalStr(name, "min") )
        icp_x_min = std::stof(value) * unit_length;
      if (equalStr(name, "max") )
        icp_x_max = std::stof(value) * unit_length;
      if (equalStr(name, "step") )
        icp_x_step = std::stof(value) * unit_length;
    }
    if (axis == Y) {
      if (equalStr(name, "min") )
        icp_y_min = std::stof(value) * unit_length;
      if (equalStr(name, "max") )
        icp_y_max = std::stof(value) * unit_length;
      if (equalStr(name, "step") )
        icp_y_step = std::stof(value) * unit_length;
    }
  }
  if (coordinate == POLAR) {
    if (axis == RADIUS) {
      if (equalStr(name, "min") )
        icp_r_min = std::stof(value) * unit_length;
      if (equalStr(name, "max") )
        icp_r_max = std::stof(value) * unit_length;
      if (equalStr(name, "step") )
        icp_r_step = std::stof(value) * unit_length;
    }
    if (axis == ANGLE) {
      if (equalStr(name, "min") )
        icp_th_min = std::stof(value) * unit_angle;
      if (equalStr(name, "max") )
        icp_th_max = std::stof(value) * unit_angle;
      if (equalStr(name, "step") )
        icp_th_step = std::stof(value) * unit_angle;
    }
  }
  break;
case SWING:
  if (coordinate == CARTESIAN) {
    if (axis == X) {
      if (equalStr(name, "min") )
        swf_x_min = std::stof(value) * unit_length;
      if (equalStr(name, "max") )
        swf_x_max = std::stof(value) * unit_length;
      if (equalStr(name, "step") )
        swf_x_step = std::stof(value) * unit_length;
    }
    if (axis == Y) {
      if (equalStr(name, "min") )
        swf_y_min = std::stof(value) * unit_length;
      if (equalStr(name, "max") )
        swf_y_max = std::stof(value) * unit_length;
      if (equalStr(name, "step") )
        swf_y_step = std::stof(value) * unit_length;
    }
  }
  if (coordinate == POLAR) {
    if (axis == RADIUS) {
      if (equalStr(name, "min") )
        swf_r_min = std::stof(value) * unit_length;
      if (equalStr(name, "max") )
        swf_r_max = std::stof(value) * unit_length;
      if (equalStr(name, "step") )
        swf_r_step = std::stof(value) * unit_length;
    }
    if (axis == ANGLE) {
      if (equalStr(name, "min") )
        swf_th_min = std::stof(value) * unit_angle;
      if (equalStr(name, "max") )
        swf_th_max = std::stof(value) * unit_angle;
      if (equalStr(name, "step") )
        swf_th_step = std::stof(value) * unit_angle;
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
  if (equalStr(element_name, "coordinate") ) {
  if (equalStr(attribute_name, "type") ) {
    if (coordinate == POLAR)
      str = "polar";
    if (coordinate == CARTESIAN)
      str = "cartesian";
  }
  }
  return str;
}

double Param::getVal(const char *element_name, const char *attribute_name) {
  double val = 0.0;
  if (equalStr(element_name, "icp_x") ) {
    if (equalStr(attribute_name, "min") )
      val = icp_x_min;
    if (equalStr(attribute_name, "max") )
      val = icp_x_max;
    if (equalStr(attribute_name, "step") )
      val = icp_x_step;
    if (equalStr(attribute_name, "num") )
      val = icp_x_num;
  }
  if (equalStr(element_name, "icp_y") ) {
    if (equalStr(attribute_name, "min") )
      val = icp_y_min;
    if (equalStr(attribute_name, "max") )
      val = icp_y_max;
    if (equalStr(attribute_name, "step") )
      val = icp_y_step;
    if (equalStr(attribute_name, "num") )
      val = icp_y_num;
  }
  if (equalStr(element_name, "icp_r") ) {
    if (equalStr(attribute_name, "min") )
      val = icp_r_min;
    if (equalStr(attribute_name, "max") )
      val = icp_r_max;
    if (equalStr(attribute_name, "step") )
      val = icp_r_step;
    if (equalStr(attribute_name, "num") )
      val = icp_r_num;
  }
  if (equalStr(element_name, "icp_th") ) {
    if (equalStr(attribute_name, "min") )
      val = icp_th_min;
    if (equalStr(attribute_name, "max") )
      val = icp_th_max;
    if (equalStr(attribute_name, "step") )
      val = icp_th_step;
    if (equalStr(attribute_name, "num") )
      val = icp_th_num;
  }
  if (equalStr(element_name, "swf_x") ) {
    if (equalStr(attribute_name, "min") )
      val = swf_x_min;
    if (equalStr(attribute_name, "max") )
      val = swf_x_max;
    if (equalStr(attribute_name, "step") )
      val = swf_x_step;
    if (equalStr(attribute_name, "num") )
      val = swf_x_num;
  }
  if (equalStr(element_name, "swf_y") ) {
    if (equalStr(attribute_name, "min") )
      val = swf_y_min;
    if (equalStr(attribute_name, "max") )
      val = swf_y_max;
    if (equalStr(attribute_name, "step") )
      val = swf_y_step;
    if (equalStr(attribute_name, "num") )
      val = swf_y_num;
  }
  if (equalStr(element_name, "swf_r") ) {
    if (equalStr(attribute_name, "min") )
      val = swf_r_min;
    if (equalStr(attribute_name, "max") )
      val = swf_r_max;
    if (equalStr(attribute_name, "step") )
      val = swf_r_step;
    if (equalStr(attribute_name, "num") )
      val = swf_r_num;
  }
  if (equalStr(element_name, "swf_th") ) {
    if (equalStr(attribute_name, "min") )
      val = swf_th_min;
    if (equalStr(attribute_name, "max") )
      val = swf_th_max;
    if (equalStr(attribute_name, "step") )
      val = swf_th_step;
    if (equalStr(attribute_name, "num") )
      val = swf_th_num;
  }
  return val;
}

int Param::round(double value) {
  int result = (int)value;

  double decimal = value - (int)value;
  if (decimal >= 0.5) {
    result += 1;
  }

  return result;
}

void Param::calcNum() {
  using namespace Pa;
  if (coordinate == CARTESIAN) {
  icp_x_num = round( ( icp_x_max - icp_x_min ) / icp_x_step) + 1;
  icp_y_num = round( ( icp_y_max - icp_y_min ) / icp_y_step) + 1;
  swf_x_num = round( ( swf_x_max - swf_x_min ) / swf_x_step) + 1;
  swf_y_num = round( ( swf_y_max - swf_y_min ) / swf_y_step) + 1;
  }
  if (coordinate == POLAR) {
    icp_r_num  = round( ( icp_r_max - icp_r_min ) / icp_r_step) + 1;
    icp_th_num = round( ( icp_th_max - icp_th_min ) / icp_th_step) + 1;
    swf_r_num  = round( ( swf_r_max - swf_r_min ) / swf_r_step) + 1;
    swf_th_num = round( ( swf_th_max - swf_th_min ) / swf_th_step) + 1;
  }
}

void Param::print() {
  using namespace Pa;
  printf("-------------------------------------------\n");
  printf("coordinate:\n");
  if (coordinate == CARTESIAN)
    printf("\ttype: %s\n", "cartesian");
  if (coordinate == POLAR)
    printf("\ttype: %s\n", "polar");

  printf("unit:\n");
  printf("\tlength: %lf [m]\n", unit_length);
  printf("\tangle : %lf [rad]\n", unit_angle);

  printf("icp:\n");
  if (coordinate == CARTESIAN) {
  printf("\tx:\n");
  printf("\t\tmin : %lf\n", icp_x_min);
  printf("\t\tmax : %lf\n", icp_x_max);
  printf("\t\tstep: %lf\n", icp_x_step);
  printf("\t\tnum : %d \n", icp_x_num);
  printf("\ty:\n");
  printf("\t\tmin : %lf\n", icp_y_min);
  printf("\t\tmax : %lf\n", icp_y_max);
  printf("\t\tstep: %lf\n", icp_y_step);
  printf("\t\tnum : %d \n", icp_y_num);
  }else if(coordinate == POLAR) {
    printf("\tradius:\n");
    printf("\t\tmin : %lf\n", icp_r_min);
    printf("\t\tmax : %lf\n", icp_r_max);
    printf("\t\tstep: %lf\n", icp_r_step);
    printf("\t\tnum : %d \n", icp_r_num);
    printf("\tangle:\n");
    printf("\t\tmin : %lf\n", icp_th_min);
    printf("\t\tmax : %lf\n", icp_th_max);
    printf("\t\tstep: %lf\n", icp_th_step);
    printf("\t\tnum : %d \n", icp_th_num);
  }

  printf("swing:\n");
  if (coordinate == CARTESIAN) {
    printf("\tx:\n");
    printf("\t\tmin : %lf\n", swf_x_min);
    printf("\t\tmax : %lf\n", swf_x_max);
    printf("\t\tstep: %lf\n", swf_x_step);
    printf("\t\tnum : %d \n", swf_x_num);
    printf("\ty:\n");
    printf("\t\tmin : %lf\n", swf_y_min);
    printf("\t\tmax : %lf\n", swf_y_max);
    printf("\t\tstep: %lf\n", swf_y_step);
    printf("\t\tnum : %d \n", swf_y_num);
  }else if(coordinate == POLAR) {
    printf("\tradius:\n");
    printf("\t\tmin : %lf\n", swf_r_min);
    printf("\t\tmax : %lf\n", swf_r_max);
    printf("\t\tstep: %lf\n", swf_r_step);
    printf("\t\tnum : %d \n", swf_r_num);
    printf("\tangle:\n");
    printf("\t\tmin : %lf\n", swf_th_min);
    printf("\t\tmax : %lf\n", swf_th_max);
    printf("\t\tstep: %lf\n", swf_th_step);
    printf("\t\tnum : %d \n", swf_th_num);
  }
  printf("-------------------------------------------\n");
}

} // namespace Capt