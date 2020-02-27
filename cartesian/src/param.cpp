#include "param.h"

namespace Capt {

Param::Param(const std::string &name) : Loader(name) {
  printf("\x1b[36mParam File: %s\x1b[39m\n", name.c_str() );

  element    = CaptEnum::PARAM_ELE_NONE;
  coordinate = CaptEnum::COORD_NONE;
  axis       = CaptEnum::AXIS_NONE;

  unit_length = 0.0;

  icp_x_min = 0.0;
  icp_x_max = 0.0;
  icp_x_stp = 0.0;
  icp_y_min = 0.0;
  icp_y_max = 0.0;
  icp_y_stp = 0.0;

  swf_x_min = 0.0;
  swf_x_max = 0.0;
  swf_x_stp = 0.0;
  swf_y_min = 0.0;
  swf_y_max = 0.0;
  swf_y_stp = 0.0;
  swf_z_min = 0.0;
  swf_z_max = 0.0;
  swf_z_stp = 0.0;

  exc_x_min = 0.0;
  exc_x_max = 0.0;
  exc_y_min = 0.0;
  exc_y_max = 0.0;

  cop_x_min = 0.0;
  cop_x_max = 0.0;
  cop_x_stp = 0.0;
  cop_y_min = 0.0;
  cop_y_max = 0.0;
  cop_y_stp = 0.0;

  map_x_min = 0.0;
  map_x_max = 0.0;
  map_x_stp = 0.0;
  map_y_min = 0.0;
  map_y_max = 0.0;
  map_y_stp = 0.0;

  parse();
}

Param::~Param() {
}

void Param::callbackElement(const std::string &name, const bool is_start) {
  using namespace CaptEnum;
  if (is_start) {
  switch (element) {
  case PARAM_ELE_NONE:
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
    if (equalStr(name, "except") )
      element = EXCEPT;
    if (equalStr(name, "cop") )
      element = COP;
    if (equalStr(name, "map") )
      element = MAP;
    break;
  case ICP:
  case SWING:
  case EXCEPT:
  case COP:
  case MAP:
    if (equalStr(name, "x") )
      axis = AXIS_X;
    if (equalStr(name, "y") )
      axis = AXIS_Y;
    if (equalStr(name, "z") )
      axis = AXIS_Z;
    break;
  default:
    break;
  }
  } else {
    switch (element) {
    case COORDINATE:
      element = PARAM_ELE_NONE;
      calcNum();
      break;
    case UNIT:
      element = COORDINATE;
      axis    = AXIS_NONE;
      break;
    case ICP:
    case SWING:
    case EXCEPT:
    case COP:
    case MAP:
      if (axis != AXIS_NONE)
        axis = AXIS_NONE;
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
  using namespace CaptEnum;
  switch (element) {
case COORDINATE:
  if (equalStr(name, "type") ) {
    if (equalStr(value, "cartesian") )
      coordinate =  COORD_CARTESIAN;
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
  break;
case ICP:
  if (coordinate == COORD_CARTESIAN) {
    if (axis == AXIS_X) {
      if (equalStr(name, "min") )
        icp_x_min = std::stof(value) * unit_length;
      if (equalStr(name, "max") )
        icp_x_max = std::stof(value) * unit_length;
      if (equalStr(name, "stp") )
        icp_x_stp = std::stof(value) * unit_length;
    }
    if (axis == AXIS_Y) {
      if (equalStr(name, "min") )
        icp_y_min = std::stof(value) * unit_length;
      if (equalStr(name, "max") )
        icp_y_max = std::stof(value) * unit_length;
      if (equalStr(name, "stp") )
        icp_y_stp = std::stof(value) * unit_length;
    }
  }
  break;
case SWING:
  if (coordinate == COORD_CARTESIAN) {
    if (axis == AXIS_X) {
      if (equalStr(name, "min") )
        swf_x_min = std::stof(value) * unit_length;
      if (equalStr(name, "max") )
        swf_x_max = std::stof(value) * unit_length;
      if (equalStr(name, "stp") )
        swf_x_stp = std::stof(value) * unit_length;
    }
    if (axis == AXIS_Y) {
      if (equalStr(name, "min") )
        swf_y_min = std::stof(value) * unit_length;
      if (equalStr(name, "max") )
        swf_y_max = std::stof(value) * unit_length;
      if (equalStr(name, "stp") )
        swf_y_stp = std::stof(value) * unit_length;
    }
    if (axis == AXIS_Z) {
      if (equalStr(name, "min") )
        swf_z_min = std::stof(value) * unit_length;
      if (equalStr(name, "max") )
        swf_z_max = std::stof(value) * unit_length;
      if (equalStr(name, "stp") )
        swf_z_stp = std::stof(value) * unit_length;
    }
  }
  break;
case EXCEPT:
  if (coordinate == COORD_CARTESIAN) {
    if (axis == AXIS_X) {
      if (equalStr(name, "min") )
        exc_x_min = std::stof(value) * unit_length;
      if (equalStr(name, "max") )
        exc_x_max = std::stof(value) * unit_length;
    }
    if (axis == AXIS_Y) {
      if (equalStr(name, "min") )
        exc_y_min = std::stof(value) * unit_length;
      if (equalStr(name, "max") )
        exc_y_max = std::stof(value) * unit_length;
    }
  }
  break;
case COP:
  if (coordinate == COORD_CARTESIAN) {
    if (axis == AXIS_X) {
      if (equalStr(name, "min") )
        cop_x_min = std::stof(value) * unit_length;
      if (equalStr(name, "max") )
        cop_x_max = std::stof(value) * unit_length;
      if (equalStr(name, "stp") )
        cop_x_stp = std::stof(value) * unit_length;
    }
    if (axis == AXIS_Y) {
      if (equalStr(name, "min") )
        cop_y_min = std::stof(value) * unit_length;
      if (equalStr(name, "max") )
        cop_y_max = std::stof(value) * unit_length;
      if (equalStr(name, "stp") )
        cop_y_stp = std::stof(value) * unit_length;
    }
  }
  break;
case MAP:
  if (coordinate == COORD_CARTESIAN) {
    if (axis == AXIS_X) {
      if (equalStr(name, "min") )
        map_x_min = std::stof(value) * unit_length;
      if (equalStr(name, "max") )
        map_x_max = std::stof(value) * unit_length;
      if (equalStr(name, "stp") )
        map_x_stp = std::stof(value) * unit_length;
    }
    if (axis == AXIS_Y) {
      if (equalStr(name, "min") )
        map_y_min = std::stof(value) * unit_length;
      if (equalStr(name, "max") )
        map_y_max = std::stof(value) * unit_length;
      if (equalStr(name, "stp") )
        map_y_stp = std::stof(value) * unit_length;
    }
  }
  break;
default:
  break;
  }
}

void Param::read(std::string *val, const std::string &name){
  using namespace CaptEnum;
  if (equalStr(name, "coordinate") ) {
  if (coordinate == COORD_CARTESIAN)
    *val = "cartesian";
  }
}

void Param::read(int *val, const std::string &name) {
  if (equalStr(name, "icp_x_num") )
    *val = icp_x_num;
  if (equalStr(name, "icp_y_num") )
    *val = icp_y_num;
  if (equalStr(name, "swf_x_num") )
    *val = swf_x_num;
  if (equalStr(name, "swf_y_num") )
    *val = swf_y_num;
  if (equalStr(name, "swf_z_num") )
    *val = swf_z_num;
  if (equalStr(name, "cop_x_num") )
    *val = cop_x_num;
  if (equalStr(name, "cop_y_num") )
    *val = cop_y_num;
  if (equalStr(name, "map_x_num") )
    *val = map_x_num;
  if (equalStr(name, "map_y_num") )
    *val = map_y_num;
}

void Param::read(double *val, const std::string &name){
  if (equalStr(name, "icp_x_min") )
    *val = icp_x_min;
  if (equalStr(name, "icp_x_max") )
    *val = icp_x_max;
  if (equalStr(name, "icp_x_stp") )
    *val = icp_x_stp;

  if (equalStr(name, "icp_y_min") )
    *val = icp_y_min;
  if (equalStr(name, "icp_y_max") )
    *val = icp_y_max;
  if (equalStr(name, "icp_y_stp") )
    *val = icp_y_stp;

  if (equalStr(name, "swf_x_min") )
    *val = swf_x_min;
  if (equalStr(name, "swf_x_max") )
    *val = swf_x_max;
  if (equalStr(name, "swf_x_stp") )
    *val = swf_x_stp;

  if (equalStr(name, "swf_y_min") )
    *val = swf_y_min;
  if (equalStr(name, "swf_y_max") )
    *val = swf_y_max;
  if (equalStr(name, "swf_y_stp") )
    *val = swf_y_stp;

  if (equalStr(name, "swf_z_min") )
    *val = swf_z_min;
  if (equalStr(name, "swf_z_max") )
    *val = swf_z_max;
  if (equalStr(name, "swf_z_stp") )
    *val = swf_z_stp;

  if (equalStr(name, "exc_x_min") )
    *val = exc_x_min;
  if (equalStr(name, "exc_x_max") )
    *val = exc_x_max;

  if (equalStr(name, "exc_y_min") )
    *val = exc_y_min;
  if (equalStr(name, "exc_y_max") )
    *val = exc_y_max;

  if (equalStr(name, "cop_x_min") )
    *val = cop_x_min;
  if (equalStr(name, "cop_x_max") )
    *val = cop_x_max;
  if (equalStr(name, "cop_x_stp") )
    *val = cop_x_stp;

  if (equalStr(name, "cop_y_min") )
    *val = cop_y_min;
  if (equalStr(name, "cop_y_max") )
    *val = cop_y_max;
  if (equalStr(name, "cop_y_stp") )
    *val = cop_y_stp;

  if (equalStr(name, "map_x_min") )
    *val = map_x_min;
  if (equalStr(name, "map_x_max") )
    *val = map_x_max;
  if (equalStr(name, "map_x_stp") )
    *val = map_x_stp;

  if (equalStr(name, "map_y_min") )
    *val = map_y_min;
  if (equalStr(name, "map_y_max") )
    *val = map_y_max;
  if (equalStr(name, "map_y_stp") )
    *val = map_y_stp;
}

void Param::calcNum() {
  using namespace CaptEnum;
  double epsilon = 0.00001;
  if (coordinate == COORD_CARTESIAN) {
  // icp
  if(icp_x_stp > epsilon)
    icp_x_num = round( ( icp_x_max - icp_x_min ) / icp_x_stp) + 1;
  else
    icp_x_num = 0;
  if(icp_y_stp > epsilon)
    icp_y_num = round( ( icp_y_max - icp_y_min ) / icp_y_stp) + 1;
  else
    icp_y_num = 0;
  // swf
  if(swf_x_stp > epsilon)
    swf_x_num = round( ( swf_x_max - swf_x_min ) / swf_x_stp) + 1;
  else
    swf_x_num = 0;
  if(swf_y_stp > epsilon)
    swf_y_num = round( ( swf_y_max - swf_y_min ) / swf_y_stp) + 1;
  else
    swf_y_num = 0;
  if(swf_z_stp > epsilon)
    swf_z_num = round( ( swf_z_max - swf_z_min ) / swf_z_stp) + 1;
  else
    swf_z_num = 0;
  // cop
  if(cop_x_stp > epsilon)
    cop_x_num = round( ( cop_x_max - cop_x_min ) / cop_x_stp) + 1;
  else
    cop_x_num = 0;
  if(cop_y_stp > epsilon)
    cop_y_num = round( ( cop_y_max - cop_y_min ) / cop_y_stp) + 1;
  else
    cop_y_num = 0;
  // map
  if(map_x_stp > epsilon)
    map_x_num = round( ( map_x_max - map_x_min ) / map_x_stp) + 1;
  else
    map_x_num = 0;
  if(map_y_stp > epsilon)
    map_y_num = round( ( map_y_max - map_y_min ) / map_y_stp) + 1;
  else
    map_y_num = 0;
  }
}

void Param::print() {
  using namespace CaptEnum;
  printf("-------------------------------------------\n");
  printf("coordinate:\n");
  if (coordinate == COORD_CARTESIAN)
    printf("\ttype: %s\n", "cartesian");
  else
    printf("\ttype: %s\n", "error");

  printf("unit:\n");
  printf("\tlength: %lf [m]\n", unit_length);

  printf("icp:\n");
  if (coordinate == COORD_CARTESIAN) {
  printf("\tx:\n");
  printf("\t\tmin: %+lf\n", icp_x_min);
  printf("\t\tmax: %+lf\n", icp_x_max);
  printf("\t\tstp: %+lf\n", icp_x_stp);
  printf("\t\tnum: %d  \n", icp_x_num);
  printf("\ty:\n");
  printf("\t\tmin: %+lf\n", icp_y_min);
  printf("\t\tmax: %+lf\n", icp_y_max);
  printf("\t\tstp: %+lf\n", icp_y_stp);
  printf("\t\tnum: %d  \n", icp_y_num);
  }

  printf("swing:\n");
  if (coordinate == COORD_CARTESIAN) {
    printf("\tx:\n");
    printf("\t\tmin: %+lf\n", swf_x_min);
    printf("\t\tmax: %+lf\n", swf_x_max);
    printf("\t\tstp: %+lf\n", swf_x_stp);
    printf("\t\tnum: %d  \n", swf_x_num);
    printf("\ty:\n");
    printf("\t\tmin: %+lf\n", swf_y_min);
    printf("\t\tmax: %+lf\n", swf_y_max);
    printf("\t\tstp: %+lf\n", swf_y_stp);
    printf("\t\tnum: %d  \n", swf_y_num);
    printf("\tz:\n");
    printf("\t\tmin: %+lf\n", swf_z_min);
    printf("\t\tmax: %+lf\n", swf_z_max);
    printf("\t\tstp: %+lf\n", swf_z_stp);
    printf("\t\tnum: %d  \n", swf_z_num);
  }

  printf("except:\n");
  if (coordinate == COORD_CARTESIAN) {
    printf("\tx:\n");
    printf("\t\tmin: %+lf\n", exc_x_min);
    printf("\t\tmax: %+lf\n", exc_x_max);
    printf("\ty:\n");
    printf("\t\tmin: %+lf\n", exc_y_min);
    printf("\t\tmax: %+lf\n", exc_y_max);
  }

  printf("cop:\n");
  if (coordinate == COORD_CARTESIAN) {
    printf("\tx:\n");
    printf("\t\tmin: %+lf\n", cop_x_min);
    printf("\t\tmax: %+lf\n", cop_x_max);
    printf("\t\tstp: %+lf\n", cop_x_stp);
    printf("\t\tnum: %d  \n", cop_x_num);
    printf("\ty:\n");
    printf("\t\tmin: %+lf\n", cop_y_min);
    printf("\t\tmax: %+lf\n", cop_y_max);
    printf("\t\tstp: %+lf\n", cop_y_stp);
    printf("\t\tnum: %d  \n", cop_y_num);
  }

  printf("map:\n");
  if (coordinate == COORD_CARTESIAN) {
    printf("\tx:\n");
    printf("\t\tmin: %+lf\n", map_x_min);
    printf("\t\tmax: %+lf\n", map_x_max);
    printf("\t\tstp: %+lf\n", map_x_stp);
    printf("\t\tnum: %d  \n", map_x_num);
    printf("\ty:\n");
    printf("\t\tmin: %+lf\n", map_y_min);
    printf("\t\tmax: %+lf\n", map_y_max);
    printf("\t\tstp: %+lf\n", map_y_stp);
    printf("\t\tnum: %d  \n", map_y_num);
  }
  printf("-------------------------------------------\n");
}

} // namespace Capt