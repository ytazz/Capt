#include "graph.h"

namespace CA {

Graph::Graph(const std::string &name) : Loader(name), pi(M_PI) {
  element = Gr::NOELEMENT;
  coordinate = Gr::NOCOORD;

  unit_length = 0.0;
  unit_angle = 0.0;

  radius_min = 0.0;
  radius_max = 0.0;
  radius_step = 0.0;
  radius_tick = 0.0;
  angle_min = 0.0;
  angle_max = 0.0;
  angle_step = 0.0;
  angle_tick = 0.0;
}

Graph::~Graph() {}

void Graph::callbackElement(const std::string &name, const bool is_start) {
  using namespace Gr;
  if (is_start) {
    switch (element) {
    case NOELEMENT:
      if (equalStr(name, "coordinate"))
        element = COORDINATE;
      break;
    case COORDINATE:
      if (equalStr(name, "unit"))
        element = UNIT;
      if (equalStr(name, "radius"))
        element = RADIUS;
      if (equalStr(name, "angle"))
        element = ANGLE;
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
    case RADIUS:
    case ANGLE:
      element = COORDINATE;
      break;
    default:
      break;
    }
  }
}

void Graph::callbackAttribute(const std::string &name,
                              const std::string &value) {
  using namespace Gr;
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
  case RADIUS:
    if (coordinate == POLAR) {
      if (equalStr(name, "min"))
        radius_min = std::stof(value) * unit_length;
      if (equalStr(name, "max"))
        radius_max = std::stof(value) * unit_length;
      if (equalStr(name, "step"))
        radius_step = std::stof(value) * unit_length;
      if (equalStr(name, "tick"))
        radius_tick = std::stof(value) * unit_length;
    }
    break;
  case ANGLE:
    if (coordinate == POLAR) {
      if (equalStr(name, "min"))
        angle_min = std::stof(value) * unit_angle;
      if (equalStr(name, "max"))
        angle_max = std::stof(value) * unit_angle;
      if (equalStr(name, "step"))
        angle_step = std::stof(value) * unit_angle;
      if (equalStr(name, "tick"))
        angle_tick = std::stof(value) * unit_angle;
    }
    break;
  default:
    break;
  }
}

void Graph::get(const char *element_name, const char *attribute_name,
                std::string &str) {
  using namespace Gr;
  if (equalStr(element_name, "coordinate")) {
    if (equalStr(attribute_name, "type")) {
      if (coordinate == POLAR)
        str = "polar";
      if (coordinate == CARTESIAN)
        str = "cartesian";
    }
  }
}

void Graph::get(const char *element_name, const char *attribute_name,
                float *val) {
  if (equalStr(element_name, "radius")) {
    if (equalStr(attribute_name, "min"))
      *val = radius_min;
    if (equalStr(attribute_name, "max"))
      *val = radius_max;
    if (equalStr(attribute_name, "step"))
      *val = radius_step;
    if (equalStr(attribute_name, "tick"))
      *val = radius_tick;
  }
  if (equalStr(element_name, "angle")) {
    if (equalStr(attribute_name, "min"))
      *val = angle_min;
    if (equalStr(attribute_name, "max"))
      *val = angle_max;
    if (equalStr(attribute_name, "step"))
      *val = angle_step;
    if (equalStr(attribute_name, "tick"))
      *val = angle_tick;
  }
}

void Graph::print() {
  using namespace Gr;
  printf("-------------------------------------------\n");
  printf("coordinate:\n");
  if (coordinate == POLAR)
    printf("\ttype: %s\n", "polar");
  if (coordinate == CARTESIAN)
    printf("\ttype: %s\n", "cartesian");

  printf("unit:\n");
  printf("\tlength: %lf\n", unit_length);
  printf("\tangle : %lf\n", unit_angle);

  printf("radius:\n");
  printf("\tmin : %lf\n", radius_min);
  printf("\tmax : %lf\n", radius_max);
  printf("\tstep: %lf\n", radius_step);
  printf("\ttick: %lf\n", radius_tick);

  printf("angle:\n");
  printf("\tmin : %lf\n", angle_min);
  printf("\tmax : %lf\n", angle_max);
  printf("\tstep: %lf\n", angle_step);
  printf("\ttick: %lf\n", angle_tick);
  printf("-------------------------------------------\n");
}

} // namespace CA