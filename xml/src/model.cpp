#include "model.h"

namespace CA {

Model::Model(const std::string &name) : Loader(name), pi(M_PI) {
  element = Mo::NOELEMENT;
  foot = Mo::NOFOOT;
  shape = Mo::NOSHAPE;

  trn = {0.0, 0.0};
  rot = 0.0;

  robot_name = "";

  unit_length = 0.0;
  unit_mass = 0.0;
  unit_time = 0.0;
  unit_angle = 0.0;

  mass = 0.0;
  com_height = 0.0;
  step_time_min = 0.0;
  foot_vel_max = 0.0;

  foot_r.clear();
  foot_l.clear();
}

Model::~Model() {}

void Model::callbackElement(const std::string &name, const bool is_start) {
  using namespace Mo;
  if (is_start) {
    switch (element) {
    case NOELEMENT:
      if (equalStr(name, "robot"))
        element = ROBOT;
      break;
    case ROBOT:
      if (equalStr(name, "unit"))
        element = UNIT;
      if (equalStr(name, "physics"))
        element = PHYSICS;
      if (equalStr(name, "link"))
        element = LINK;
      break;
    case UNIT:
    case PHYSICS:
      break;
    case LINK:
      if (equalStr(name, "shape"))
        element = SHAPE;
      break;
    default:
      break;
    }
  } else {
    switch (element) {
    case NOELEMENT:
      break;
    case ROBOT:
      element = NOELEMENT;
      break;
    case UNIT:
    case PHYSICS:
    case LINK:
      foot = NOFOOT;
      element = ROBOT;
      break;
    case SHAPE:
      if (equalStr(name, "shape")) {
        shape = NOSHAPE;
        trn = {0.0, 0.0};
        rot = 0.0;
        element = LINK;
      }
      break;
    default:
      break;
    }
  }
}

void Model::callbackAttribute(const std::string &name,
                              const std::string &value) {
  using namespace Mo;
  switch (element) {
  case ROBOT:
    if (equalStr(name, "name"))
      robot_name = value;
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
    if (equalStr(name, "mass")) {
      if (equalStr(value, "kg"))
        unit_mass = 1.0;
      if (equalStr(value, "g"))
        unit_mass = 1.0 / 1000.0;
    }
    if (equalStr(name, "time")) {
      if (equalStr(value, "s"))
        unit_time = 1.0;
      if (equalStr(value, "ms"))
        unit_time = 1.0 / 1000.0;
    }
    if (equalStr(name, "angle")) {
      if (equalStr(value, "rad"))
        unit_angle = 1.0;
      if (equalStr(value, "deg"))
        unit_angle = pi / 180.0;
    }
    break;
  case PHYSICS:
    if (equalStr(name, "mass"))
      mass = std::stof(value);
    if (equalStr(name, "com_height"))
      com_height = std::stof(value);
    if (equalStr(name, "step_time_min"))
      step_time_min = std::stof(value);
    if (equalStr(name, "foot_vel_max"))
      foot_vel_max = std::stof(value);
    break;
  case LINK:
    if (equalStr(name, "name")) {
      if (equalStr(value, "foot_r"))
        foot = RFOOT;
      if (equalStr(value, "foot_l"))
        foot = LFOOT;
    }
    break;
  case SHAPE:
    if (equalStr(name, "type")) {
      if (equalStr(value, "box"))
        shape = BOX;
      if (equalStr(value, "polygon"))
        shape = POLYGON;
      if (equalStr(value, "circle"))
        shape = CIRCLE;
      if (equalStr(value, "reverse"))
        shape = REVERSE;
    }
    if (equalStr(name, "trn"))
      trn = convertStrToVec(value);
    if (equalStr(name, "rot"))
      rot = std::stof(value);
    switch (shape) {
    case POLYGON:
      if (equalStr(name, "point")) {
        if (foot == RFOOT)
          foot_r.push_back(convertStrToVec(value) * unit_length);
        if (foot == LFOOT)
          foot_l.push_back(convertStrToVec(value) * unit_length);
      }
      break;
    case REVERSE:
      if (foot == RFOOT) {
        for (size_t i = 0; i < foot_l.size(); i++) {
          foot_r.push_back({foot_l[i].x, -foot_l[i].y});
        }
      }
      if (foot == LFOOT)
        for (size_t i = 0; i < foot_r.size(); i++) {
          foot_l.push_back({foot_r[i].x, -foot_r[i].y});
        }
      break;
    default:
      break;
    }
    break;
  default:
    break;
  }
}

void Model::get(const char *element_name, const char *attribute_name,
                std::string &str) {
  if (equalStr(element_name, "robot")) {
    if (equalStr(attribute_name, "name"))
      str = robot_name;
  }
}

void Model::get(const char *element_name, const char *attribute_name,
                float *val) {
  if (equalStr(element_name, "physics")) {
    if (equalStr(attribute_name, "mass"))
      *val = mass;
    if (equalStr(attribute_name, "com_height"))
      *val = com_height;
    if (equalStr(attribute_name, "step_time_min"))
      *val = step_time_min;
    if (equalStr(attribute_name, "foot_vel_max"))
      *val = foot_vel_max;
  }
}

void Model::get(const char *element_name, const char *attribute_name,
                std::vector<Vector2> *vec) {
  if (equalStr(element_name, "link")) {
    if (equalStr(attribute_name, "foot_r"))
      *vec = foot_r;
    if (equalStr(attribute_name, "foot_l"))
      *vec = foot_l;
  }
}

void Model::print() {
  printf("-------------------------------------------\n");
  printf("robot:\n");
  printf("\tname: %s\n", robot_name.c_str());
  printf("unit:\n");
  printf("\tlength: %lf\n", unit_length);
  printf("\tmass  : %lf\n", unit_mass);
  printf("\ttime  : %lf\n", unit_time);
  printf("\tangle : %lf\n", unit_angle);
  printf("physics:\n");
  printf("\tmass         : %lf\n", mass);
  printf("\tcom_height   : %lf\n", com_height);
  printf("\tstep_time_min: %lf\n", step_time_min);
  printf("\tfoot_vel_max : %lf\n", foot_vel_max);
  printf("link: foot_r\n");
  for (size_t i = 0; i < foot_r.size(); i++) {
    printf("\t%lf, %lf\n", foot_r[i].x, foot_r[i].y);
  }
  printf("link: foot_l\n");
  for (size_t i = 0; i < foot_l.size(); i++) {
    printf("\t%lf, %lf\n", foot_l[i].x, foot_l[i].y);
  }
  printf("-------------------------------------------\n");
}

} // namespace CA
