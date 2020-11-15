#include "model.h"

namespace Capt {

Model::Model(const std::string &name) : Loader(name) {
  printf("\x1b[36mModel File: %s\x1b[39m\n", name.c_str() );

  element = MODEL_ELE_NONE;
  shape   = SHAPE_NONE;
  foot    = FOOT_NONE;

  robot_name = "";

  unit_length = 0.0;
  unit_mass   = 0.0;
  unit_time   = 0.0;

  mass             = 0.0;
  com_height       = 0.0;
  step_time_min    = 0.0;
  foot_vel_max     = 0.0;
  swing_height_max = 0.0;

  foot_r.clear();
  foot_l.clear();

  parse();
}

Model::~Model() {
}

void Model::callbackElement(const std::string &name, const bool is_start) {
  if (is_start) {
    switch (element) {
    case MODEL_ELE_NONE:
      if (equalStr(name, "robot") )
        element = MODEL_ELE_ROBOT;
      break;
    case MODEL_ELE_ROBOT:
      if (equalStr(name, "unit") )
        element = MODEL_ELE_UNIT;
      if (equalStr(name, "physics") )
        element = MODEL_ELE_PHYSICS;
      if (equalStr(name, "environment") )
        element = MODEL_ELE_ENVIRONMENT;
      if (equalStr(name, "foot") )
        element = MODEL_ELE_FOOT;
      break;
    case MODEL_ELE_UNIT:
    case MODEL_ELE_PHYSICS:
    case MODEL_ELE_ENVIRONMENT:
      break;
    case MODEL_ELE_FOOT:
      if (equalStr(name, "shape") )
        element = MODEL_ELE_SHAPE;
      break;
    default:
      break;
    }
  } else {
    switch (element) {
    case MODEL_ELE_NONE:
      break;
    case MODEL_ELE_ROBOT:
      element = MODEL_ELE_NONE;
      break;
    case MODEL_ELE_UNIT:
    case MODEL_ELE_PHYSICS:
    case MODEL_ELE_ENVIRONMENT:
    case MODEL_ELE_FOOT:
      element = MODEL_ELE_ROBOT;
      foot    = FOOT_NONE;
      break;
    case MODEL_ELE_SHAPE:
      if (equalStr(name, "shape") ) {
        shape   = SHAPE_NONE;
        element = MODEL_ELE_FOOT;
      }
      break;
    default:
      break;
    }
  }
}

void Model::callbackAttribute(const std::string &name,
                              const std::string &value) {
  switch (element) {
  case MODEL_ELE_ROBOT:
    if (equalStr(name, "name") )
      robot_name = value;
    break;
  case MODEL_ELE_UNIT:
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
    break;
  case MODEL_ELE_PHYSICS:
    if (equalStr(name, "mass") )
      mass = std::stof(value);
    if (equalStr(name, "com_height") )
      com_height = std::stof(value);
    if (equalStr(name, "step_time_min") )
      step_time_min = std::stof(value);
    if (equalStr(name, "foot_vel_max") )
      foot_vel_max = std::stof(value);
    if (equalStr(name, "swing_height_max") )
      swing_height_max = std::stof(value);
    break;
  case MODEL_ELE_ENVIRONMENT:
    if (equalStr(name, "gravity") )
      gravity = std::stof(value);
    if (equalStr(name, "friction") )
      friction = std::stof(value);
    break;
  case MODEL_ELE_FOOT:
    if (equalStr(name, "name") ) {
      if (equalStr(value, "foot_r") )
        foot = FOOT_R;
      if (equalStr(value, "foot_l") )
        foot = FOOT_L;
    }
    break;
  case MODEL_ELE_SHAPE:
    if (equalStr(name, "type") ) {
      if (equalStr(value, "polygon") )
        shape = SHAPE_POLYGON;
      if (equalStr(value, "reverse") )
        shape = SHAPE_REVERSE;
    }
    switch (shape) {
    case SHAPE_POLYGON:
      if (equalStr(name, "point") ) {
        if (foot == FOOT_R) {
          foot_r.push_back(convertStrToVec2(value) * unit_length);
          if (foot_r.size() > 3) {
            polygon.clear();
            polygon.setVertex(foot_r);
            foot_r_convex = polygon.getConvexHull();
          }
        }
        if (foot == FOOT_L) {
          foot_l.push_back(convertStrToVec2(value) * unit_length);
          if (foot_l.size() > 3) {
            polygon.clear();
            polygon.setVertex(foot_l);
            foot_l_convex = polygon.getConvexHull();
          }
        }
      }
      break;
    case SHAPE_REVERSE:
      if (foot == FOOT_R) {
        for (size_t i = 0; i < foot_l.size(); i++) {
          foot_r.push_back(vec2_t(foot_l[i].x(), -foot_l[i].y() ) );
        }
        if (foot_r.size() > 3) {
          polygon.clear();
          polygon.setVertex(foot_r);
          foot_r_convex = polygon.getConvexHull();
        }
      }
      if (foot == FOOT_L) {
        for (size_t i = 0; i < foot_r.size(); i++) {
          foot_l.push_back(vec2_t(foot_r[i].x(), -foot_r[i].y() ) );
        }
        if (foot_l.size() > 3) {
          polygon.clear();
          polygon.setVertex(foot_l);
          foot_l_convex = polygon.getConvexHull();
        }
      }
      break;
    default:
      break;
    }
  default:
    break;
  }
}

void Model::read(std::string *val, const std::string &name){
  if (equalStr(name, "name") )
    *val = robot_name;
}

void Model::read(int *val, const std::string &name){
  if (equalStr(name, "foot_r_num") )
    *val = (int)foot_r.size();
  if (equalStr(name, "foot_r_convex_num") )
    *val = (int)foot_r_convex.size();
  if (equalStr(name, "foot_l_num") )
    *val = (int)foot_l.size();
  if (equalStr(name, "foot_l_convex_num") )
    *val = (int)foot_l_convex.size();
}

void Model::read(float *val, const std::string &name){
  if (equalStr(name, "mass") )
    *val = mass;
  if (equalStr(name, "com_height") )
    *val = com_height;
  if (equalStr(name, "step_time_min") )
    *val = step_time_min;
  if (equalStr(name, "foot_vel_max") )
    *val = foot_vel_max;
  if (equalStr(name, "swing_height_max") )
    *val = swing_height_max;
  if (equalStr(name, "gravity") )
    *val = gravity;
  if (equalStr(name, "friction") )
    *val = friction;

  if (equalStr(name, "omega") )
    *val = sqrt(gravity / com_height);
}

void Model::read(arr2_t *val, const std::string &name){
  if (equalStr(name, "foot_r") )
    *val = foot_r;
  if (equalStr(name, "foot_r_convex") )
    *val = foot_r_convex;
  if (equalStr(name, "foot_l") )
    *val = foot_l;
  if (equalStr(name, "foot_l_convex") )
    *val = foot_l_convex;
}

void Model::read(arr2_t *val, const std::string &name, vec2_t offset){
  read(val, name);
  for (size_t i = 0; i < val->size(); i++) {
    ( *val )[i] += offset;
  }
}

void Model::print() {
  printf("-------------------------------------------\n");
  printf("robot:\n");
  printf("\tname: %s\n", robot_name.c_str() );
  printf("unit:\n");
  printf("\tlength: %lf [m]\n", unit_length);
  printf("\tmass  : %lf [kg]\n", unit_mass);
  printf("\ttime  : %lf [s]\n", unit_time);
  printf("physics:\n");
  printf("\tmass            : %lf\n", mass);
  printf("\tcom_height      : %lf\n", com_height);
  printf("\tstep_time_min   : %lf\n", step_time_min);
  printf("\tfoot_vel_max    : %lf\n", foot_vel_max);
  printf("\tswing_height_max: %lf\n", swing_height_max);
  printf("environment:\n");
  printf("\tgravity : %lf\n", gravity);
  printf("\tfriction: %lf\n", friction);
  printf("foot: foot_r\n");
  for (size_t i = 0; i < foot_r.size(); i++) {
    printf("\t%lf, %lf\n", foot_r[i].x(), foot_r[i].y() );
  }
  printf("foot: foot_l\n");
  for (size_t i = 0; i < foot_l.size(); i++) {
    printf("\t%lf, %lf\n", foot_l[i].x(), foot_l[i].y() );
  }
  printf("-------------------------------------------\n");
}

} // namespace Capt