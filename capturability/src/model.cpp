#include "model.h"

namespace Capt {

Model::Model(const std::string &name) : Loader(name), pi(M_PI) {
  printf("\x1b[36mModel File: %s\x1b[39m\n", name.c_str() );

  element = MODEL_ELE_NONE;
  foot    = FOOT_NONE;
  shape   = SHAPE_NONE;
  link    = LINK_NONE;

  robot_name = "";

  unit_length = 0.0;
  unit_mass   = 0.0;
  unit_time   = 0.0;
  unit_angle  = 0.0;

  total_mass    = 0.0;
  com_height    = 0.0;
  step_time_min = 0.0;
  foot_vel_max  = 0.0;
  step_height   = 0.0;

  foot_r.clear();
  foot_l.clear();

  for (int i = 0; i < NUM_LINK; i++) {
    trn[i]  = Eigen::Vector3f::Zero();
    axis[i] = Eigen::Vector3f::Zero();
    com[i]  = Eigen::Vector3f::Zero();
    mass[i] = 0.0f;
  }

  link_name[TORSO]            = "Torso";
  link_name[HEAD_YAW]         = "HeadYaw";
  link_name[HEAD_PITCH]       = "HeadPitch";
  link_name[R_SHOULDER_PITCH] = "RShoulderPitch";
  link_name[R_SHOULDER_ROLL]  = "RShoulderRoll";
  link_name[R_ELBOW_YAW]      = "RElbowYaw";
  link_name[R_ELBOW_ROLL]     = "RElbowRoll";
  link_name[R_WRIST_YAW]      = "RWristYaw";
  link_name[L_SHOULDER_PITCH] = "LShoulderPitch";
  link_name[L_SHOULDER_ROLL]  = "LShoulderRoll";
  link_name[L_ELBOW_YAW]      = "LElbowYaw";
  link_name[L_ELBOW_ROLL]     = "LElbowRoll";
  link_name[L_WRIST_YAW]      = "LWristYaw";
  link_name[R_HIP_YAWPITCH]   = "RHipYawPitch";
  link_name[R_HIP_ROLL]       = "RHipRoll";
  link_name[R_HIP_PITCH]      = "RHipPitch";
  link_name[R_KNEE_PITCH]     = "RKneePitch";
  link_name[R_ANKLE_PITCH]    = "RAnklePitch";
  link_name[R_ANKLE_ROLL]     = "RAnkleRoll";
  link_name[R_FOOT]           = "RFoot";
  link_name[L_HIP_YAWPITCH]   = "LHipYawPitch";
  link_name[L_HIP_ROLL]       = "LHipRoll";
  link_name[L_HIP_PITCH]      = "LHipPitch";
  link_name[L_KNEE_PITCH]     = "LKneePitch";
  link_name[L_ANKLE_PITCH]    = "LAnklePitch";
  link_name[L_ANKLE_ROLL]     = "LAnkleRoll";
  link_name[L_FOOT]           = "LFoot";

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
      if (equalStr(name, "link") )
        element = MODEL_ELE_LINK;
      break;
    case MODEL_ELE_UNIT:
    case MODEL_ELE_PHYSICS:
    case MODEL_ELE_ENVIRONMENT:
      break;
    case MODEL_ELE_FOOT:
      if (equalStr(name, "shape") )
        element = MODEL_ELE_SHAPE;
      break;
    case MODEL_ELE_LINK:
      if (equalStr(name, "joint") )
        element = MODEL_ELE_LINK_JOINT;
      if (equalStr(name, "physics") )
        element = MODEL_ELE_LINK_PHYSICS;
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
    case MODEL_ELE_LINK:
      element = MODEL_ELE_ROBOT;
      foot    = FOOT_NONE;
      link    = LINK_NONE;
      break;
    case MODEL_ELE_SHAPE:
      if (equalStr(name, "shape") ) {
        shape   = SHAPE_NONE;
        element = MODEL_ELE_FOOT;
      }
      break;
    case MODEL_ELE_LINK_JOINT:
    case MODEL_ELE_LINK_PHYSICS:
      element = MODEL_ELE_LINK;
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
    if (equalStr(name, "angle") ) {
      if (equalStr(value, "rad") )
        unit_angle = 1.0;
      if (equalStr(value, "deg") )
        unit_angle = pi / 180.0;
    }
    break;
  case MODEL_ELE_PHYSICS:
    if (equalStr(name, "mass") )
      total_mass = std::stof(value);
    if (equalStr(name, "com_height") )
      com_height = std::stof(value);
    if (equalStr(name, "step_time_min") )
      step_time_min = std::stof(value);
    if (equalStr(name, "foot_vel_max") )
      foot_vel_max = std::stof(value);
    if (equalStr(name, "step_height") )
      step_height = std::stof(value);
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
      if (equalStr(value, "box") )
        shape = SHAPE_BOX;
      if (equalStr(value, "polygon") )
        shape = SHAPE_POLYGON;
      if (equalStr(value, "circle") )
        shape = SHAPE_CIRCLE;
      if (equalStr(value, "reverse") )
        shape = SHAPE_REVERSE;
    }
    // if (equalStr(name, "trn"))
    //   trn = convertStrToVec(value);
    // if (equalStr(name, "rot"))
    //   rot = std::stof(value);
    switch (shape) {
    case SHAPE_POLYGON:
      if (equalStr(name, "point") ) {
        if (foot == FOOT_R) {
          foot_r.push_back(convertStrToVec(value) * unit_length);
          if (foot_r.size() > 3) {
            polygon.clear();
            polygon.setVertex(foot_r);
            foot_r_convex = polygon.getConvexHull();
          }
        }
        if (foot == FOOT_L) {
          foot_l.push_back(convertStrToVec(value) * unit_length);
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
          Vector2 v;
          v.setCartesian(foot_l[i].x, -foot_l[i].y);
          foot_r.push_back(v);
        }
        polygon.clear();
        polygon.setVertex(foot_r);
        foot_r_convex = polygon.getConvexHull();
      }
      if (foot == FOOT_L) {
        for (size_t i = 0; i < foot_r.size(); i++) {
          Vector2 v;
          v.setCartesian(foot_r[i].x, -foot_r[i].y);
          foot_l.push_back(v);
        }
        polygon.clear();
        polygon.setVertex(foot_l);
        foot_l_convex = polygon.getConvexHull();
      }
      break;
    default:
      break;
    }
  case MODEL_ELE_LINK:
    if (equalStr(name, "name") ) {
      for (int i = static_cast<int>( ELink::TORSO );
           i < static_cast<int>( ELink::NUM_LINK ); i++) {
        if (equalStr(value, link_name[i]) ) {
          link = static_cast<ELink>( i );
        }
      }
    }
    if (link != LINK_NONE) {
      if (equalStr(name, "trn") )
        trn[link] = convertStrToVec3(value) * unit_length;
      if (equalStr(name, "axis") )
        axis[link] = convertStrToVec3(value) * unit_length;
    }
    break;
  case MODEL_ELE_LINK_JOINT:
    if (link != LINK_NONE) {
      if (equalStr(name, "limit") ) {
        Vector2 limit_vec = convertStrToVec(value) * unit_angle;
        limit[link][LIMIT_LOWER] = limit_vec.x;
        limit[link][LIMIT_UPPER] = limit_vec.y;
      }
    }
    break;
  case MODEL_ELE_LINK_PHYSICS:
    if (link != LINK_NONE) {
      if (equalStr(name, "com") )
        com[link] = convertStrToVec3(value) * unit_length;
      if (equalStr(name, "mass") )
        mass[link] = stof(value) * unit_mass;
    }
    break;
  default:
    break;
  }
}

std::string Model::getStr(const char *element_name,
                          const char *attribute_name) {
  std::string str = "";
  if (equalStr(element_name, "robot") ) {
    if (equalStr(attribute_name, "name") )
      str = robot_name;
  }
  return str;
}

double Model::getVal(const char *element_name, const char *attribute_name) {
  double val = 0.0;
  if (equalStr(element_name, "physics") ) {
    if (equalStr(attribute_name, "mass") )
      val = total_mass;
    if (equalStr(attribute_name, "com_height") )
      val = com_height;
    if (equalStr(attribute_name, "step_time_min") )
      val = step_time_min;
    if (equalStr(attribute_name, "foot_vel_max") )
      val = foot_vel_max;
    if (equalStr(attribute_name, "step_height") )
      val = step_height;
  }
  if (equalStr(element_name, "environment") ) {
    if (equalStr(attribute_name, "gravity") )
      val = gravity;
    if (equalStr(attribute_name, "friction") )
      val = friction;
  }
  return val;
}

std::vector<Vector2> Model::getVec(const char *element_name,
                                   const char *attribute_name) {
  std::vector<Vector2> vec;
  if (equalStr(element_name, "foot") ) {
    if (equalStr(attribute_name, "foot_r") )
      vec = foot_r;
    if (equalStr(attribute_name, "foot_r_convex") )
      vec = foot_r_convex;
    if (equalStr(attribute_name, "foot_l") )
      vec = foot_l;
    if (equalStr(attribute_name, "foot_l_convex") )
      vec = foot_l_convex;
  }
  return vec;
}

std::vector<Vector2> Model::getVec(const char *element_name,
                                   const char *attribute_name,
                                   vec2_t translation) {
  std::vector<Vector2> vec;
  if (equalStr(element_name, "foot") ) {
    if (equalStr(attribute_name, "foot_r") ) {
      for (size_t i = 0; i < foot_r.size(); i++) {
        vec.push_back(foot_r[i] + translation);
      }
    }
    if (equalStr(attribute_name, "foot_r_convex") ) {
      for (size_t i = 0; i < foot_r_convex.size(); i++) {
        vec.push_back(foot_r_convex[i] + translation);
      }
    }
    if (equalStr(attribute_name, "foot_l") ) {
      for (size_t i = 0; i < foot_l.size(); i++) {
        vec.push_back(foot_l[i] + translation);
      }
    }
    if (equalStr(attribute_name, "foot_l_convex") ) {
      for (size_t i = 0; i < foot_l_convex.size(); i++) {
        vec.push_back(foot_l_convex[i] + translation);
      }
    }
  }
  return vec;
}

std::vector<Vector2> Model::getVec(const char *element_name,
                                   const char *attribute_name,
                                   vec3_t translation) {
  vec2_t trans;
  trans.setCartesian(translation.x(), translation.y() );
  std::vector<Vector2> vec = getVec(element_name, attribute_name, trans);
  return vec;
}

double Model::getLinkVal(ELink link, const char *attribute_name) {
  double val = 0.0;
  if (equalStr(attribute_name, "lower_limit") )
    val = limit[link][LIMIT_LOWER];
  if (equalStr(attribute_name, "upper_limit") )
    val = limit[link][LIMIT_UPPER];
  if (equalStr(attribute_name, "mass") )
    val = mass[link];
  return val;
}

double Model::getLinkVal(int link_id, const char *attribute_name) {
  ELink link = static_cast<ELink>( link_id );
  return getLinkVal(link, attribute_name);
}

vec3_t Model::getLinkVec(ELink link, const char *attribute_name) {
  vec3_t vec = Eigen::Vector3f::Zero();
  if (equalStr(attribute_name, "trn") )
    vec = trn[link];
  if (equalStr(attribute_name, "axis") )
    vec = axis[link];
  if (equalStr(attribute_name, "com") )
    vec = com[link];
  return vec;
}

vec3_t Model::getLinkVec(int link_id, const char *attribute_name) {
  ELink link = static_cast<ELink>( link_id );
  return getLinkVec(link, attribute_name);
}

void Model::print() {
  printf("-------------------------------------------\n");
  printf("robot:\n");
  printf("\tname: %s\n", robot_name.c_str() );
  printf("unit:\n");
  printf("\tlength: %lf [m]\n", unit_length);
  printf("\tmass  : %lf [kg]\n", unit_mass);
  printf("\ttime  : %lf [s]\n", unit_time);
  printf("\tangle : %lf [rad]\n", unit_angle);
  printf("physics:\n");
  printf("\tmass         : %lf\n", total_mass);
  printf("\tcom_height   : %lf\n", com_height);
  printf("\tstep_time_min: %lf\n", step_time_min);
  printf("\tfoot_vel_max : %lf\n", foot_vel_max);
  printf("\tstep_height  : %lf\n", step_height);
  printf("environment:\n");
  printf("\tgravity : %lf\n", gravity);
  printf("\tfriction: %lf\n", friction);
  printf("foot: foot_r\n");
  for (size_t i = 0; i < foot_r.size(); i++) {
    printf("\t%lf, %lf\n", foot_r[i].x, foot_r[i].y);
  }
  printf("foot: foot_l\n");
  for (size_t i = 0; i < foot_l.size(); i++) {
    printf("\t%lf, %lf\n", foot_l[i].x, foot_l[i].y);
  }
  if(equalStr(robot_name, "nao") ) {
    for (int i = 0; i < NUM_LINK; i++) {
      printf("link: %s\n", link_name[i].c_str() );
      printf("\ttrn: %lf, %lf, %lf\n", trn[i].x(), trn[i].y(), trn[i].z() );
      printf("\taxis: %lf, %lf, %lf\n", axis[i].x(), axis[i].y(), axis[i].z() );
      printf("\tlimit: %f, %f\n", limit[i][LIMIT_LOWER], limit[i][LIMIT_UPPER]);
      printf("\tcom: %lf, %lf, %lf\n", com[i].x(), com[i].y(), com[i].z() );
      printf("\tmass: %lf\n", mass[i]);
    }
  }
  printf("-------------------------------------------\n");
}

} // namespace Capt
