#ifndef __MODEL_H__
#define __MODEL_H__

#include "loader.h"
#include "vector.h"
#include <iostream>
#include <math.h>
#include <string.h>
#include <vector>

namespace CA {

enum EModelElement {
  MODEL_ELE_NONE,
  MODEL_ELE_ROBOT,
  MODEL_ELE_UNIT,
  MODEL_ELE_PHYSICS,
  MODEL_ELE_ENVIRONMENT,
  MODEL_ELE_LINK,
  MODEL_ELE_LINK_JOINT,
  MODEL_ELE_LINK_PHYSICS,
  MODEL_ELE_FOOT,
  MODEL_ELE_SHAPE
};

enum ELink {
  TORSO,
  HEAD_YAW,
  HEAD_PITCH,
  R_SHOULDER_PITCH,
  R_SHOULDER_ROLL,
  R_ELBOW_YAW,
  R_ELBOW_ROLL,
  R_WRIST_YAW,
  L_SHOULDER_PITCH,
  L_SHOULDER_ROLL,
  L_ELBOW_YAW,
  L_ELBOW_ROLL,
  L_WRIST_YAW,
  R_HIP_YAWPITCH,
  R_HIP_ROLL,
  R_HIP_PITCH,
  R_KNEE_PITCH,
  R_ANKLE_PITCH,
  R_ANKLE_ROLL,
  R_FOOT,
  L_HIP_YAWPITCH,
  L_HIP_ROLL,
  L_HIP_PITCH,
  L_KNEE_PITCH,
  L_ANKLE_PITCH,
  L_ANKLE_ROLL,
  L_FOOT,
  NUM_LINK,
  LINK_NONE
};

enum EFoot { FOOT_NONE, FOOT_R, FOOT_L };

enum EShape {
  SHAPE_NONE,
  SHAPE_BOX,
  SHAPE_POLYGON,
  SHAPE_CIRCLE,
  SHAPE_REVERSE
};

enum ELimit { LIMIT_LOWER, LIMIT_UPPER, NUM_LIMIT };

class Model : public Loader {

public:
  explicit Model(const std::string &name);
  ~Model();

  void callbackElement(const std::string &name, const bool is_start) override;
  void callbackAttribute(const std::string &name,
                         const std::string &value) override;

  std::vector<Vector2> reverseShape(std::vector<Vector2> points);

  float getVal(const char *element_name, const char *attribute_name);
  std::string getStr(const char *element_name, const char *attribute_name);
  std::vector<Vector2> getVec(const char *element_name,
                              const char *attribute_name);
  std::vector<Vector2> getVec(const char *element_name,
                              const char *attribute_name, vec2_t translation);
  std::vector<Vector2> getVec(const char *element_name,
                              const char *attribute_name, vec3_t translation);
  float getLinkVal(ELink link, const char *attribute_name);
  float getLinkVal(int link_id, const char *attribute_name);
  vec3_t getLinkVec(ELink link, const char *attribute_name);
  vec3_t getLinkVec(int link_id, const char *attribute_name);
  void print();

private:
  EModelElement element;
  EFoot foot;
  EShape shape;
  ELink link;

  float pi;

  std::string robot_name;
  float unit_length, unit_mass, unit_time, unit_angle;
  float total_mass, com_height, step_time_min, foot_vel_max, step_height;
  float gravity, friction;

  std::vector<Vector2> foot_r, foot_l;

  std::string link_name[NUM_LINK];

  vec3_t trn[NUM_LINK];
  vec3_t axis[NUM_LINK];
  // limit[*][0] = lower limit, limit[*][1] = upper limit
  float limit[NUM_LINK][NUM_LIMIT];
  vec3_t com[NUM_LINK];
  float mass[NUM_LINK];
};

} // namespace CA

#endif // __MODEL_H__
