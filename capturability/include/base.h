#ifndef __BASE_H__
#define __BASE_H__

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

namespace Capt {

typedef Eigen::Vector2f vec2_t;
typedef Eigen::Vector2i vec2i_t;
typedef Eigen::Vector3f vec3_t;
typedef std::vector<vec2_t> arr2_t;

double dot(vec2_t v1, vec2_t v2);
double cross(vec2_t v1, vec2_t v2);
vec2_t normal(vec2_t v);
vec2_t mirror(vec2_t v);

int round(double val);

vec3_t vec2Tovec3(vec2_t vec2);
vec2_t vec3Tovec2(vec3_t vec3);

enum OccupancyType {
  NONE,
  EMPTY,
  OBSTACLE,
  OPEN,
  CLOSED,
  GOAL
};

enum Foot { FOOT_NONE, FOOT_R, FOOT_L };

struct Footstep {
  Foot   suf;
  vec3_t pos;
  vec3_t icp;
  vec3_t cop;

  void substitute(Foot suf, vec2_t pos, vec2_t icp, vec2_t cop){
    this->suf     = suf;
    this->pos.x() = pos.x();
    this->pos.y() = pos.y();
    this->pos.z() = 0.0;
    this->icp.x() = icp.x();
    this->icp.y() = icp.y();
    this->icp.z() = 0.0;
    this->cop.x() = cop.x();
    this->cop.y() = cop.y();
    this->cop.z() = 0.0;
  }
};

} // namespace Capt

#endif // __BASE_H__