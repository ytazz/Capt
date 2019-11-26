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

enum OccupancyType {
  NONE,
  EMPTY,
  OBSTACLE,
  OPEN,
  CLOSED,
  GOAL
};

} // namespace Capt

#endif // __BASE_H__