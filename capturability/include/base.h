#ifndef __BASE_H__
#define __BASE_H__

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

namespace Capt {

typedef Eigen::Vector2f vec2_t;
typedef Eigen::Vector3f vec3_t;
typedef std::vector<vec2_t> arr2_t;

float  dot(vec2_t v1, vec2_t v2);
float  cross(vec2_t v1, vec2_t v2);
vec2_t normal(vec2_t v);

int round(double val);

} // namespace Capt

#endif // __BASE_H__