#ifndef __BASE_H__
#define __BASE_H__

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <cmath>

#include <sbxml.h>

namespace Capt {

// Eigen typedefs
typedef float           real_t;
typedef Eigen::Vector2i vec2i_t;
typedef Eigen::Vector2f vec2_t;
typedef Eigen::Vector3f vec3_t;
typedef Eigen::Vector4f vec4_t;

//#define EPSILON 0.001
const real_t pi = 3.14159265358979f;

//
inline real_t WrapRadian(real_t theta){
	while(theta >  pi) theta -= 2*pi;
	while(theta < -pi) theta += 2*pi;
	return theta;
}

// round to the nearest integer
inline int Round(float f){
	int i = (int)f;

	float d = f - i;
	if(d > 0.0f) {
		if (d >= 0.5f) {
			i++;
		}
	}
	else{
		if (d <= -0.5f) {
			i--;
		}
	}

	return i;
}

// foot enum
struct Foot{
	enum{
		Right,
		Left,
	};
};

// trajectory decision variables
struct Step {
	Foot   s_suf;
	vec3_t pos;
	real_t ori;
	vec3_t cop;
	vec3_t icp;
};

struct Footstep : public std::vector<Step>{
	int cur;  //< current footstep index
};

} // namespace Capt

#endif // __BASE_H__