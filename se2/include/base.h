#pragma once

#include <vector>
#include <cmath>

#include <sbxml.h>
using namespace Scenebuilder;

namespace Capt {

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

} // namespace Capt
