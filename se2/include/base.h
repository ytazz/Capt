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
inline int Round(real_t f){
	int i = (int)f;

	real_t d = f - i;
	if(d > 0.0) {
		if (d >= 0.5) {
			i++;
		}
	}
	else{
		if (d <= -0.5) {
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
