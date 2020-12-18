#pragma once

#include "base.h"
#include <iostream>
#include <string>

namespace Capt {

struct State{
	vec2_t icp;  //< icp  x,y
	vec4_t swg;  //< swing foot pose x,y,z,r

	void Set(vec2_t _icp, vec4_t _swg){
		icp = _icp;
		swg = _swg;
	}

	State(){}
	State(vec2_t _icp, vec4_t _swg){
		Set(_icp, _swg);
	}
};

} // namespace Capt
