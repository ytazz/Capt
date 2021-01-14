#pragma once

#include "base.h"
#include <iostream>
#include <string>

namespace Capt {

struct State{
	vec3_t swg;  //< swing foot pose x,y,r
	vec2_t icp;  //< icp  x,y

	void Set(const vec3_t& _swg, const vec2_t& _icp){
		swg = _swg;
		icp = _icp;
	}

	State(){}
	State(const vec3_t& _swg, const vec2_t& _icp){
		Set(_swg, _icp);
	}
};

} // namespace Capt
