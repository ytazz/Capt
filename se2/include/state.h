#pragma once

#include "base.h"
#include <iostream>
#include <string>

namespace Capt {

struct State {
	vec2_t icp;
	vec4_t swg;

	void set(vec2_t _icp, vec4_t _swg){
		icp = _icp;
		swg = _swg;
	}

	State(){}
	State(vec2_t _icp, vec4_t _swg){
		set(_icp, _swg);
	}

};

} // namespace Capt
