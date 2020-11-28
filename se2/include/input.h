#pragma once

#include "base.h"
#include <iostream>
#include <string>

namespace Capt {

struct Input {
	vec2_t cop;
	vec3_t land;

	void Set(vec2_t _cop, vec3_t _land){
		cop  = _cop ;
		land = _land;
	}

	Input(){}
	Input(vec2_t _cop, vec3_t _land){
		Set(_cop, _land);
	}

};

} // namespace Capt
