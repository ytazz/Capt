#pragma once

#include "base.h"
#include <iostream>
#include <string>

namespace Capt {

struct Input {
	vec2_t cop;   //< cop x,y
	vec3_t land;  //< landing pose x,y,r
	real_t tau;   //< step duration

	void Set(vec2_t _cop, vec3_t _land, real_t _tau){
		cop  = _cop ;
		land = _land;
		tau  = _tau ;
	}

	Input(){}
	Input(vec2_t _cop, vec3_t _land, real_t _tau){
		Set(_cop, _land, _tau);
	}

};

} // namespace Capt
