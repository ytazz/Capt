#pragma once

#include "base.h"

namespace Capt {

class Capturability;
class Swing;

// trajectory decision variables
struct Footstep{
	struct Step {
		Foot   s_suf;
		vec3_t pos;
		real_t ori;
		vec3_t cop;
		vec3_t icp;

		void Read(Scenebuilder::XMLNode* node);
	};

	std::vector<Step> steps;

	int cur;  //< current footstep index

	void Read(Scenebuilder::XMLNode* node);
	void Calc(Capturability* cap, Swing* swing);

};

}
