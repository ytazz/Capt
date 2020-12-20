#pragma once

#include "base.h"

namespace Capt {

class Capturability;
class Swing;

// trajectory decision variables
struct Footstep{
	struct Step {
		real_t  stride;
		real_t  spacing;
		real_t  turn;
		
		int     side;
		vec3_t  footPos[2];
		real_t  footOri[2];
		vec3_t  cop;
		vec3_t  icp;
		real_t  duration;

		void Read (Scenebuilder::XMLNode* node);
		void Print();
	};

	std::vector<Step> steps;

	int cur;  //< current footstep index

	void Read(Scenebuilder::XMLNode* node);
	void Calc(Capturability* cap, Swing* swing);

};

}
