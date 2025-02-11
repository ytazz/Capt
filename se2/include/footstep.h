﻿#pragma once

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
		real_t  duration;
		
		int     side;
		vec3_t  footPos   [2];
		vec3_t  footVel   [2];
		real_t  footOri   [2];
		real_t  footAngvel[2];
		vec3_t  cop;
		vec3_t  icp;
		real_t  telapsed;

		void Read (Scenebuilder::XMLNode* node);
		void Print();

		Step();
	};

	std::vector<Step> steps;

	int cur;  //< current footstep index

	void Read(Scenebuilder::XMLNode* node);
	void Calc(Capturability* cap, Swing* swing);

};

}
