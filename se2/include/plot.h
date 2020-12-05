#pragma once

#include "capturability.h"
#include "grid.h"
#include "input.h"
#include "state.h"

namespace Capt {

class Plot{
public:
	State  st;

	Capturability* cap;

	vec2_t CartesianToGraph(vec2_t point);
	vec2_t CartesianToGraph(float x, float y);

	int c_num;    // number of color

	std::vector< std::pair<Input, int> > cap_input;

public:
	// Capture Regionのデータを格納するCapture Map
	void SetCaptureInput(Input in, int nstep);

	void PrintFootRegion();
	void PrintState     (State state);
	void PrintIcp       (vec2_t icp);
	void PrintSwg       (vec4_t swg);
	void Print();

	void Read(Scenebuilder::XMLNode* node);

	 Plot();
	~Plot();
};
}
