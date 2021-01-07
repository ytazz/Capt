#pragma once

#include "capturability.h"
#include "grid.h"
#include "input.h"
#include "state.h"

namespace Capt {

class Plot{
public:
	State  st;
	Input  in;
	int    nmax;
	int    angle_div;

	Capturability* cap;

	vec2_t CartesianToGraph(vec2_t point);
	vec2_t CartesianToGraph(float x, float y);

	int c_num;    // number of color

	std::vector< std::pair<Input, int> > cap_input;

public:
	// Capture Regionのデータを格納するCapture Map
	void SetCaptureInput(Input in, int nstep);

	void PrintLandingRegion(const string& filename);
	void PrintIcp          (const string& filename, const vec2_t& icp );
	void PrintFoot         (const string& filename, const vec4_t& pose);
	void Print             (const string& basename);

	void Read(Scenebuilder::XMLNode* node);

	 Plot();
	~Plot();
};
}
