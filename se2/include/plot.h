#pragma once

#include "capturability.h"
#include "grid.h"
#include "input.h"
#include "state.h"

namespace Capt {

class Plot{
public:
	State  st;
	int    nmin;
	int    nmax;
	int    angle_div;

	Capturability* cap;

	vec2_t CartesianToGraph(vec2_t point);
	vec2_t CartesianToGraph(float x, float y);

	int c_num;    // number of color

	std::vector< std::pair<Input, int> > cap_input;

public:
	
	void PrintLandingRegion(const string& filename, const Capturability::Region& r);
	void PrintIcp          (const string& filename, const vec2_t& icp );
	void PrintFoot         (const string& filename, const vec3_t& pose);
	void PrintBasin        (const string& filename, int n);
	void Print             (const string& basename);

	void Read(Scenebuilder::XMLNode* node);

	 Plot();
	~Plot();
};
}
