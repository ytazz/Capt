#include "plot.h"
#include <iostream>

using namespace std;
using namespace Capt;

int main(int argc, char const *argv[]) {
	Scenebuilder::XML xmlCapt;
	xmlCapt.Load("conf/capt.xml");

	Scenebuilder::XML xmlPlot;
	xmlPlot.Load("conf/plot.xml");

	Capturability cap;
	Plot          plot;

	plot.cap = &cap;
	
	cap .Read(xmlCapt.GetRootNode());
	plot.Read(xmlPlot.GetRootNode());

	cap.Load("data/");
	printf("load done\n");

	printf("get cap regions\n");
	CaptureBasin    basin;
	vector<vec2_t>  tau_range_valid;
	cap.GetCaptureBasin(plot.st, plot.nmin, plot.nmax, basin, tau_range_valid);
	printf("get done: %d\n", basin.size());

	Input in;
	for(int i = 0; i < basin.size(); i++){
		CaptureState& cs = basin[i];

		State stnext;
		stnext.swg = cap.grid->xyr[cap.swg_to_xyr[cs.swg_id]];
		stnext.icp = cap.grid->xy [cs.icp_id];
		//in.tau = cap.CalcMinDuration(plot.st.swg, stnext.swg);
		in.tau = tau_range_valid[i][0];
		cap.CalcInput(plot.st, stnext, in);
		plot.cap_input.push_back(make_pair(in, cs.nstep));
	}

	plot.Print("data/");

	return 0;
}
