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
	CaptureBasin basin;
	cap.GetCaptureBasin(plot.st, 0, plot.nmax, basin);
	printf("get done: %d\n", basin.size());

	for(CaptureState& cs : basin){
		State stnext;
		stnext.swg = cap.grid->xyzr[cap.swg_to_xyzr[cs.swg_id]];
		stnext.icp = cap.grid->xy  [cs.icp_id];
		Input in   = cap.CalcInput(plot.st, stnext, in.tau);
		plot.SetCaptureInput(in, cs.nstep);
	}

	plot.Print("data/");

	return 0;
}
