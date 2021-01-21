#define _CRT_SECURE_NO_WARNINGS

#include <capturability.h>
#include <iostream>
#include <chrono>

#include <sbtimer.h>
Timer timer;

using namespace std;
using namespace Capt;

const int nmax = 10;

int main(int argc, char const *argv[]) {
	Scenebuilder::XML xml;
	xml.Load("conf/capt.xml");
  
	Capturability cap;
	cap.Read(xml.GetRootNode());
	
	timer.CountUS();
	cap.Analyze();
	int comptime = timer.CountUS();

	cap.Save("data/");
	
	FILE* file = fopen("data/comptime.csv", "w");
	fprintf(file, "%d", comptime);
	fclose(file);

	file = fopen("data/basin_size.csv", "w");
	real_t dx = cap.grid->x.stp;
	real_t dy = cap.grid->y.stp;
	real_t dr = cap.grid->r.stp;
	real_t dv = (dx*dy*dr)*(dx*dy); 
	for(int n = 0; n < cap.cap_basin.size(); n++){
		int    sz      = cap.cap_basin[n].size();
		real_t vol     = ((real_t)sz)*dv;
		real_t log_vol = (vol == 0.0 ? 0.0 : log(vol));
		fprintf(file, "%d, %f, %f\n", sz, vol, log_vol);
	}
	fclose(file);

	return 0;
}