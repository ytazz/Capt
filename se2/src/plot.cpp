#include "plot.h"

using namespace std;

namespace Capt {

Plot::Plot(){
}

Plot::~Plot() {
}

void Plot::Read(Scenebuilder::XMLNode* node){
	node->Get(st.swg[0], ".swg_x"    );
	node->Get(st.swg[1], ".swg_y"    );
	node->Get(st.swg[2], ".swg_r"    );
	node->Get(st.icp[0], ".icp_x"    );
	node->Get(st.icp[1], ".icp_y"    );
	node->Get(nmin     , ".nmin"     );
	node->Get(nmax     , ".nmax"     );
	node->Get(angle_div, ".angle_div");
}

void Plot::SetCaptureInput(Input in, int nstep){
	cap_input.push_back(make_pair(in, nstep));
}

void Plot::PrintLandingRegion(const string& filename, const Capturability::Region& r){
	vector<vec2_t> vertex;
	if(r.type == Capturability::Region::Type::Rectangle){
		vertex.push_back(vec2_t(r.min.x, r.min.y));
		vertex.push_back(vec2_t(r.min.x, r.max.y));
		vertex.push_back(vec2_t(r.max.x, r.max.y));
		vertex.push_back(vec2_t(r.max.x, r.min.y));
		vertex.push_back(vec2_t(r.min.x, r.min.y));
	}
	if(r.type == Capturability::Region::Type::Radial){
		real_t dangle = 2.0*r.angle/(real_t)angle_div;
		real_t angle  = pi/2.0 - r.angle;
		real_t dist   = r.near;
		int    i = 0;
		for( ; i < angle_div; i++){
			vertex.push_back(dist*vec2_t(cos(angle), sin(angle)));
			angle += dangle;
		}
		vertex.push_back(dist*vec2_t(cos(angle), sin(angle)));
		dist = r.far;
		for( ; i >= 0; i--){
			vertex.push_back(dist*vec2_t(cos(angle), sin(angle)));
			angle -= dangle;
		}
		vertex.push_back(vertex.front());
	}
	for(vec2_t& v : vertex)
		v = CartesianToGraph(v);

	FILE *fp = fopen(filename.c_str(), "w");
	for(vec2_t& v : vertex) {
		fprintf(fp, "%f %f\n", v.x, v.y);
	}
	fclose(fp);
}

void Plot::PrintIcp(const string& filename, const vec2_t& icp){
	vec2_t point = CartesianToGraph(icp);
	FILE  *fp    = fopen(filename.c_str(), "w");
	fprintf(fp, "%f %f\n", point.x, point.y);
	fclose(fp);
}

void Plot::PrintFoot(const string& filename, const vec3_t& pose){
	vec2_t pt[4];
	pt[0] = vec2_t(cap->cop_x.min, cap->cop_y.min);
	pt[1] = vec2_t(cap->cop_x.max, cap->cop_y.min);
	pt[2] = vec2_t(cap->cop_x.max, cap->cop_y.max);
	pt[3] = vec2_t(cap->cop_x.min, cap->cop_y.max);

	FILE *fp;
	fp = fopen(filename.c_str(), "w");
	vec2_t point;
	for (size_t i = 0; i <= 4; i++) {
		point = CartesianToGraph(vec2_t(pose[0], pose[1]) + mat2_t::Rot(pose[2])*pt[i%4]);
		fprintf(fp, "%f %f\n", point.x, point.y);
	}
	fclose(fp);
}

void Plot::Print(const string& basename){
	stringstream ss;
	for(int i = 0; i < cap->regions.size(); i++){
		ss.str("");
		ss << basename << "landing" << i << ".dat";
		PrintLandingRegion(ss.str(), cap->regions[i]);
	}
	PrintFoot(basename + "sup.dat", vec3_t(0.0, 0.0, 0.0));
	PrintFoot(basename + "swg.dat", st.swg               );
	PrintIcp (basename + "icp.dat", st.icp               );

	// mapをグラフ上の対応する点に変換
	for(int n = 0; n < 10; n++){
		stringstream ss;
		ss << basename << "data" << n << ".dat";
		FILE* file;
		file = fopen(ss.str().c_str(), "w");

		for(int i = 0; i < (int)cap_input.size(); i++){
			if(cap_input[i].second == n){
				vec2_t cop  = CartesianToGraph(cap_input[i].first.cop .x, cap_input[i].first.cop .y);
				vec2_t land = CartesianToGraph(cap_input[i].first.land.x, cap_input[i].first.land.y);
				fprintf(file, "%f %f %f %f %d\n", cop.x, cop.y, land.x, land.y, cap_input[i].second);
			}
		}

		fclose(file);
	}
}

vec2_t Plot::CartesianToGraph(vec2_t point){
  return vec2_t(-point.y, point.x);
}

vec2_t Plot::CartesianToGraph(float x, float y){
  return CartesianToGraph(vec2_t(x, y));
}

}
