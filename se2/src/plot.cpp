﻿#include "plot.h"

using namespace std;

namespace Capt {

Plot::Plot(){
}

Plot::~Plot() {
}

void Plot::Read(Scenebuilder::XMLNode* node){
	node->Get(st.swg[0], ".swg_x"    );
	node->Get(st.swg[1], ".swg_y"    );
	node->Get(st.swg[2], ".swg_z"    );
	node->Get(st.swg[3], ".swg_r"    );
	node->Get(st.icp[0], ".icp_x"    );
	node->Get(st.icp[1], ".icp_y"    );
	node->Get(nmax     , ".nmax"     );
	node->Get(angle_div, ".angle_div");
}

void Plot::SetCaptureInput(Input in, int nstep){
	cap_input.push_back(make_pair(in, nstep));
}

void Plot::PrintLandingRegion(const string& filename){
	vector<vec2_t> vertex;

	real_t dangle = 2.0*cap->swg_angle/(real_t)angle_div;
	real_t angle  = pi/2.0 - cap->swg_angle;
	real_t dist   = cap->swg_near;
	int    i = 0;
	for( ; i <= angle_div; i++){
		vertex.push_back(dist*vec2_t(cos(angle), sin(angle)));
		angle += dangle;
	}
	dist = cap->swg_far;
	for( ; i >= 0; i--){
		vertex.push_back(dist*vec2_t(cos(angle), sin(angle)));
		angle -= dangle;
	}
	vertex.push_back(vertex.front());
	/*
	vertex[0] << cap->swg_x.min, cap->swg_y.min;
	vertex[1] << cap->exc_x.min, cap->swg_y.min;
	vertex[2] << cap->exc_x.min, cap->exc_y.max;
	vertex[3] << cap->exc_x.max, cap->exc_y.max;
	vertex[4] << cap->exc_x.max, cap->swg_y.min;
	vertex[5] << cap->swg_x.max, cap->swg_y.min;
	vertex[6] << cap->swg_x.max, cap->swg_y.max;
	vertex[7] << cap->swg_x.min, cap->swg_y.max;
	vertex[8] << cap->swg_x.min, cap->swg_y.min;
	*/
	for(int i = 0; i < vertex.size(); i++)
		vertex[i] = CartesianToGraph(vertex[i]);

	FILE *fp = fopen(filename.c_str(), "w");
	for(int i = 0; i < vertex.size(); i++) {
		fprintf(fp, "%f %f\n", vertex[i].x(), vertex[i].y() );
	}
	fclose(fp);
}

void Plot::PrintIcp(const string& filename, const vec2_t& icp){
	vec2_t point = CartesianToGraph(icp);
	FILE  *fp    = fopen(filename.c_str(), "w");
	fprintf(fp, "%f %f\n", point.x(), point.y() );
	fclose(fp);
}

void Plot::PrintFoot(const string& filename, const vec4_t& pose){
	vec2_t pt[4];
	pt[0] = vec2_t(cap->cop_x.min, cap->cop_y.min);
	pt[1] = vec2_t(cap->cop_x.max, cap->cop_y.min);
	pt[2] = vec2_t(cap->cop_x.max, cap->cop_y.max);
	pt[3] = vec2_t(cap->cop_x.min, cap->cop_y.max);

	FILE *fp;
	fp = fopen(filename.c_str(), "w");
	vec2_t point;
	for (size_t i = 0; i <= 4; i++) {
		point = CartesianToGraph(vec2_t(pose[0], pose[1]) + Eigen::Rotation2D<real_t>(pose[3])*pt[i%4]);
		fprintf(fp, "%f %f\n", point.x(), point.y());
	}
	fclose(fp);
}

void Plot::Print(const string& basename){
	PrintLandingRegion(basename + "landing.dat");
	PrintFoot(basename + "sup.dat", vec4_t(0.0, 0.0, 0.0, 0.0));
	PrintFoot(basename + "swg.dat", st.swg                    );
	PrintIcp (basename + "icp.dat", st.icp                    );

	// mapをグラフ上の対応する点に変換
	FILE *fp;
	fp = fopen((basename + "data.dat").c_str(), "w");

	for(int i = 0; i < (int)cap_input.size(); i++){
		vec2_t cop  = CartesianToGraph(cap_input[i].first.cop .x(), cap_input[i].first.cop .y());
		vec2_t land = CartesianToGraph(cap_input[i].first.land.x(), cap_input[i].first.land.y());
		fprintf(fp, "%f %f %f %f %d\n", cop.x(), cop.y(), land.x(), land.y(), cap_input[i].second);
	}

	fclose(fp);
}

vec2_t Plot::CartesianToGraph(vec2_t point){
  return vec2_t(-point.y(), point.x());
}

vec2_t Plot::CartesianToGraph(float x, float y){
  return CartesianToGraph(vec2_t(x, y));
}

}
