#include "plot.h"

using namespace std;

namespace Capt {

Plot::Plot(){
}

Plot::~Plot() {
}

void Plot::Read(Scenebuilder::XMLNode* node){
	node->Get(st.swg[0], ".swg_x");
	node->Get(st.swg[1], ".swg_y");
	node->Get(st.swg[2], ".swg_z");
	node->Get(st.swg[3], ".swg_r");
	node->Get(st.icp[0], ".icp_x");
	node->Get(st.icp[1], ".icp_y");
}

void Plot::PrintFootRegion(){
	vec2_t vertex[9];
	vertex[0] << cap->swg_x.min, cap->swg_y.min;
	vertex[1] << cap->exc_x.min, cap->swg_y.min;
	vertex[2] << cap->exc_x.min, cap->exc_y.max;
	vertex[3] << cap->exc_x.max, cap->exc_y.max;
	vertex[4] << cap->exc_x.max, cap->swg_y.min;
	vertex[5] << cap->swg_x.max, cap->swg_y.min;
	vertex[6] << cap->swg_x.max, cap->swg_y.max;
	vertex[7] << cap->swg_x.min, cap->swg_y.max;
	vertex[8] << cap->swg_x.min, cap->swg_y.min;
	
	for(int i = 0; i < 9; i++)
		vertex[i] = CartesianToGraph(vertex[i]);

	FILE *fp = fopen("data/foot_region.dat", "w");
	for(int i = 0; i < 9; i++) {
		fprintf(fp, "%lf %lf\n", vertex[i].x(), vertex[i].y() );
	}
	fclose(fp);
}

void Plot::PrintState(State state){
	PrintIcp(state.icp);
	PrintSwg(state.swg);
}

void Plot::PrintIcp(vec2_t icp){
	vec2_t point = CartesianToGraph(icp);
	FILE  *fp    = fopen("data/icp.dat", "w");
	fprintf(fp, "%lf %lf\n", point.x(), point.y() );
	fclose(fp);
}

void Plot::PrintSwg(vec4_t swg){
	arr2_t foot_r, foot_l;

	FILE *fp;
	fp = fopen("data/foot_r.dat", "w");
	vec2_t point;
	for (size_t i = 0; i < foot_r.size(); i++) {
		// グラフ座標に合わせる
		point = CartesianToGraph(foot_r[i]);
		fprintf(fp, "%lf %lf\n", point.x(), point.y() );
	}
	fclose(fp);

	fp = fopen("data/foot_l.dat", "w");
	for (size_t i = 0; i < foot_l.size(); i++) {
		// グラフ座標に合わせる
		point = CartesianToGraph(foot_l[i]);
		fprintf(fp, "%lf %lf\n", point.x(), point.y() );
	}
	fclose(fp);
}

void Plot::SetCaptureInput(Input in, int nstep){
	cap_input.push_back(make_pair(in, nstep));
}

void Plot::Print(){
	PrintFootRegion();

	// mapをグラフ上の対応する点に変換
	FILE *fp;
	fp = fopen("data/data.dat", "w");

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
