#include "grid.h"

namespace Capt {

void Grid1D::init(){
	num = Capt::round((max - min)/stp) + 1;
	val.resize(num);
	for(int i = 0; i < num; i++)
		val[i] = min + stp*i;

	printf("grid1d init: min %f  max %f  stp %f  num %d\n", min, max, stp, num);
}

int Grid1D::round(real_t v){
	return std::min(std::max(0, Capt::round((v - min)/stp)), num-1);
}

void Grid1D::indexRange(real_t fmin, real_t fmax, int& imin, int& imax){
	  imin = std::min(std::max(0, (int)ceil((fmin - min)/stp)), num-1);
	  imax = std::min(std::max(0, (int)ceil((fmax - min)/stp)), num-1);
}

int Grid2D::num(){
	return axis[0]->num*axis[1]->num;
}

int Grid2D::toIndex(Index2D idx2){
	return axis[0]->num*idx2[1] + idx2[0];
}

void Grid2D::fromIndex(int idx, Index2D& idx2){
	idx2[0] = idx%axis[0]->num; idx/=axis[0]->num;
	idx2[1] = idx;
}

Index2D Grid2D::round(vec2_t v){
  return Index2D(axis[0]->round(v[0]), axis[1]->round(v[1]));
}

vec2_t Grid2D::operator[](int idx){
  Index2D idx2;
  fromIndex(idx, idx2);
  return operator[](idx2);
}

vec2_t Grid2D::operator[](Index2D idx2){
  return vec2_t(axis[0]->val[idx2[0]], axis[1]->val[idx2[1]]);
}

int Grid3D::num(){
  return axis[0]->num*axis[1]->num*axis[2]->num;
}

int Grid3D::toIndex(Index3D idx3){
  return axis[0]->num*(axis[1]->num*idx3[2] + idx3[1]) + idx3[0];
}

void Grid3D::fromIndex(int idx, Index3D& idx3){
  idx3[0] = idx%axis[0]->num; idx/=axis[0]->num;
  idx3[1] = idx%axis[1]->num; idx/=axis[1]->num;
  idx3[2] = idx;
}

Index3D Grid3D::round(vec3_t v){
  return Index3D(axis[0]->round(v[0]), axis[1]->round(v[1]), axis[2]->round(v[2]));
}

vec3_t Grid3D::operator[](int idx){
  Index3D idx3;
  fromIndex(idx, idx3);
  return operator[](idx3);
}

vec3_t Grid3D::operator[](Index3D idx3){
  return vec3_t(axis[0]->val[idx3[0]], axis[1]->val[idx3[1]], axis[2]->val[idx3[2]]);
}

Grid::Grid(){

}

Grid::~Grid() {
}

void Grid::Read(Scenebuilder::XMLNode* node){
	node->Get(x.min, ".x_min");
	node->Get(x.max, ".x_max");
	node->Get(x.stp, ".x_stp");

	node->Get(y.min, ".y_min");
	node->Get(y.max, ".y_max");
	node->Get(y.stp, ".y_stp");

	node->Get(z.min, ".z_min");
	node->Get(z.max, ".z_max");
	node->Get(z.stp, ".z_stp");

	node->Get(r.min, ".r_min");
	node->Get(r.max, ".r_max");
	node->Get(r.stp, ".r_stp");

	node->Get(t.min, ".t_min");
	node->Get(t.max, ".t_max");
	node->Get(t.stp, ".t_stp");

	x.init();
	y.init();
	z.init();
	r.init();
	t.init();

	xy.axis[0] = &x;
	xy.axis[1] = &y;

	xyz.axis[0] = &x;
	xyz.axis[1] = &y;
	xyz.axis[2] = &z;

	xyzr.axis[0] = &x;
	xyzr.axis[1] = &y;
	xyzr.axis[2] = &z;
	xyzr.axis[3] = &r;

	xyt.axis[0] = &x;
	xyt.axis[1] = &y;
	xyt.axis[2] = &t;

}

} // namespace Capt