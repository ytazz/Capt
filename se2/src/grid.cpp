#include "grid.h"

namespace Capt {

//-------------------------------------------------------------------------------------------------

Grid1D::Grid1D(){
	min = 0.0f;
	max = 0.0f;
	stp = 1.0f;
	num = 0;
}

void Grid1D::Init(){
	num = Capt::Round((max - min)/stp) + 1;
	idx.resize(num);
	val.resize(num);
	for(int i = 0; i < num; i++){
		idx[i] = i;
		val[i] = min + stp*i;
	}

	printf("grid1d init: min %f  max %f  stp %f  num %d\n", min, max, stp, num);
}

void Grid1D::Read(Scenebuilder::XMLNode* node){
	node->Get(min, ".min");
	node->Get(max, ".max");
	node->Get(stp, ".stp");
}

int Grid1D::Round(real_t v){
	return std::min(std::max(0, Capt::Round((v - min)/stp)), num-1);
}

void Grid1D::IndexRange(real_t fmin, real_t fmax, int& imin, int& imax){
	  imin = std::min(std::max(0, (int)ceil((fmin - min)/stp)), num-1);
	  imax = std::min(std::max(0, (int)ceil((fmax - min)/stp)), num-1);
}

void Grid1D::Neighbors(real_t v, vector<int>& idxs){
	idxs.clear();
	if(v < min){
		idxs.push_back(0);
		return;
	}
	if(v >= max){
		idxs.push_back(num-1);
		return;
	}
	int i = (int)floor((v - min)/stp);
	idxs.push_back(i+0);
	idxs.push_back(i+1);
}

//-------------------------------------------------------------------------------------------------

int Grid2D::Num(){
	return axis[0]->num*axis[1]->num;
}

int Grid2D::ToIndex(Index2D idx2){
	return axis[0]->num*idx2[1] + idx2[0];
}

void Grid2D::FromIndex(int idx, Index2D& idx2){
	idx2[0] = idx%axis[0]->num; idx/=axis[0]->num;
	idx2[1] = idx;
}

Index2D Grid2D::Round(const vec2_t& v){
  return Index2D(axis[0]->Round(v[0]), axis[1]->Round(v[1]));
}

void Grid2D::Neighbors(const vec2_t& v, vector<Index2D>& idxs){
	vector<int> idxs0, idxs1;
	axis[0]->Neighbors(v[0], idxs0);
	axis[1]->Neighbors(v[1], idxs1);

	idxs.clear();
	for(int i0 : idxs0)
	for(int i1 : idxs1)
		idxs.push_back(Index2D(i0, i1));
}

vec2_t Grid2D::operator[](int idx){
  Index2D idx2;
  FromIndex(idx, idx2);
  return operator[](idx2);
}

vec2_t Grid2D::operator[](Index2D idx2){
  return vec2_t(axis[0]->val[idx2[0]], axis[1]->val[idx2[1]]);
}

//-------------------------------------------------------------------------------------------------

int Grid3D::Num(){
  return axis[0]->num*axis[1]->num*axis[2]->num;
}

int Grid3D::ToIndex(Index3D idx3){
  return axis[0]->num*(axis[1]->num*idx3[2] + idx3[1]) + idx3[0];
}

void Grid3D::FromIndex(int idx, Index3D& idx3){
  idx3[0] = idx%axis[0]->num; idx/=axis[0]->num;
  idx3[1] = idx%axis[1]->num; idx/=axis[1]->num;
  idx3[2] = idx;
}

Index3D Grid3D::Round(const vec3_t& v){
  return Index3D(axis[0]->Round(v[0]), axis[1]->Round(v[1]), axis[2]->Round(v[2]));
}

void Grid3D::Neighbors(const vec3_t& v, vector<Index3D>& idxs){
	vector<int> idxs0, idxs1, idxs2;
	axis[0]->Neighbors(v[0], idxs0);
	axis[1]->Neighbors(v[1], idxs1);
	axis[2]->Neighbors(v[2], idxs2);

	idxs.clear();
	for(int i0 : idxs0)
	for(int i1 : idxs1)
	for(int i2 : idxs2)
		idxs.push_back(Index3D(i0, i1, i2));
}

vec3_t Grid3D::operator[](int idx){
  Index3D idx3;
  FromIndex(idx, idx3);
  return operator[](idx3);
}

vec3_t Grid3D::operator[](Index3D idx3){
  return vec3_t(axis[0]->val[idx3[0]], axis[1]->val[idx3[1]], axis[2]->val[idx3[2]]);
}

//-------------------------------------------------------------------------------------------------

int Grid4D::Num(){
  return axis[0]->num*
	     axis[1]->num*
	     axis[2]->num*
	     axis[3]->num;
}

int Grid4D::ToIndex(Index4D idx4){
  return axis[0]->num*( axis[1]->num*( axis[2]->num*idx4[3] + idx4[2]) + idx4[1]) + idx4[0];
}

void Grid4D::FromIndex(int idx, Index4D& idx4){
  idx4[0] = idx%axis[0]->num; idx/=axis[0]->num;
  idx4[1] = idx%axis[1]->num; idx/=axis[1]->num;
  idx4[2] = idx%axis[2]->num; idx/=axis[2]->num;
  idx4[3] = idx;
}

Index4D Grid4D::Round(vec4_t v){
  return Index4D(axis[0]->Round(v[0]), axis[1]->Round(v[1]), axis[2]->Round(v[2]), axis[3]->Round(v[3]));
}

vec4_t Grid4D::operator[](int idx){
  Index4D idx4;
  FromIndex(idx, idx4);
  return operator[](idx4);
}

vec4_t Grid4D::operator[](Index4D idx4){
  return vec4_t(axis[0]->val[idx4[0]], axis[1]->val[idx4[1]], axis[2]->val[idx4[2]], axis[3]->val[idx4[3]]);
}

//-------------------------------------------------------------------------------------------------

Grid::Grid(){

}

Grid::~Grid() {
}

void Grid::Read(Scenebuilder::XMLNode* node){
	x.Read(node->GetNode("x"));
	y.Read(node->GetNode("y"));
	r.Read(node->GetNode("r"));
	
	x.Init();
	y.Init();
	r.Init();

	xy.axis[0] = &x;
	xy.axis[1] = &y;

	xyr.axis[0] = &x;
	xyr.axis[1] = &y;
	xyr.axis[2] = &r;

}

} // namespace Capt