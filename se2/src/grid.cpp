#include "grid.h"

namespace Capt {

void Grid1D::init(){
  num = Capt::round((max - min)/stp) + 1;
  val.resize(num);
  for(int i = 0; i < num; i++)
    val[i] = min + stp*i;

  printf("grid1d init: min %f  max %f  stp %f  num %d\n", min, max, stp, num);
}

int Grid1D::round(float v){
  return std::min(std::max(0, Capt::round((v - min)/stp)), num-1);
}

void Grid1D::indexRange(float fmin, float fmax, int& imin, int& imax){
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

Grid::Grid(Param *param) : param(param) {
  //state.clear();

  param->read(&x.min, "grid_x_min");
  param->read(&x.max, "grid_x_max");
  param->read(&x.stp, "grid_x_stp");

  param->read(&y.min, "grid_y_min");
  param->read(&y.max, "grid_y_max");
  param->read(&y.stp, "grid_y_stp");

  param->read(&z.min, "grid_z_min");
  param->read(&z.max, "grid_z_max");
  param->read(&z.stp, "grid_z_stp");

  param->read(&t.min, "grid_t_min");
  param->read(&t.max, "grid_t_max");
  param->read(&t.stp, "grid_t_stp");

  x.init();
  y.init();
  z.init();
  t.init();

  xy.axis[0] = &x;
  xy.axis[1] = &y;

  xyz.axis[0] = &x;
  xyz.axis[1] = &y;
  xyz.axis[2] = &z;

  xyt.axis[0] = &x;
  xyt.axis[1] = &y;
  xyt.axis[2] = &t;
}

Grid::~Grid() {
}

} // namespace Capt