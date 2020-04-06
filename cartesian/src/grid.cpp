#include "grid.h"

namespace Capt {

int Grid1D::round(float v){
  return std::min(std::max(0, Capt::round((v - min)/stp)), num);
}

void Grid1D::indexRange(float fmin, float fmax, int& imin, int& imax){
  imin = std::min(std::max(0, (int)ceil((fmin - min)/stp)), num);
  imax = std::min(std::max(0, (int)ceil((fmax - min)/stp)), num);
}

Grid::Grid(Param *param) : param(param) {
  state.clear();

  // icp
  param->read(&icp_x.min, "icp_x_min");
  param->read(&icp_x.max, "icp_x_max");
  param->read(&icp_x.stp, "icp_x_stp");
  param->read(&icp_y.min, "icp_y_min");
  param->read(&icp_y.max, "icp_y_max");
  param->read(&icp_y.stp, "icp_y_stp");
  // swf
  param->read(&swf_x.min, "swf_x_min");
  param->read(&swf_x.max, "swf_x_max");
  param->read(&swf_x.stp, "swf_x_stp");
  param->read(&swf_y.min, "swf_y_min");
  param->read(&swf_y.max, "swf_y_max");
  param->read(&swf_y.stp, "swf_y_stp");
  param->read(&swf_z.min, "swf_z_min");
  param->read(&swf_z.max, "swf_z_max");
  param->read(&swf_z.stp, "swf_z_stp");

  // num of each grid
  param->read(&icp_x.num, "icp_x_num");
  param->read(&icp_y.num, "icp_y_num");
  param->read(&swf_x.num, "swf_x_num");
  param->read(&swf_y.num, "swf_y_num");
  param->read(&swf_z.num, "swf_z_num");


  // generate array of state values
  float icp_x_, icp_y_;
  float swf_x_, swf_y_, swf_z_;
  int i, j, k, l, m;
  for (i = 0, icp_x_ = icp_x.min; i < icp_x.num; i++, icp_x_ += icp_x.stp)
  for (j = 0, icp_y_ = icp_y.min; j < icp_y.num; j++, icp_y_ += icp_y.stp)
  for (k = 0, swf_x_ = swf_x.min; k < swf_x.num; k++, swf_x_ += swf_x.stp)
  for (l = 0, swf_y_ = swf_y.min; l < swf_y.num; l++, swf_y_ += swf_y.stp)
  for (m = 0, swf_z_ = swf_z.min; m < swf_z.num; m++, swf_z_ += swf_z.stp)
    state.push_back(State(icp_x_, icp_y_, swf_x_, swf_y_, swf_z_));

  for (i = 0, icp_x_ = icp_x.min; i < icp_x.num; i++, icp_x_ += icp_x.stp)
  for (j = 0, icp_y_ = icp_y.min; j < icp_y.num; j++, icp_y_ += icp_y.stp)
    icp.push_back(vec2_t(icp_x_, icp_y_));

  for (k = 0, swf_x_ = swf_x.min; k < swf_x.num; k++, swf_x_ += swf_x.stp)
  for (l = 0, swf_y_ = swf_y.min; l < swf_y.num; l++, swf_y_ += swf_y.stp)
  for (m = 0, swf_z_ = swf_z.min; m < swf_z.num; m++, swf_z_ += swf_z.stp)
    swf.push_back(vec3_t(swf_x_, swf_y_, swf_z_));

  icp_num = icp_y.num*icp_x.num;
  swf_num = swf_z.num*swf_y.num*swf_x.num;

}

Grid::~Grid() {
}

//vec2_t Grid::roundVec(vec2_t vec, vec2_t res){
//  return vec2_t(res.x()*round(vec.x()/res.x()), res.y()*round(vec.y()/res.y()));
//}

int Grid::roundState(State state_) {
  int icp_x_id = icp_x.round(state_.icp.x());
  int icp_y_id = icp_y.round(state_.icp.y());
  int swf_x_id = swf_x.round(state_.swf.x());
  int swf_y_id = swf_y.round(state_.swf.y());
  int swf_z_id = swf_z.round(state_.swf.z());

  return getStateIndex(getIcpIndex(icp_x_id, icp_y_id), getSwfIndex(swf_x_id, swf_y_id, swf_z_id));
}

//int Grid::getStateIndex(State state_) {
//  return roundState(state_).id;
//}

/*
bool Grid::existState(int state_id) {
  return (0 <= state_id && state_id < (int)state.size());
}

bool Grid::existState(State state_) {
  return (state_.icp.x() >= icp_x.min - icp_x.stp / 2.0f &&
          state_.icp.x() <  icp_x.max + icp_x.stp / 2.0f &&
          state_.icp.y() >= icp_y.min - icp_y.stp / 2.0f &&
          state_.icp.y() <  icp_y.max + icp_y.stp / 2.0f &&
          state_.swf.x() >= swf_x.min - swf_x.stp / 2.0f &&
          state_.swf.x() <  swf_x.max + swf_x.stp / 2.0f &&
          state_.swf.y() >= swf_y.min - swf_y.stp / 2.0f &&
          state_.swf.y() <  swf_y.max + swf_y.stp / 2.0f &&
          state_.swf.z() >= swf_z.min - swf_z.stp / 2.0f &&
          state_.swf.z() <  swf_z.max + swf_z.stp / 2.0f );
}
*/
int Grid::getIcpIndex(int icp_x_id, int icp_y_id){
  return icp_y.num*icp_x_id + icp_y_id;
}

int Grid::getSwfIndex(int swf_x_id, int swf_y_id, int swf_z_id){
  return swf_z.num*(swf_y.num*swf_x_id + swf_y_id) + swf_z_id;
}

int Grid::getStateIndex(int icp_id, int swf_id){
  return swf_num*icp_id + swf_id;
}

/*
vec2_t Grid::getIcp(int index){
  int idx = index / icp_y.num;
  int idy = index % icp_y.num;
  if( 0 <= idx && idx < icp_x.num &&
      0 <= idy && idy < icp_y.num ) {
    return vec2_t(icp_x.min + icp_x.stp * idx,
                  icp_y.min + icp_y.stp * idy);
  }
  return vec2_t(-1, -1);
}

vec2_t Grid::getSwf(int index){
  int idx = index / swf_y_num;
  int idy = index % swf_y_num;
  if( 0 <= idx && idx < swf_x_num &&
      0 <= idy && idy < swf_y_num ) {
    return vec2_t(swf_x[CaptEnum::MIN] + swf_x[CaptEnum::STP] * idx,
                  swf_y[CaptEnum::MIN] + swf_y[CaptEnum::STP] * idy);
  }
  return vec2_t(-1, -1);
}

int Grid::indexIcp(vec2_t icp){
  int icp_x_id = round( ( icp.x() - icp_x[CaptEnum::MIN] ) / icp_x[CaptEnum::STP]);
  int icp_y_id = round( ( icp.y() - icp_y[CaptEnum::MIN] ) / icp_y[CaptEnum::STP]);
  if(0 <= icp_x_id && icp_x_id < icp_x_num &&
     0 <= icp_y_id && icp_y_id < icp_y_num) {
     return icp_y_num * icp_x_id + icp_y_id;
  }
  return -1;
}

int Grid::indexSwf(vec2_t swf){
  int swf_x_id = round( ( swf.x() - swf_x[CaptEnum::MIN] ) / swf_x[CaptEnum::STP]);
  int swf_y_id = round( ( swf.y() - swf_y[CaptEnum::MIN] ) / swf_y[CaptEnum::STP]);
  // int swf_z_id = round( ( swf.z() - swf_z[CaptEnum::MIN] ) / swf_z[CaptEnum::STP]);
  int swf_z_id = 0;
  if(0 <= swf_x_id && swf_x_id < swf_x_num &&
     0 <= swf_y_id && swf_y_id < swf_y_num &&
     0 <= swf_z_id && swf_z_id < swf_z_num ) {
     return swf_z_num * (swf_y_num * swf_x_id + swf_y_id) + swf_z_id;
  }
  return -1;
}
*/

int Grid::getNumIcp() {
  return icp_x.num*icp_y.num;
}

int Grid::getNumSwf() {
  return swf_x.num*swf_y.num*swf_z.num;
}

int Grid::getNumState() {
  return getNumIcp()*getNumSwf();
}

} // namespace Capt