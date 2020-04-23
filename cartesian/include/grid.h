#ifndef __GRID_H__
#define __GRID_H__

#include "param.h"
#include "state.h"
#include "base.h"
#include <iostream>
#include <string>
#include <vector>
#include <array>

namespace Capt {

struct Grid1D{
  float min;
  float max;
  float stp;
  int   num;
  std::vector<float> val;

  void init ();
  int  round(float v);
  void indexRange(float fmin, float fmax, int& imin, int& imax);
};

struct Index2D : std::array<int, 2>{
  Index2D(){}
  Index2D(int i0, int i1){
      at(0) = i0;
      at(1) = i1;
  }
};
struct Index3D : std::array<int, 3>{
  Index3D(){}
  Index3D(int i0, int i1, int i2){
    at(0) = i0;
    at(1) = i1;
    at(2) = i2;
  }
};
struct Grid2D{
  Grid1D* axis[2];

  int     num      ();
  int     toIndex  (Index2D idx2);
  void    fromIndex(int idx, Index2D& idx2);
  Index2D round(vec2_t v);

  vec2_t operator[](int idx);
  vec2_t operator[](Index2D idx2);
};
struct Grid3D{
  Grid1D* axis[3];

  int     num      ();
  int     toIndex  (Index3D idx3);
  void    fromIndex(int idx, Index3D& idx3);
  Index3D round(vec3_t v);

  vec3_t operator[](int idx);
  vec3_t operator[](Index3D idx3);
};

class Grid {
public:
  Grid(Param *param);
  ~Grid();

  //int  roundIcp(vec2_t icp);
  //int  roundSwf(vec3_t swf);
  //int  roundState(State st);

  //int indexIcp(vec2_t icp);
  //int indexSwf(vec2_t swf);
  //int indexCop(vec2_t cop);

  //int  getIcpIndex  (int icp_x_id, int icp_y_id);
  //int  getSwfIndex  (int swf_x_id, int swf_y_id, int swf_z_id);
  //int  getStateIndex(int icp_id, int swf_id);

  Param *param;

  //int icp_num;
  //int swf_num;
  //std::vector<vec2_t>  icp;
  //std::vector<vec3_t>  swf;
  //std::vector<State>   state;

  Grid1D x;
  Grid1D y;
  Grid1D z;
  Grid1D t;

  Grid2D xy;
  Grid3D xyz;
  Grid3D xyt;
};

} // namespace Capt

#endif // __GRID_H__