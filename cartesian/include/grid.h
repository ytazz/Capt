#ifndef __GRID_H__
#define __GRID_H__

#include "param.h"
#include "state.h"
#include "base.h"
#include <iostream>
#include <string>
#include <vector>

namespace Capt {

struct Grid1D{
  float min;
  float max;
  float stp;
  int   num;

  int  round(float v);
  void indexRange(float fmin, float fmax, int& imin, int& imax);
};

class Grid {
public:
  Grid(Param *param);
  ~Grid();

  //bool existState(int state_id);
  //bool existState(State state_);

  int getNumIcp();
  int getNumSwf();
  int getNumState();

  //int getStateIndex(State state_);

  //vec2_t  roundVec(vec2_t vec, vec2_t res);
  int     roundState(State state_);

  //int indexIcp(vec2_t icp);
  //int indexSwf(vec2_t swf);
  //int indexCop(vec2_t cop);

  int  getIcpIndex  (int icp_x_id, int icp_y_id);
  int  getSwfIndex  (int swf_x_id, int swf_y_id, int swf_z_id);
  int  getStateIndex(int icp_id, int swf_id);

  Param *param;

  int icp_num;
  int swf_num;
  std::vector<vec2_t>  icp;
  std::vector<vec3_t>  swf;
  std::vector<State>   state;

  Grid1D icp_x;
  Grid1D icp_y;
  Grid1D swf_x;
  Grid1D swf_y;
  Grid1D swf_z;

};

} // namespace Capt

#endif // __GRID_H__