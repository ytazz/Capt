#ifndef __CAPTURABILITY_H__
#define __CAPTURABILITY_H__

#include "model.h"
#include "param.h"
#include "grid.h"
#include "swing.h"
#include "input.h"
#include "state.h"
#include <iostream>
#include <stdio.h>
#include <vector>

namespace Capt {

struct CaptureRange{
  int    swf_id;
  vec2_t icp_min;
  vec2_t icp_max;

  bool includes(const CaptureRange& r);

  CaptureRange(){
    swf_id = -1;
  }
  CaptureRange(int _swf_id, vec2_t _icp_min, vec2_t _icp_max){
    swf_id  = _swf_id;
    icp_min = _icp_min;
    icp_max = _icp_max;
  }
};

//struct CaptureBasin : public std::vector< CaptureRange >{
//  bool includes(const CaptureRange& r);
//  void add(const CaptureRange& r);
//};
struct CaptureState{
  int state_id;
  int nstep;

  CaptureState(){}
  CaptureState(int _state_id, int _nstep){
    state_id = _state_id;
    nstep    = _nstep;
  }
};
struct CaptureBasin : public std::vector< CaptureState >{

};

struct CaptureTransition{
  uint16_t swf_id;
  uint8_t  icp_x_id_min, icp_x_id_max;
  uint8_t  icp_y_id_min, icp_y_id_max;
  uint32_t next_id;

  CaptureTransition(){}
  CaptureTransition(int _swf_id, int _icp_x_id_min, int _icp_x_id_max, int _icp_y_id_min, int _icp_y_id_max, int _next_id){
    swf_id       = (uint16_t)_swf_id;
    icp_x_id_min = (uint8_t )_icp_x_id_min;
    icp_x_id_max = (uint8_t )_icp_x_id_max;
    icp_y_id_min = (uint8_t )_icp_y_id_min;
    icp_y_id_max = (uint8_t )_icp_y_id_max;
    next_id      = (uint32_t)_next_id;
  }
};

struct CaptureTransitions : public std::vector< CaptureTransition >{

};

class Capturability {
public:
  Capturability(Model* model, Param* param, Grid* grid, Swing* swing);
  ~Capturability();

  Model*    model;
  Param*    param;
  Grid*     grid;
  Swing*    swing;

  std::vector< CaptureBasin > cap_basin;
  std::vector< CaptureTransitions > cap_trans;
  std::vector< int > swf_id_valid;

  bool isSteppable(vec2_t swf);
  bool isInsideSupport(vec2_t cop);
  Input calcInput(State& st, State& stnext);

  void analyze();
  void saveBasin(std::string file_name, int n, bool binary);
  void saveTrans(std::string file_name, int n, bool binary);
  void loadBasin(std::string file_name, int n, bool binary);
  void loadTrans(std::string file_name, int n, bool binary);

  void getCaptureBasin (State st, int nstep, CaptureBasin& basin);
  //void getCaptureRegion(int state_id, int nstep, std::vector<CaptureRegion>& region);

  bool isCapturable(int state_id, int nstep);

private:
  // 踏み出しできない領域
  Grid1D exc_x;
  Grid1D exc_y;
  Grid1D cop_x;
  Grid1D cop_y;

  float g, h, T;
};

} // namespace Capt

#endif // __CAPTURABILITY_H__