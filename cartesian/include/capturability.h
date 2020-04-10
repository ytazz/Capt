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
  //int state_id;
  int swf_id;
  int icp_id;
  int nstep;

  CaptureState(){}
  CaptureState(int _swf_id, int _icp_id, int _nstep){
    //state_id = _state_id;
    swf_id = _swf_id;
    icp_id = _icp_id;
    nstep  = _nstep;
  }
};
struct CaptureBasin : public std::vector< CaptureState >{
  std::vector< std::pair<int, int> > swf_index;
};

struct CaptureTransition{
  uint16_t swf_id;
  uint8_t  icp_x_id_min, icp_x_id_max;
  uint8_t  icp_y_id_min, icp_y_id_max;
  uint16_t next_swf_id;
  uint8_t  next_icp_id;

  CaptureTransition(){}
  CaptureTransition(int _swf_id, int _icp_x_id_min, int _icp_x_id_max, int _icp_y_id_min, int _icp_y_id_max, int _next_swf_id, int _next_icp_id){
    swf_id       = (uint16_t)_swf_id;
    icp_x_id_min = (uint8_t )_icp_x_id_min;
    icp_x_id_max = (uint8_t )_icp_x_id_max;
    icp_y_id_min = (uint8_t )_icp_y_id_min;
    icp_y_id_max = (uint8_t )_icp_y_id_max;
    next_swf_id  = (uint16_t)_next_swf_id;
    next_icp_id  = (uint8_t )_next_icp_id;
  }
};

struct CaptureTransitions : public std::vector< CaptureTransition >{
  std::vector< std::pair<int,int> > swf_index;   //< index range for each swf_id
};

class Capturability {
public:
  Capturability(Model* model, Param* param);
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
  Input calcInput(const State& st, const State& stnext);

  void analyze();
  void saveBasin(std::string file_name, int n, bool binary);
  void saveTrans(std::string file_name, int n, bool binary);
  void loadBasin(std::string file_name, int n, bool binary);
  void loadTrans(std::string file_name, int n, bool binary);

  void getCaptureBasin (State st, int nstep, CaptureBasin& basin);
  //void getCaptureRegion(int state_id, int nstep, std::vector<CaptureRegion>& region);

  // checks is given state is capturable
  // if nstep is -1, then all N is checked and capturable N is stored in nstep
  // otherwise N specified by nstep is checked
  bool isCapturable(int swf_id, int icp_id, int& nstep);

  bool isCapturableTransition(int swf_id, int icp_x_id, int icp_y_id, int next_id, int& nstep);

  void findNearest(int swf_id, int icp_x_id, int icp_y_id, int next_swf_id, int next_icp_id, int& mod_swf_id, int& mod_icp_id);
  //void findNearest(const State& st, const State& stnext, int& next_swf_id, int& next_icp_id);

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