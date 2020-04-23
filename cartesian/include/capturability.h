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
#include <utility>

namespace Capt {

struct CaptureState{
  int      swf_id;
  int      icp_id;
  Index2D  swf_to_icp_idx2;
  int      nstep;

  CaptureState(){}
  CaptureState(int _swf_id, int _icp_id, int _nstep){
    swf_id = _swf_id;
    icp_id = _icp_id;
    nstep  = _nstep;
  }
};

struct CaptureBasin : public std::vector< CaptureState >{
  std::vector< std::pair<int, int> > swf_index;
};

struct CaptureTransition{
  uint8_t icp_x_id_min, icp_x_id_max;
  uint8_t icp_y_id_min, icp_y_id_max;

  CaptureTransition(){}
  CaptureTransition(int _icp_x_id_min, int _icp_x_id_max, int _icp_y_id_min, int _icp_y_id_max){
    icp_x_id_min = _icp_x_id_min;
    icp_x_id_max = _icp_x_id_max;
    icp_y_id_min = _icp_y_id_min;
    icp_y_id_max = _icp_y_id_max;
  }
};

struct CaptureTransitions : public std::vector< CaptureTransition >{
  vector<int>  swf_index;
};

class Capturability {
public:
  Capturability(Model* model, Param* param);
  ~Capturability();

  Model*    model;
  Param*    param;
  Grid*     grid;
  Swing*    swing;

  std::vector< CaptureBasin >               cap_basin;
  std::vector< CaptureTransitions >         cap_trans;
  std::vector< int >                        duration_map;      //< [swf_id0, swf_id1]   (stepping) -> t  (step duration)
  std::vector< std::pair<vec2_t, vec2_t> >  icp_map;           //< [x,y,t] (relative icp, step duration) -> allowable icp range
  std::vector< std::pair<vec2_t, vec2_t> >  mu_map;
  std::vector< int >                        swf_id_valid;      //< array of valid swf_id in [x,y,z]
  std::vector< int >                        valid_swf_id_map;  //< [x,y,z] -> index to swf_id_valid or -1

  float  step_weight;
  float  swf_weight;
  float  icp_weight;

  bool isSteppable(vec2_t swf);
  bool isInsideSupport(vec2_t cop);
  //bool isFeasibleTransition(int swf_id, vec2_t icp, int next_swf_id, int next_icp_id);
  void calcFeasibleIcpRange(int swf, const CaptureState& csnext, std::pair<vec2_t, vec2_t>& icp_range);
  //std::pair<vec2_t, vec2_t>  calcFeasibleNextIcpRange(vec3_t swf, vec2_t icp, int next_swf_id);
  Input calcInput(const State& st, const State& stnext);

  void calcDurationMap();
  void calcIcpMap();
  void calcMuMap();
  void analyze();
  void saveBasin      (std::string filename, int n, bool binary);
  void saveDurationMap(std::string filename, bool binary);
  void saveIcpMap     (std::string filename, bool binary);
  void saveMuMap      (std::string filename, bool binary);
  void saveTrans      (std::string filename, int n, bool binary);
  void saveTransIndex (std::string filename, int n, bool binary);
  void loadBasin      (std::string filename, int n, bool binary);
  void loadDurationMap(std::string filename, bool binary);
  void loadIcpMap     (std::string filename, bool binary);
  void loadMuMap      (std::string filename, bool binary);
  void loadTrans      (std::string filename, int n, bool binary);
  void loadTransIndex (std::string filename, int n, bool binary);

  void getCaptureBasin (State st, int nstep, CaptureBasin& basin);
  //void getCaptureRegion(int state_id, int nstep, std::vector<CaptureRegion>& region);

  // checks is given state is capturable
  // if nstep is -1, then all N is checked and capturable N is stored in nstep
  // otherwise N specified by nstep is checked
  bool isCapturable(int swf_id, int icp_id, int& nstep);

  //bool isCapturableTransition(int swf_id, int icp_x_id, int icp_y_id, int next_id, int& nstep);

  bool findNearest(const State& st, const State& stnext, CaptureState& cs);
  //void findNearest(const State& st, const State& stnext, int& next_swf_id, int& next_icp_id);

private:
  Grid1D swf_x;  //< steppable region
  Grid1D swf_y;
  Grid1D exc_x;  //< unsteppable region
  Grid1D exc_y;
  Grid1D cop_x;  //< cop support region
  Grid1D cop_y;
  Grid1D icp_x;  //< cop support region
  Grid1D icp_y;

  float g, h, T;
};

} // namespace Capt

#endif // __CAPTURABILITY_H__