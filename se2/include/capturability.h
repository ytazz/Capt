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

class Capturability;

struct CaptureState{
  int      swf_id;
  int      icp_id;
  int      swf_to_icp_id;
  int      nstep;

  CaptureState(){}
  CaptureState(int _swf_id, int _icp_id, int _nstep, Capturability* cap);
};

struct CaptureBasin : public std::vector< CaptureState >{
  std::vector< std::pair<int, int> > swf_index;
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
  std::vector< int >                        duration_map;  //< [swf_id0, swf_id1]   (stepping) -> t  (step duration)
  std::vector< std::pair<vec2_t, vec2_t> >  icp_map;       //< [x,y,t] (relative icp, step duration) -> allowable icp range
  std::vector< int >                        swf_to_xyz;    //< array of valid swf_id in [x,y,z]
  std::vector< int >                        xyz_to_swf;    //< [x,y,z] -> index to swf_id_valid or -1

  float  step_weight;
  float  swf_weight;
  float  icp_weight;

  bool isSteppable(vec2_t swf);
  bool isInsideSupport(vec2_t cop, float margin = 1.0e-5f);
  void calcFeasibleIcpRange(int swf, const CaptureState& csnext, std::pair<vec2_t, vec2_t>& icp_range);
  Input calcInput(const State& st, const State& stnext);

  void calcDurationMap();
  void calcIcpMap();
  void analyze();
  void save(const std::string& basename);
  void load(const std::string& basename);

  void getCaptureBasin (State st, int nstep, CaptureBasin& basin);

  // checks is given state is capturable
  // if nstep is -1, then all N is checked and capturable N is stored in nstep
  // otherwise N specified by nstep is checked
  bool isCapturable(int swf_id, int icp_id, int& nstep);

  bool findNearest(const State& st, const State& stnext, CaptureState& cs);

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