#pragma once

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

#include <sbxml.h>
using namespace Scenebuilder;

namespace Capt {

class Capturability;

struct CaptureState{
  int      swg_id;
  int      icp_id;
  int      swg_to_icp_id;
  int      nstep;

  CaptureState(){}
  CaptureState(int _swg_id, int _icp_id, int _nstep, Capturability* cap);
};

struct CaptureBasin : public std::vector< CaptureState >{
  std::vector< std::pair<int, int> > swg_index;
};

class Capturability {
public:
	Grid1D swg_x;  //< steppable region
	Grid1D swg_y;
	Grid1D swg_r;
	Grid1D exc_x;  //< unsteppable region
	Grid1D exc_y;
	Grid1D cop_x;  //< cop support region
	Grid1D cop_y;
	Grid1D icp_x;  //< cop support region
	Grid1D icp_y;

	float g, h, T;
  
	Model*    model;
	Param*    param;
	Grid*     grid;
	Swing*    swing;

	std::vector< CaptureBasin >               cap_basin;
	std::vector< int >                        duration_map;  //< [swg_id0, swg_id1]   (stepping) -> t  (step duration)
	std::vector< std::pair<vec2_t, vec2_t> >  icp_map;       //< [x,y,t] (relative icp, step duration) -> allowable icp range
	std::vector< int >                        swg_to_xyzr;   //< array of valid swg_id in [x,y,z,r]
	std::vector< int >                        xyzr_to_swg;   //< [x,y,z,r] -> index to swg_id_valid or -1

	float  step_weight;
	float  swg_weight;
	float  icp_weight;

	bool isSteppable(vec2_t p_swg, real_t r_swg);
	bool isInsideSupport(vec2_t cop, float margin = 1.0e-5f);
	void calcFeasibleIcpRange(int swg, const CaptureState& csnext, std::pair<vec2_t, vec2_t>& icp_range);
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
	bool isCapturable(int swg_id, int icp_id, int& nstep);

	bool findNearest(const State& st, const State& stnext, CaptureState& cs);

	void Read(XMLNode* node);

	Capturability(Model* model, Param* param);
	~Capturability();

};

} // namespace Capt
