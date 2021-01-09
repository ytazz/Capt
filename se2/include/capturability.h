#pragma once

#include "base.h"
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
	int      swg_id;
	int      icp_id;
	//int      mu_id;
	//int      swg_to_icp_id;
	int      nstep;

	CaptureState(){}
	CaptureState(int _swg_id, int _icp_id, int _nstep, Capturability* cap);
};

struct CaptureBasin : public std::vector< CaptureState >{
	std::vector< std::pair<int, int> > swg_index;
};

class Capturability {
public:
	real_t swg_near ;
	real_t swg_far  ;
	real_t swg_angle;
	
	Grid1D cop_x;  //< cop support region
	Grid1D cop_y;
	Grid1D icp_x;  //< cop support region
	Grid1D icp_y;

	real_t g, h, T;
 
	Grid*     grid;
	Swing*    swing;

	std::vector< CaptureBasin >               cap_basin;
	//std::vector< int >                        duration_map;  //< [swg_id0, swg_id1]   (stepping) -> t  (step duration)
	//std::vector< std::pair<vec2_t, vec2_t> >  icp_map;       //< [x,y,t] (relative icp, step duration) -> allowable icp range
	std::vector< int >                        swg_to_xyzr;   //< array of valid swg_id in [x,y,z,r]
	std::vector< int >                        xyzr_to_swg;   //< [x,y,z,r] -> index to swg_id_valid or -1

	real_t step_weight;
	real_t swg_weight;
	real_t tau_weight;
	real_t icp_weight;

	bool   IsSteppable         (const vec2_t& p_swg, real_t r_swg);
	bool   IsInsideSupport     (const vec2_t& cop, real_t margin = 1.0e-5);
	//void  CalcFeasibleIcpRange(int swg, const CaptureState& csnext, std::pair<vec2_t, vec2_t>& icp_range);
	bool   CalcFeasibleDurationRange(const State& stnext, const vec2_t& icp, vec2_t& tau_range);
	void   CalcFeasibleIcpRange(const State& stnext, const vec2_t& tau_range, std::pair<vec2_t, vec2_t>& icp_range);
	real_t CalcMinDuration     (const vec4_t& swg0, const vec4_t& swg1);
	//State  CalcNextState       (const State& st, const Input& in    );
	Input  CalcInput           (const State& st, const State& stnext, real_t tau);
	bool   Check               (const State& st, Input& in, State& st_mod, int& nstep, bool& modified);

	//void  CalcDurationMap();
	//void  CalcIcpMap();
	void  Analyze();
	void  Save(const std::string& basename);
	void  Load(const std::string& basename);

	void  GetCaptureBasin (const State& st, int nstepMin, int nstepMax, CaptureBasin& basin);

	// checks is given state is capturable
	// if nstep is -1, then all N is checked and capturable N is stored in nstep
	// otherwise N specified by nstep is checked
	bool  IsCapturable(int swg_id, int icp_id, int& nstep);

	bool  FindNearest(const State& st, const Input& in_ref, const State& stnext_ref, CaptureState& cs_opt, real_t& tau_opt, int& n_opt);

	void  Read(Scenebuilder::XMLNode* node);

	 Capturability();
	~Capturability();

};

} // namespace Capt
