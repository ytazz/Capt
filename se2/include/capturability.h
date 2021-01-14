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
	//std::vector< std::pair<int, int> > swg_index;
	//std::vector< std::pair<int, int> > mu_index;
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
	//std::vector< std::vector<int> >           swg_map;       //< ainv_id, swg_id -> array of swg_id
	//std::vector< std::pair<vec2_t, vec2_t> >  icp_map;       //< ainv_id, mu_id  -> icp range
	std::vector< int >                        swg_to_xyr;    //< array of valid swg_id in [x,y,r]
	std::vector< int >                        xyr_to_swg;    //< [x,y,r] -> index to swg_id_valid or -1

	real_t step_weight;
	real_t swg_weight;
	real_t tau_weight;
	real_t icp_weight;

	bool   IsSteppable         (const vec2_t& p_swg, real_t r_swg);
	bool   IsInsideSupport     (const vec2_t& cop, real_t margin = 1.0e-5);
	void   CalcMu              (const vec3_t& swg, const vec2_t& icp, vec2_t& mu);
	void   CalcTauRange        (const vec2_t& ainv_range, vec2_t& tau_range);
	void   CalcAinvRange       (const vec2_t& tau_range, vec2_t& ainv_range);
	bool   CalcFeasibleAinvRange(const vec2_t& mu, const vec2_t& icp, vec2_t& ainv_range);
	void   CalcFeasibleIcpRange(const vec2_t& mu, real_t ainv, std::pair<vec2_t, vec2_t>& icp_range);
	void   CalcFeasibleIcpRange(const vec2_t& mu, const vec2_t& ainv_range, std::pair<vec2_t, vec2_t>& icp_range);
	real_t CalcMinDuration     (const vec3_t& swg0, const vec3_t& swg1);
	//void   EnumReachable       (const vec3_t& swg1, real_t tau_min, vector<bool>& swg_id_array);
	void   EnumReachable       (const vector< pair<int, real_t> >& seed, vector<bool>& swg_id_array);
	void   CalcInput           (const State& st, const State& stnext, Input& in);
	bool   Check               (const State& st, Input& in, State& st_mod, int& nstep, bool& modified);
	//State  CalcNextState       (const State& st, const Input& in    );
	
	//void  CalcDurationMap();
	//void  CalcSwgMap();
	//void  CalcIcpMap();
	//void  CreateMuIndex(CaptureBasin& basin);
	void  Analyze();
	void  Save(const std::string& basename);
	void  Load(const std::string& basename);

	void  GetCaptureBasin (const State& st, int nstepMin, int nstepMax, CaptureBasin& basin, vector<vec2_t>& tau_range_valid);

	// checks is given state is capturable
	// if nstep is -1, then all N is checked and capturable N is stored in nstep
	// otherwise N specified by nstep is checked
	//bool  IsCapturable(int swg_id, int icp_id, int& nstep);

	bool  FindNearest(const State& st, const Input& in_ref, const State& stnext_ref, CaptureState& cs_opt, real_t& tau_opt, int& n_opt);

	void  Read(Scenebuilder::XMLNode* node);

	 Capturability();
	~Capturability();

};

} // namespace Capt
