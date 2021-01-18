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
	int      nstep;

	CaptureState(){}
	CaptureState(int _swg_id, int _icp_id, int _nstep, Capturability* cap);
};

struct CaptureBasin : public std::vector< CaptureState >{
};

class Capturability {
public:
	struct Region{
		struct Type{
			enum{
				Rectangle,
				Radial,
			};
		};
		int     type;
		vec2_t  min;
		vec2_t  max;
		real_t  near ;
		real_t  far  ;
		real_t  angle;

		void Read       (XMLNode* node);
		bool IsSteppable(const vec2_t& p_swg, real_t r_swg);
	
	};
	
	Grid1D cop_x;  //< cop support region
	Grid1D cop_y;

	real_t g, h, T;
	real_t cop_margin;
	real_t step_weight;
	real_t swg_weight;
	real_t tau_weight;
	real_t icp_weight;

	vector<Region>  regions;
 
	Grid*     grid;
	Swing*    swing;

	std::vector< CaptureBasin >  cap_basin;
	std::vector< int >           swg_to_xyr;    //< array of valid swg_id in [x,y,r]
	std::vector< int >           xyr_to_swg;    //< [x,y,r] -> index to swg_id_valid or -1

	bool   IsSteppable         (const vec2_t& p_swg, real_t r_swg);
	bool   IsInsideSupport     (const vec2_t& cop, real_t margin);
	void   CalcMu              (const vec3_t& swg, const vec2_t& icp, vec2_t& mu);
	void   CalcTauRange        (const vec2_t& ainv_range, vec2_t& tau_range);
	void   CalcAinvRange       (const vec2_t& tau_range, vec2_t& ainv_range);
	bool   CalcFeasibleAinvRange(const vec2_t& mu, const vec2_t& icp, real_t margin, vec2_t& ainv_range);
	//void   CalcFeasibleIcpRange(const vec2_t& mu, real_t ainv, std::pair<vec2_t, vec2_t>& icp_range);
	void   CalcFeasibleIcpRange(const vec2_t& mu,  const vec2_t& ainv_range, real_t margin, std::pair<vec2_t, vec2_t>& icp_range);
	void   CalcFeasibleMuRange (const vec2_t& icp, const vec2_t& ainv_range, real_t margin, std::pair<vec2_t, vec2_t>& mu_range );
	real_t CalcMinDuration     (const vec3_t& swg0, const vec3_t& swg1);
	void   EnumReachable       (const vector< pair<int, real_t> >& seed, vector<bool>& swg_id_array);
	void   CalcInput           (const State& st, const State& stnext, Input& in);
	bool   Check               (const State& st, Input& in, State& st_mod, int& nstep, bool& duration_modified, bool& step_modified);
	
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
