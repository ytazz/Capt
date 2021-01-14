#pragma once

#include "base.h"

namespace Capt {

class Swing {
public:
	struct Type{
		enum{
			Rectangle,
			Spline,
		};
	};

	int     type;
	real_t  v_max, w_max, z_max;
	real_t  slope;
	real_t  dsp_duration;

	vec3_t  p_swg, p_land;
	real_t  r_swg, r_land;
	real_t  duration;

	vec2_t  move;
	real_t  turn;
	real_t  travel;
	real_t  v_const;
	real_t  w_const;
	real_t  tau_ascend, tau_travel, tau_descend;
	bool    ready;

public:
	void Read(Scenebuilder::XMLNode* node);
	void Init();

	// set swing foot position, landing position and step duration
	void SetSwg     (vec3_t _p_swg , real_t _r_swg );
	void SetLand    (vec3_t _p_land, real_t _r_land);
	void SetDuration(real_t _tau);

	// get minimum step duration
	real_t GetMinDuration();

	// get reachable radius
	void GetReachableRadius(real_t tau, real_t& dp, real_t& dr);

	// get swing foot position
	// t is elapsed time after set() is called
	void  GetTraj(real_t t, vec3_t& p, real_t& r, vec3_t& v, real_t& w);

	bool IsDescending(real_t t);

 	 Swing();
	~Swing();
};

} // namespace Capt
