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
	real_t  v_max, w_max;
	real_t  z_max, z_slope;
	real_t  dsp_duration;

	vec2_t  p_swg, p_land;
	vec2_t  v_swg;
	real_t  r_swg, r_land;
	real_t  w_swg;
	real_t  duration;

	vec2_t  move;
	real_t  turn;
	real_t  travel;
	bool    ready;

public:
	void Read(Scenebuilder::XMLNode* node);
	void Init();

	// set swing foot position, landing position and step duration
	void SetSwingPose    (const vec2_t& _p_swg , real_t _r_swg );
	void SetSwingVelocity(const vec2_t& _v_swg , real_t _w_swg );
	void SetLandingPose  (const vec2_t& _p_land, real_t _r_land);
	void SetDuration     (real_t _tau);

	// get minimum step duration
	real_t GetMinDuration();

	// get reachable radius
	void   GetReachableRadius(real_t tau, real_t& dp, real_t& dr);

	// get swing foot position
	// t is elapsed time after set() is called
	void   GetTraj(real_t t, vec2_t& p, real_t& r, vec2_t& v, real_t& w);
	void   GetVerticalVelocity(const vec2_t& p, const vec2_t& v, real_t pz, real_t& vz);

	 Swing();
	~Swing();
};

} // namespace Capt
