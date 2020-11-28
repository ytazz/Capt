#pragma once

#include "param.h"
#include "base.h"

namespace Capt {

class Swing {
public:
	vec3_t  p_swg, p_land;
	real_t  r_swg, r_land;
	real_t  dist, dist_x, dist_y;
	real_t  turn;
	real_t  tau_ascend, tau_travel, tau_turn, tau_descend;
	real_t  v_const, w_const, z_max;

public:
	// set swing foot position and landing position
	// set() should not be called after swing foot starts descending
	void Set(vec3_t _p_swg, real_t _r_swg, vec3_t _p_land, real_t _r_land);

	// get step duration
	real_t GetDuration();

	// get swing foot position
	// t is elapsed time after set() is called
	void  GetTraj(real_t t, vec3_t& p, real_t& r);

	// swing foot is swinging down or not
	//bool isSwingDown(float dt);

	void Read(Scenebuilder::XMLNode* node);

 	 Swing();
	~Swing();
};

} // namespace Capt
