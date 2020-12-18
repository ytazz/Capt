#include "swing.h"

namespace Capt {

Swing::Swing() {
	p_swg  = vec3_t::Zero();
	p_land = vec3_t::Zero();
	r_swg  = 0.0f;
	r_land = 0.0f;

	tau_ascend  = 0.0f;
	tau_travel  = 0.0f;
	tau_descend = 0.0f;

	v_const = 1.0f;
	w_const = 1.0f;
	z_max   = 0.0f;
}

void Swing::Read(Scenebuilder::XMLNode* node){
	node->Get(v_const, ".v_const");
	node->Get(w_const, ".w_const");
	node->Get(z_max  , ".z_max"  );
}

Swing::~Swing() {
}

void Swing::Set(vec3_t _p_swg, real_t _r_swg, vec3_t _p_land, real_t _r_land){
	p_swg  = _p_swg ;
	r_swg  = _r_swg ;
	p_land = _p_land;
	r_land = _r_land;

	dist_x =  p_land.x() - p_swg.x();
	dist_y =  p_land.y() - p_swg.y();
	dist   = sqrt( dist_x * dist_x + dist_y * dist_y );

	turn = WrapRadian(r_land - r_swg);
  
	tau_ascend  = (z_max - p_swg.z())/v_const;
	tau_travel  = dist/v_const;
	tau_turn    = std::abs(turn)/w_const;
	tau_descend = z_max/v_const;
}

float Swing::GetDuration() {
	return tau_ascend + std::max(tau_travel, tau_turn) + tau_descend;
}

bool Swing::IsDescending(real_t t){
	return t > GetDuration() - tau_descend;
}

void Swing::GetTraj(real_t t, vec3_t& p, real_t& r) {
	// ascending
	if(0 <= t && t < tau_ascend) {
		p.x() = p_swg.x();
		p.y() = p_swg.y();
		p.z() = p_swg.z() + v_const * t;

		r = r_swg;
	}
	// traveling to landing position
	if(tau_ascend <= t && t < tau_ascend + std::max(tau_travel, tau_turn)) {
		p.x() = p_swg.x() + v_const * (dist_x/dist)*(t - tau_ascend);
		p.y() = p_swg.y() + v_const * (dist_y/dist)*(t - tau_ascend);
		p.z() = z_max;

		r = r_swg + w_const * (t - tau_ascend);
	}
	// descending
	if(tau_ascend + tau_travel <= t && t < tau_ascend + tau_travel + tau_descend) {
		p.x() = p_land.x();
		p.y() = p_land.y();
		p.z() = z_max - v_const*(t - (tau_ascend + tau_travel));

		r = r_land;
	}
	// after landing
	if(t >= tau_ascend + tau_travel + tau_descend){
		p.x() = p_land.x();
		p.y() = p_land.y();
		p.z() = p_land.z();

		r = r_land;
	}
}

} // namespace Capt