#include "swing.h"

namespace Capt {

const real_t eps = 1.0e-10;

Swing::Swing() {
	type = Type::Rectangle;

	p_swg  = vec2_t();
	v_swg  = vec2_t();
	p_land = vec2_t();
	r_swg  = 0.0;
	w_swg  = 0.0;
	r_land = 0.0;

	v_max        = 1.0;
	w_max        = 1.0;
	z_max        = 0.0;
	z_slope      = 1.0;
	dsp_duration = 0.0;

	ready = false;
}

void Swing::Read(Scenebuilder::XMLNode* node){
	node->Get(v_max       , ".v_max"       );
	node->Get(w_max       , ".w_max"       );
	node->Get(z_max       , ".z_max"       );
	node->Get(z_slope     , ".z_slope"     );
	node->Get(dsp_duration, ".dsp_duration");

	string str;
	node->Get(str, ".type");
	if(str == "rectangle") type = Type::Rectangle;
	if(str == "spline"   ) type = Type::Spline;

	ready = false;
}

Swing::~Swing() {
}

void Swing::SetSwingPose(const vec2_t& _p_swg, real_t _r_swg){
	p_swg = _p_swg ;
	r_swg = _r_swg ;
	ready = false;
}

void Swing::SetSwingVelocity(const vec2_t& _v_swg, real_t _w_swg){
	v_swg = _v_swg ;
	w_swg = _w_swg ;
	ready = false;
}

void Swing::SetLandingPose(const vec2_t& _p_land, real_t _r_land){
	p_land = _p_land;
	r_land = _r_land;
	ready  = false;
}

void Swing::SetDuration(real_t _tau){
	duration = _tau;
	ready    = false;
}

void Swing::Init(){
	move = p_land - p_swg;
	turn = WrapRadian(r_land - r_swg);
  
	ready = true;
}

real_t Swing::GetMinDuration(){
	if(!ready)
		Init();

	if(type == Type::Rectangle){
		return std::max(move.norm()/v_max, std::abs(turn)/w_max) + dsp_duration;
	}
	if(type == Type::Spline){
		return 1.5*std::max(move.norm()/v_max, std::abs(turn)/w_max) + dsp_duration;
	}

	return 0.0;
}

void Swing::GetReachableRadius(real_t tau, real_t& dp, real_t& dr){
	if(!ready)
		Init();

	if(type == Type::Rectangle){
		dp = std::max(0.0, v_max*(tau - dsp_duration));
		dr = std::max(0.0, w_max*(tau - dsp_duration));
	}
	if(type == Type::Spline){
		dp = std::max(0.0, v_max*(tau - dsp_duration)/1.5);
		dr = std::max(0.0, w_max*(tau - dsp_duration)/1.5);
	}
}

void Swing::GetTraj(real_t t, vec2_t& p, real_t& r, vec2_t& v, real_t& w) {
	if(!ready)
		Init();

	// no swing movement if foothold does not change
	if(p_swg == p_land && r_swg == r_land){
		p = p_swg;
		r = r_swg;
		v = vec2_t();
		w = 0.0;
		return;
	}
	// after landing
	real_t tau_travel = duration - dsp_duration;
	if(t > tau_travel){
		p = p_land;
		v = vec2_t();
		r = r_land;
		w = 0.0;
		return;
	}

	if(type == Type::Rectangle){
		p = p_swg + move*(t/tau_travel);
		v = move/tau_travel;
		r = r_swg + turn*(t/tau_travel);
		w = turn/tau_travel;
	}
	if(type == Type::Spline){
		p = InterpolatePos(t,
			0.0       , p_swg , v_swg,
			tau_travel, p_land, vec2_t(),
			Interpolate::Cubic);
		v = InterpolateVel(t,
			0.0       , p_swg , v_swg,
			tau_travel, p_land, vec2_t(),
			Interpolate::Cubic);

		r = InterpolatePos(t,
			0.0       , r_swg , w_swg,
			tau_travel, r_land, 0.0,
			Interpolate::Cubic);
		w = InterpolateVel(t,
			0.0       , r_swg , w_swg,
			tau_travel, r_land, 0.0,
			Interpolate::Cubic);

	}
}

void Swing::GetVerticalVelocity(const vec2_t& p, const vec2_t& v, real_t pz, real_t& vz){
	vec2_t d = p_land - p;
	
	// horizontal approaching speed to landing position
	real_t vd;
	real_t dnorm = d.norm();
	if(dnorm < eps){
		// right above landing position
		vz = 0.0;
		return;
	}
	vd = (d*v)/dnorm;

	// decend
	if(pz >= z_slope*dnorm){
		vz = -(pz/dnorm)*vd;
		return;
	}
	// keep altitude
	if(pz >= z_max){
		vz = 0.0;
		return;
	}
	// ascend
	vz = z_slope*vd;

}

} // namespace Capt