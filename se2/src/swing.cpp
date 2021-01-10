#include "swing.h"

namespace Capt {

const real_t eps = 1.0e-10;

Swing::Swing() {
	type = Type::Rectangle;

	p_swg  = vec3_t::Zero();
	p_land = vec3_t::Zero();
	r_swg  = 0.0;
	r_land = 0.0;

	tau_ascend  = 0.0;
	tau_travel  = 0.0;
	tau_descend = 0.0;

	v_max        = 1.0;
	w_max        = 1.0;
	z_max        = 0.0;
	slope        = 1.0;
	dsp_duration = 0.0;

	ready = false;
}

void Swing::Read(Scenebuilder::XMLNode* node){
	node->Get(v_max       , ".v_max"       );
	node->Get(w_max       , ".w_max"       );
	node->Get(z_max       , ".z_max"       );
	node->Get(slope       , ".slope"       );
	node->Get(dsp_duration, ".dsp_duration");

	string str;
	node->Get(str, ".type");
	if(str == "rectangle") type = Type::Rectangle;
	if(str == "spline"   ) type = Type::Spline;

	ready = false;
}

Swing::~Swing() {
}

void Swing::SetSwg(vec3_t _p_swg, real_t _r_swg){
	p_swg = _p_swg ;
	r_swg = _r_swg ;
	ready = false;
}

void Swing::SetLand(vec3_t _p_land, real_t _r_land){
	p_land = _p_land;
	r_land = _r_land;
	ready  = false;
}

void Swing::SetDuration(real_t _tau){
	duration = _tau;
	ready    = false;
}

void Swing::Init(){
	move.x = p_land.x - p_swg.x;
	move.y = p_land.y - p_swg.y;
	turn   = WrapRadian(r_land - r_swg);
  
	if(type == Type::Rectangle){
		travel = (z_max - p_swg.z) + move.norm() + z_max;
		v_const= travel        /(duration - dsp_duration);
		w_const= std::abs(turn)/(duration - dsp_duration);

		tau_ascend  = (z_max - p_swg.z)/v_const;
		tau_travel  = move.norm()      /v_const;
		tau_descend = z_max            /v_const;
		//duration    = tau_ascend + tau_travel + tau_descend + dsp_duration;
	}

	ready = true;
}

real_t Swing::GetMinDuration(){
	if(!ready)
		Init();

	if(type == Type::Rectangle){
		return travel/v_max + dsp_duration;
	}
	if(type == Type::Spline){
		return 1.5*move.norm()/v_max + dsp_duration;
	}

	return 0.0;
}

bool Swing::IsDescending(real_t t){
	if(!ready)
		Init();

	if(type == Type::Rectangle){
		return t > duration - (tau_descend + dsp_duration);
	}
	if(type == Type::Spline){
		return t > duration - dsp_duration;
	}

	
	return false;
}

void Swing::GetTraj(real_t t, vec3_t& p, real_t& r, vec3_t& v, real_t& w) {
	if(!ready)
		Init();

	// no swing movement if foothold does not change
	if(p_swg == p_land && r_swg == r_land){
		p = p_swg;
		r = r_swg;
		v = vec3_t();
		w = 0.0;
		return;
	}

	if(type == Type::Rectangle){
		// ascending
		if(0 <= t && t < tau_ascend) {
			p = vec3_t(p_swg.x, p_swg.y, p_swg.z + v_const * t);
			v = vec3_t(0.0, 0.0, v_const);
			r = r_swg;
			w = 0.0;
		}
		// traveling to landing position
		if(tau_ascend <= t && t < tau_ascend + tau_travel) {
			p = vec3_t(
				p_swg.x + move.x*(t - tau_ascend)/tau_travel,
				p_swg.y + move.y*(t - tau_ascend)/tau_travel,
				z_max);
			v = vec3_t(move.x/tau_travel, move.y/tau_travel, 0.0);
			r = r_swg + turn * (t - tau_ascend)/tau_travel;
			w = turn/tau_travel;
		}
		// descending
		if(tau_ascend + tau_travel <= t && t < tau_ascend + tau_travel + tau_descend) {
			p = vec3_t(p_land.x, p_land.y, z_max - v_const*(t - (tau_ascend + tau_travel)));
			v = vec3_t(0.0, 0.0, -v_const);
			r = r_land;
			w = 0.0;
		}
		// after landing
		if(t >= tau_ascend + tau_travel + tau_descend){
			p = p_land;
			v = vec3_t();
			r = r_land;
			w = 0.0;
		}
	}
	if(type == Type::Spline){
		vec2_t p2, v2;
		vec2_t p2_swg (p_swg .x, p_swg .y);
		vec2_t p2_land(p_land.x, p_land.y);

		p2 = InterpolatePos(t,
			0.0     , p2_swg , vec2_t(),
			duration, p2_land, vec2_t(),
			Interpolate::Cubic);
		v2 = InterpolateVel(t,
			0.0     , p2_swg , vec2_t(),
			duration, p2_land, vec2_t(),
			Interpolate::Cubic);

		real_t d0 = (p2 - p2_swg ).norm();
		real_t d1 = (p2 - p2_land).norm();
		real_t z0 = p_swg.z + slope*d0;
		real_t z1 = slope*d1;
		real_t pz = std::min(std::min(z0, z1), z_max);
		real_t vz;
		vec2_t d     = p2_land - p2;
		real_t dnorm = d.norm();
		real_t vd;
		
		if(dnorm < eps)
			 vd = 0.0;
		else vd = -(d*v2)/d.norm();

		if(pz == z0)
			vz = -slope*vd;
		if(pz == z1)
			vz =  slope*vd;
		if(pz == z_max)
			vz =  0.0;

		p = vec3_t(p2.x, p2.y, pz);
		v = vec3_t(v2.x, v2.y, vz);

		r = InterpolatePos(t,
			0.0     , r_swg , 0.0,
			duration, r_land, 0.0,
			Interpolate::Cubic);
		w = InterpolateVel(t,
			0.0     , r_swg , 0.0,
			duration, r_land, 0.0,
			Interpolate::Cubic);

	}
}

} // namespace Capt