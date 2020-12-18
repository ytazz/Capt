#include "plot.h"
#include <iostream>
#include <sbtimer.h>

using namespace std;
using namespace Capt;

struct Phase{
	enum{
		Init,
		Wait,
		Dsp,
		Ssp,
		Stop,
		Fail,
	};
};

real_t        t;
real_t        dt;
real_t        elapsed;
int           phase;

// capturability
Swing         *swing;
Capturability *cap;
real_t        omega;
real_t        h;
real_t        v_max;
real_t        z_max;

vec3_t        comPos;
vec3_t        comVel;
vec3_t        comAcc;
vec3_t        cop;
vec3_t        icp;
vec3_t        force;

int           s_sup;
vec3_t        supPos;
real_t        supOri;
vec3_t        swgPos;
real_t        swgOri;
vec3_t        landPos;
real_t        landOri;
real_t        duration;
Footstep      footstep;

Scenebuilder::Timer  timer;
int cnt;

FILE *fpTime;

void Init(){
	dt = 0.001;

    swing = new Capt::Swing();
    cap   = new Capt::Capturability();

    cap->step_weight = 0.0f;
    cap->swg_weight  = 1.0f;
    cap->icp_weight  = 10.0f;

    cap->Load("data/");
    printf("load done\n");
    
    phase   = Phase::Init;
    t       = 0.0;
    elapsed = 0.0;

    comPos = vec3_t(0.0f,  0.0f, h   );
    comVel = vec3_t(0.0f,  0.0f, 0.0f);
    comAcc = vec3_t(0.0f,  0.0f, 0.0f);
    cop    = vec3_t(0.0f,  0.0f, 0.0f);
    icp    = vec3_t(0.0f,  0.0f, 0.0f);
    force  = vec3_t(0.0f,  0.0f, 0.0f);

	supPos = vec3_t(0.0f, +0.2f, 0.0f);
	supOri = 0.0;
    swgPos = vec3_t(0.0f, -0.2f, 0.0f);
	swgOri = 0.0;

}

void Control(){
    switch(phase){
	case Phase::Init:
		if(t > 1.0) {
			phase = Phase::Wait;
		}
		break;
	case Phase::Wait:
		if(t > 2.0) {
			real_t initIcpX = footstep[1].icp.x();
			real_t initIcpY = footstep[1].icp.y();
			cop    = vec3_t(initIcpX, initIcpY, 0.0f);
			icp    = vec3_t(initIcpX, initIcpY, 0.0f);
			comPos = vec3_t(initIcpX, initIcpY, 0.0f);
			phase  = Phase::Dsp;
			s_sup  = Capt::Foot::Left;
		}
		break;
	case Phase::Dsp:
		// support foot exchange
		if(s_sup == Capt::Foot::Right) {
			s_sup = Capt::Foot::Left;
			printf("------ DSP (RL) ------\n");
		}
		else{
			s_sup = Capt::Foot::Right;
			printf("------ DSP (LR) ------\n");
		}
		swap(supPos, swgPos);
		swap(supOri, swgOri);

		// determine footstep nearest to current support foot
		//state.updateFootstepIndex();
		//printf("nearest footstep: %d\n", state.footstep.cur);
		footstep.cur++;

		if(footstep.cur == (int)footstep.size() - 1){
			printf("end of footstep reached\n");
			phase = Phase::Stop;
			break;
		}

		// determine next landing position
		landPos = (footstep[footstep.cur + 1].pos - footstep[footstep.cur].pos) + supPos;
		//input.icp  = (state.footstep[state.footstep.cur + 1].icp - state.footstep[state.footstep.cur].pos) + state.su@;
		
		// update swing trajectory and detemine step duration
		swing->Set(swgPos, swgOri, landPos, landOri);
		duration = swing->GetDuration();
		printf("swg : %f %f %f\n", swgPos.x(), swgPos.y(), swgPos.z());
		printf("land: %f %f\n"   , landPos.x(), landPos.y());
		//printf("icp : %f %f\n"   , input.icp.x(), input.icp.y());
		printf("duration: %f\n"  , duration);

		phase   = Phase::Ssp;
		elapsed = 0.0f;
		cnt     = 0;

		break;
	case Phase::Ssp:
		cnt++;
		// do not check too frequently, do not check right before landing
		if(cnt % 10 == 0 && duration - elapsed > z_max/v_max) {
			// support foot
			printf("------ SSP (%c) ------\n", s_sup == Capt::Foot::Right ? 'R' : 'L');
			//printf("elapsed: %f\n", elapsed);

			//printf("s: %d suf: %f,%f swg: %f,%f,%f icp:%f,%f\n",
			//  state.s_suf,
			//  state.sup.x(), state.sup.y(),
			//  state.swg.x(), state.swg.y(), state.swg.z(),
			//  state.icp.x(), state.icp.y());

			timer.CountUS();
			bool modified;
			State st;
			Input in;
			bool ret = cap->Check(st, in, modified);
			if(ret && !modified){
				printf("monitor: success\n");
			}
			if(ret && modified){
				printf("monitor: modified\n");
				vec3_t pswg (st.swg[0], st.swg[1], st.swg[2]);
				real_t rswg  = st.swg[3];
				vec3_t pland(in.land[0], in.land[1], in.land[2]);
				real_t rland = in.land[3];

				swing->Set(pswg, rswg, pland, rland);
				duration = swing->GetDuration();

				printf("land: %f,%f  duration: %f\n", in.land.x(), in.land.y(), duration);
				elapsed = 0.0f;
			}
			if(!ret){
				printf("monitor: fail\n");
			}
			fprintf(fpTime, "%lf, %lf\n", t, timer.CountUS());
		}

		// update swing foot position
		swing->GetTraj(elapsed, swgPos, swgOri);
		//printf("swg: %f,%f,%f\n", state.swg.x(), state.swg.y(), state.swg.z());

		// switch to DSP if step duration has elapsed
		if(elapsed > duration)
			phase = Phase::Dsp;

		break;
	case Phase::Stop:
		break;
	case Phase::Fail:
		break;
    default:
		break;
    }

	if(4.5 <= t && t <= 4.51) {
		// simulation 1
		//force.x() = -5000;
		// simulation 2
		force.y() = 5000;
	}else{
		force.x() = 0;
		force.y() = 0;
	}
	/*if(4.5f <= t && t <= 5.0f) {
	// simulation 3
	force.x() = -200.0f * sin( ( t - 5.5f ) * 3.14159f / 0.5f);
	force.y() = +150.0f * sin( ( t - 5.5f ) * 3.14159f / 0.5f);
	//printf("force %f,%f\n", force.x(), force.y());
	}else{
	force.x() = 0.0f;
	force.y() = 0.0f;
	}*/
	
    // simulation step
	if(phase == Phase::Ssp || phase == Phase::Dsp)
    	Step();
	
    t       += dt;
    elapsed += dt;

}

void Step(){
	comPos       += comVel * dt;
	comPos.z()    = h;
	comVel       += comAcc * dt;
	comVel.z()    = 0.0;
	comAcc        = (omega*omega)*(comPos - cop);
	comAcc.z()    = 0.0;
	icp           = comPos + comVel/omega;
	icp.z()       = 0.0;
}

int main(int argc, char const *argv[]) {
	Init();

	while(true){
		Control();
	}

	return 0;
}
