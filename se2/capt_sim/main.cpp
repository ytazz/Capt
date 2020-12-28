#include "capturability.h"
#include "swing.h"
#include "footstep.h"

#include <iostream>
#include <sbtimer.h>

using namespace std;
using namespace Capt;

struct Phase{
	enum{
		Dsp,
		Ssp,
		Stop,
		Fail,
	};
};

real_t        dt;
real_t        t;
int           phase;

Swing           swing;
Capturability   cap;
Footstep        footstep;
			    
vec3_t          comPos;
vec3_t          comVel;
vec3_t          comAcc;
vec3_t          force;
bool            modified;
bool            succeeded;
			
Footstep::Step  steps[2];

FILE*           file;

Scenebuilder::Timer  timer;
int cnt;

void Init(){
	Scenebuilder::XML xmlCapt;
	xmlCapt.Load("conf/capt.xml");

	Scenebuilder::XML xmlSim;
	xmlSim.Load("conf/sim.xml");

	dt = 0.001;

	cap     .Read(xmlCapt.GetRootNode());
	swing   .Read(xmlCapt.GetRootNode()->GetNode("swing"   ));
	footstep.Read(xmlSim .GetRootNode()->GetNode("footstep"));

	// load capturability database
	cap.Load("data/");
    printf("load done\n");

	// generate footstepa
	footstep.Calc(&cap, &swing);
	footstep.cur = 0;
    steps[0] = footstep.steps[0];
    
    t = 0.0;
    
    comPos = vec3_t(0.0, 0.0, cap.h);
	comVel = (steps[0].icp - comPos)/cap.T;
    comAcc = vec3_t(0.0, 0.0, 0.0  );
    force  = vec3_t(0.0, 0.0, 0.0  );

	file = fopen("data/log.csv", "w");
	fprintf(file, 
		"count, time, "
		"com_pos_x, com_pos_y, com_pos_z, "
		"cop_x, cop_y, cop_z, "
		"icp_x, icp_y, icp_z, "
		"foot0_x, foot0_y, foot0_z, "
		"foot1_x, foot1_y, foot1_z, "
		"succeeded, modified\n"
		);

	phase  = Phase::Dsp;
}

void Control(){
	printf("control: t %f\n", t);

	if(phase == Phase::Dsp){
		if(footstep.cur == (int)footstep.steps.size() - 1){
			printf("end of footstep reached\n");
			phase = Phase::Stop;
		}
		else{
			// determine next landing position
			Footstep::Step& st0 = footstep.steps[footstep.cur+0];
			Footstep::Step& st1 = footstep.steps[footstep.cur+1];
			int sup =  steps[0].side;
			int swg = !steps[0].side;

			steps[1].side = !steps[0].side;

			// support foot does not move
			steps[1].footPos[sup] = steps[0].footPos[sup];
			steps[1].footOri[sup] = steps[0].footOri[sup];
			
			mat3_t R = mat3_t::Rot(steps[1].footOri[sup] - st1.footOri[sup], 'z');
			steps[1].footPos[swg] = R*(st1.footPos[swg] - st1.footPos[sup]) + steps[1].footPos[sup];
			steps[1].footOri[swg] =   (st1.footOri[swg] - st1.footOri[sup]) + steps[1].footOri[sup];
			steps[1].icp          = R*(st1.icp          - st1.footPos[sup]) + steps[1].footPos[sup];
			steps[1].cop          = R*(st1.cop          - st1.footPos[sup]) + steps[1].footPos[sup];
			//input.icp  = (state.footstep[state.footstep.cur + 1].icp - state.footstep[state.footstep.cur].pos) + state.su@;
		
			// update swing trajectory and detemine step duration
			swing.Set(
				steps[0].footPos[swg], steps[0].footOri[swg],
				steps[1].footPos[swg], steps[1].footOri[swg]);

			steps[0].duration = swing.GetDuration();
			steps[0].telapsed = 0.0;
			
			steps[0].Print();
			steps[1].Print();
			
			phase = Phase::Ssp;
			cnt   = 0;
		}
	}
	if(phase == Phase::Ssp){
		cnt++;

		int sup =  steps[0].side;
		int swg = !steps[0].side;
		
		// do not check too frequently, do not check right before landing
		if(cnt % 10 == 0 && !swing.IsDescending(steps[0].telapsed)){
			timer.CountUS();

			// convert state and input to support-foot local coordinate
			real_t sign  = (sup == 0 ? 1.0 : -1.0);
			mat3_t S     = mat3_t::Diag(1.0, sign, 1.0);
			mat3_t Rsup  = mat3_t::Rot(steps[0].footOri[sup], 'z');
			vec3_t pswg  = S*(Rsup.trans()*(steps[0].footPos[swg] - steps[0].footPos[sup]));
			real_t rswg  =            sign*(steps[0].footOri[swg] - steps[0].footOri[sup]);
			vec3_t pland = S*(Rsup.trans()*(steps[1].footPos[swg] - steps[0].footPos[sup]));
			real_t rland =            sign*(steps[1].footOri[swg] - steps[0].footOri[sup]);
			vec3_t icp   = S*(Rsup.trans()*(steps[0].icp          - steps[0].footPos[sup]));
			vec3_t cop   = S*(Rsup.trans()*(steps[0].cop          - steps[0].footPos[sup]));
			State st, st_mod;
			Input in, in_mod;
			
			st.swg  = vec4_t(pswg [0], pswg [1], pswg[2], rswg);
			st.icp  = vec2_t(icp  [0], icp  [1]);
			in.cop  = vec2_t(cop  [0], cop  [1]);
			in.land = vec3_t(pland[0], pland[1], rland);
			in_mod  = in;

			succeeded = cap.Check(st, in_mod, st_mod, modified);
			if(succeeded && !modified){
				printf("monitor: success\n");
			}
			if(succeeded && modified){
				printf("monitor: modified\n");
				
				// convert back to global coordinate
				pland = vec3_t(in_mod.land[0], in_mod.land[1], 0.0);
				rland = in_mod.land[2];

				steps[1].footPos[swg] = Rsup*S*pland + steps[0].footPos[sup];
				steps[1].footOri[swg] =   sign*rland + steps[0].footOri[sup];

				steps[1].icp = Rsup*S*vec3_t(st_mod.icp[0], st_mod.icp[1], 0.0) + steps[0].footPos[sup];

				swing.Set(
					steps[0].footPos[swg], steps[0].footOri[swg],
					steps[1].footPos[swg], steps[1].footOri[swg]
				);

				// modified step duration
				steps[0].duration = swing.GetDuration();
				steps[0].telapsed = 0.0;

				printf("land: %f,%f  duration: %f\n", in.land.x, in.land.y, steps[0].duration);
			}
			if(!succeeded){
				printf("monitor: fail\n");
			}
		}
		
		// update swing foot position
		swing.GetTraj(steps[0].telapsed, steps[0].footPos[swg], steps[0].footOri[swg]);

		//printf("swg: %f,%f,%f\n", state.swg.x(), state.swg.y(), state.swg.z());

		// calc cop 
		if(steps[0].duration - steps[0].telapsed > 0.001){
			real_t alpha = exp((steps[0].duration - steps[0].telapsed)/cap.T);
			steps[0].cop = (steps[1].icp - alpha*steps[0].icp)/(1.0 - alpha);

			// limit cop to support region
			mat3_t Rsup      = mat3_t::Rot(steps[0].footOri[sup], 'z');
			vec3_t cop_local = Rsup.trans()*(steps[0].cop - steps[0].footPos[sup]);
			cop_local.x      = std::min(std::max(cap.cop_x.min, cop_local.x), cap.cop_x.max);
			cop_local.y      = std::min(std::max(cap.cop_y.min, cop_local.y), cap.cop_y.max);
			steps[0].cop     = Rsup*cop_local + steps[0].footPos[sup];
		}

	    // simulation step
    	comPos        += comVel * dt;
		comPos.z       = cap.h;
		comVel        += comAcc * dt;
		comVel.z       = 0.0;
		comAcc         = (1.0/(cap.T*cap.T))*(comPos - steps[0].cop) + force;
		comAcc.z       = 0.0;
		steps[0].icp   = comPos + cap.T*comVel;
		steps[0].icp.z = 0.0;	

		// switch to DSP if step duration has elapsed
		if(steps[0].telapsed >= steps[0].duration){
			// support foot exchange
			steps[0].side = !steps[0].side;
			footstep.cur++;
		
			phase = Phase::Dsp;
		}
    }

	if(1.5 <= t && t <= 1.5 + dt) {
		comVel.y += 0.0;	
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
	
	fprintf(file, 
		"%d, %f, "
		"%f, %f, %f, "
		"%f, %f, %f, "
		"%f, %f, %f, "
		"%f, %f, %f, "
		"%f, %f, %f, "
		"%d, %d\n",
		cnt, t,
		comPos.x, comPos.y, comPos.z,
		steps[0].cop.x, steps[0].cop.y, steps[0].cop.z,
		steps[0].icp.x, steps[0].icp.y, steps[0].icp.z,
		steps[0].footPos[0].x, steps[0].footPos[0].y, steps[0].footPos[0].z, 
		steps[0].footPos[1].x, steps[0].footPos[1].y, steps[0].footPos[1].z,
		(int)succeeded, (int)modified
		);
	
    t += dt;
    steps[0].telapsed += dt;

}

int main(int argc, char const *argv[]) {
	Init();

	while(phase != Phase::Stop){
		Control();
	}
	
	fclose(file);

	return 0;
}
