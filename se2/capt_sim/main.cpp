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
real_t        tphase;
int           phase;

Swing           swing;
Capturability   cap;
Footstep        footstep;
			    
vec3_t          comPos;
vec3_t          comVel;
vec3_t          comAcc;
vec3_t          cop;
vec3_t          icp;
vec3_t          force;
bool            modified;
bool            succeeded;
			
Footstep::Step  steps[2];

FILE*           file;

//int             s_sup;
//vec3_t          supPos;
//real_t          supOri;
//vec3_t          swgPos;
//real_t          swgOri;
//vec3_t          landPos;
//real_t          landOri;
//real_t          duration;

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
    steps[1] = footstep.steps[0];
    
    t      = 0.0;
    tphase = 0.0;

    comPos = vec3_t(steps[1].icp.x, steps[1].icp.y, cap.h);
	comVel = vec3_t(0.0, 0.0, 0.0  );
    comAcc = vec3_t(0.0, 0.0, 0.0  );
    cop    = vec3_t(steps[1].icp.x, steps[1].icp.y, 0.0);
	icp    = vec3_t(steps[1].icp.x, steps[1].icp.y, 0.0);
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
		// support foot exchange
		steps[0] = steps[1];
		footstep.cur++;

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

			steps[1].footPos[sup] = steps[0].footPos[sup];
			steps[1].footOri[sup] = steps[0].footOri[sup];
			
			mat3_t R = mat3_t::Rot(steps[1].footOri[sup] - st1.footOri[sup], 'z');
			steps[1].footPos[swg] = R*(st1.footPos[swg] - st1.footPos[sup]) + steps[1].footPos[sup];
			steps[1].footOri[swg] =   (st1.footOri[swg] - st1.footOri[sup]) + steps[1].footOri[sup];
			
			//input.icp  = (state.footstep[state.footstep.cur + 1].icp - state.footstep[state.footstep.cur].pos) + state.su@;
		
			// update swing trajectory and detemine step duration
			swing.Set(
				steps[0].footPos[swg], steps[0].footOri[swg],
				steps[1].footPos[swg], steps[1].footOri[swg]);

			steps[0].duration = swing.GetDuration();
			
			steps[0].Print();
			steps[1].Print();
			
			phase  = Phase::Ssp;
			tphase = 0.0;
			cnt    = 0;
		}
	}
	if(phase == Phase::Ssp){
		cnt++;

		int sup =  steps[0].side;
		int swg = !steps[0].side;
		
		// do not check too frequently, do not check right before landing
		if(cnt % 10 == 0 && swing.IsDescending(tphase)){
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
			State st;
			Input in;
			
			st.swg  = vec4_t(pswg [0], pswg [1], pswg[2], rswg);
			st.icp  = vec2_t(icp  [0], icp  [1]);
			in.cop  = vec2_t(cop  [0], cop  [1]);
			in.land = vec3_t(pland[0], pland[1], rland);

			succeeded = cap.Check(st, in, modified);
			if(succeeded && !modified){
				printf("monitor: success\n");
			}
			if(succeeded && modified){
				printf("monitor: modified\n");
				// convert back to global coordinate
				pswg  = vec3_t(st.swg[0], st.swg[1], st.swg[2]);
				rswg  = st.swg[3];
				pland = vec3_t(in.land[0], in.land[1], in.land[2]);
				rland = in.land[3];

				swing.Set(
					steps[0].footPos[sup], steps[0].footOri[sup],
					steps[1].footPos[swg], steps[1].footOri[swg]
				);

				// modified step duration
				steps[0].duration = swing.GetDuration();

				printf("land: %f,%f  duration: %f\n", in.land.x, in.land.y, steps[0].duration);
			}
			if(!succeeded){
				printf("monitor: fail\n");
			}
		}

		// update swing foot position
		swing.GetTraj(tphase, steps[0].footPos[swg], steps[0].footOri[swg]);

		//printf("swg: %f,%f,%f\n", state.swg.x(), state.swg.y(), state.swg.z());

		// switch to DSP if step duration has elapsed
		if(tphase >= steps[0].duration)
			phase = Phase::Dsp;

    }

	if(4.5 <= t && t <= 4.51) {
		// simulation 1
		//force.x() = -5000;
		// simulation 2
		force.y = 5000;
	}
	else{
		force.x = 0;
		force.y = 0;
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
	if(phase == Phase::Ssp || phase == Phase::Dsp){
    	comPos   += comVel * dt;
		comPos.z  = cap.h;
		comVel   += comAcc * dt;
		comVel.z  = 0.0;
		comAcc    = (1.0/(cap.T*cap.T))*(comPos - cop) + force;
		comAcc.z  = 0.0;
		icp       = comPos + cap.T*comVel;
		icp.z     = 0.0;	
	}

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
		cop.x, cop.y, cop.z,
		icp.x, icp.y, icp.z,
		steps[0].footPos[0].x, steps[0].footPos[0].x, steps[0].footPos[0].x, 
		steps[0].footPos[1].x, steps[0].footPos[1].x, steps[0].footPos[1].x,
		(int)succeeded, (int)modified
		);
	
    t      += dt;
    tphase += dt;

}

int main(int argc, char const *argv[]) {
	Init();

	while(phase != Phase::Stop){
		Control();
	}
	
	fclose(file);

	return 0;
}
