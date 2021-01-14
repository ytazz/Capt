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

struct Impulse{
	real_t  time;
	vec3_t  deltaVel;
};

real_t        dt;
real_t        t;
int           phase;

Swing            swing;
Capturability    cap;
Footstep         footstep;

vector<Impulse>  disturbance;
			    
vec3_t          comPos;
vec3_t          comVel;
vec3_t          comAcc;
vec3_t          force;
int             nstep;
bool            modified;
bool            succeeded;
int             tcheck;
			
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

	XMLNode* distNode = xmlSim.GetRootNode()->GetNode("disturbance");
	for(int i = 0; ; i++)try{
		XMLNode* imNode = distNode->GetNode("impulse", i);
		Impulse im;
		imNode->Get(im.time    , ".time"     );
		imNode->Get(im.deltaVel, ".delta_vel");
		disturbance.push_back(im);
	}
	catch(Exception&){ break; }

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
		"foot0_x, foot0_y, foot0_z, foot0_r, "
		"foot1_x, foot1_y, foot1_z, foot1_r, "
		"duration, elapsed, "
		"nstep, succeeded, modified, tcheck\n"
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
			swing.SetSwg     (steps[0].footPos[swg], steps[0].footOri[swg]);
			swing.SetLand    (steps[1].footPos[swg], steps[1].footOri[swg]);
			swing.SetDuration(st0.duration);

			steps[0].duration = st0.duration;
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
			real_t s[2];
			mat3_t S[2];
			mat3_t R[2];
			s[sup] = (sup == 0 ? 1.0 : -1.0);
			s[swg] = (swg == 0 ? 1.0 : -1.0);
			S[sup] = mat3_t::Diag(1.0, s[sup], 1.0);
			S[swg] = mat3_t::Diag(1.0, s[swg], 1.0);
			R[sup] = mat3_t::Rot(steps[0].footOri[sup], 'z');
			R[swg] = mat3_t::Rot(steps[1].footOri[swg], 'z');
			vec3_t pswg      = S[sup]*(R[sup].trans()*(steps[0].footPos[swg] - steps[0].footPos[sup]));
			vec3_t pland     = S[sup]*(R[sup].trans()*(steps[1].footPos[swg] - steps[0].footPos[sup]));
			vec3_t pswg_next = S[swg]*(R[swg].trans()*(steps[1].footPos[sup] - steps[1].footPos[swg]));
			real_t rswg      = WrapRadian(s[sup]*(steps[0].footOri[swg] - steps[0].footOri[sup]));
			real_t rland     = WrapRadian(s[sup]*(steps[1].footOri[swg] - steps[0].footOri[sup]));
			real_t rswg_next = WrapRadian(s[swg]*(steps[1].footOri[sup] - steps[1].footOri[swg]));
			vec3_t icp       = S[sup]*(R[sup].trans()*(steps[0].icp - steps[0].footPos[sup]));
			vec3_t icp_next  = S[swg]*(R[swg].trans()*(steps[1].icp - steps[1].footPos[swg]));
			vec3_t cop       = S[sup]*(R[sup].trans()*(steps[0].cop - steps[0].footPos[sup]));
			
			State st;
			State stnext;
			Input in;
			
			st    .swg  = vec4_t(pswg [0], pswg [1], pswg[2], rswg);
			st    .icp  = vec2_t(icp  [0], icp  [1]);
			in    .cop  = vec2_t(cop  [0], cop  [1]);
			in    .land = vec3_t(pland[0], pland[1], rland);
			in    .tau  = (steps[0].duration - steps[0].telapsed);
			stnext.swg  = vec4_t(pswg_next [0], pswg_next [1], pswg_next[2], rswg_next);
			stnext.icp  = vec2_t(icp_next  [0], icp_next  [1]);
			
			Input in_mod     = in;
			State stnext_mod = stnext;
			
			timer.CountUS();
			succeeded = cap.Check(st, in_mod, stnext_mod, nstep, modified);
			tcheck = timer.CountUS();

			if(succeeded && !modified){
				printf("monitor: success\n");
			}
			if(succeeded && modified){
				printf("monitor: modified\n");
				
				// convert back to global coordinate
				pland = vec3_t(in_mod.land[0], in_mod.land[1], 0.0);
				rland = in_mod.land[2];

				steps[1].footPos[swg] = R[sup]*S[sup]*pland + steps[0].footPos[sup];
				steps[1].footOri[swg] =        s[sup]*rland + steps[0].footOri[sup];

				// next icp should be converted from next support foot's local coordinate
				R[swg] = mat3_t::Rot(steps[1].footOri[swg], 'z');
				steps[1].icp = R[swg]*S[swg]*vec3_t(stnext_mod.icp[0], stnext_mod.icp[1], 0.0) + steps[1].footPos[swg];

				swing.SetSwg     (steps[0].footPos[swg], steps[0].footOri[swg]);
				swing.SetLand    (steps[1].footPos[swg], steps[1].footOri[swg]);
				swing.SetDuration(in_mod.tau);

				// modified step duration
				steps[0].duration = in_mod.tau;
				steps[0].telapsed = 0.0;

				printf("land: %f,%f  duration: %f\n", in.land.x, in.land.y, steps[0].duration);
			}
			if(!succeeded){
				printf("monitor: fail\n");
			}
		}
		
		// update swing foot position
		vec3_t v;
		real_t w;
		swing.GetTraj(steps[0].telapsed, steps[0].footPos[swg], steps[0].footOri[swg], v, w);

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

	for(Impulse& im : disturbance){
		if(im.time <= t && t <= im.time + dt){
			comVel += im.deltaVel;
		}
	}
	
	fprintf(file, 
		"%d, %f, "
		"%f, %f, %f, "
		"%f, %f, %f, "
		"%f, %f, %f, "
		"%f, %f, %f, %f, "
		"%f, %f, %f, %f, "
		"%f, %f, "
		"%d, %d, %d, %d\n",
		cnt, t,
		comPos.x, comPos.y, comPos.z,
		steps[0].cop.x, steps[0].cop.y, steps[0].cop.z,
		steps[0].icp.x, steps[0].icp.y, steps[0].icp.z,
		steps[0].footPos[0].x, steps[0].footPos[0].y, steps[0].footPos[0].z, steps[0].footOri[0],
		steps[0].footPos[1].x, steps[0].footPos[1].y, steps[0].footPos[1].z, steps[0].footOri[1],
		steps[0].duration, steps[0].telapsed,
		nstep, (int)succeeded, (int)modified, tcheck
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
