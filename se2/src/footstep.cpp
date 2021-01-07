#include "footstep.h"
#include "capturability.h"
#include "swing.h"

namespace Capt {

const real_t eps = 1.0e-10;

Footstep::Step::Step(){
	stride  = 0.0;
	spacing = 0.0;
	turn    = 0.0;
		
	side       = 0;
	footPos[0] = vec3_t(0.0, 0.0, 0.0);
	footPos[1] = vec3_t(0.0, 0.0, 0.0);
	footOri[0] = 0.0;
	footOri[1] = 0.0;
	cop        = vec3_t(0.0, 0.0, 0.0);
	icp        = vec3_t(0.0, 0.0, 0.0);
	duration = 0.0;
	telapsed = 0.0;
}

void Footstep::Step::Read(Scenebuilder::XMLNode* node){
	node->Get(stride  , ".stride"  );
	node->Get(spacing , ".spacing" );
	node->Get(turn    , ".turn"    );
	node->Get(duration, ".duration");
}

void Footstep::Step::Print(){
	printf("side: %d  foot0: %f %f %f %f  foot1: %f %f %f %f  duration: %f\n",
		side, 
		footPos[0].x, footPos[0].y, footPos[0].z, footOri[0],
		footPos[1].x, footPos[1].y, footPos[1].z, footOri[1],
		duration
	);
		
}

void Footstep::Read(Scenebuilder::XMLNode* node){
	for(int i = 0; ; i++) try{
		Scenebuilder::XMLNode* stepNode = node->GetNode("step", i);
		Step step;
		step.Read(stepNode);
		steps.push_back(step);
	}
	catch(Scenebuilder::Exception&){ break; }
}

void Footstep::Calc(Capturability* cap, Swing* swing){
	//
	steps[0].side       = 0;
	steps[0].footPos[0] = vec3_t(0.0, -steps[0].spacing/2.0, 0.0);
	steps[0].footOri[0] = 0.0;
	steps[0].footPos[1] = vec3_t(0.0,  steps[0].spacing/2.0, 0.0);
	steps[0].footOri[1] = 0.0;

	for(int i = 0; i < steps.size()-1; i++){
		steps[i+1].side = !steps[i].side;
		int sup =  steps[i].side;
		int swg = !steps[i].side;
		steps[i+1].footPos[sup] = steps[i].footPos[sup];
		steps[i+1].footOri[sup] = steps[i].footOri[sup];
		
		mat3_t R = mat3_t::Rot(steps[i].footOri[sup], 'z');
		vec3_t dp;
		real_t theta = steps[i].turn;
		real_t l = steps[i].stride;
		real_t w = (steps[i].side == 0 ? 1.0 : -1.0)*steps[i].spacing;
			
		if(std::abs(theta) < eps){
			dp = vec3_t(l, w, 0.0);
		}
		else{
			real_t r = l/theta;
			dp = vec3_t(
				(r - w/2.0)*sin(theta),
				(r + w/2.0) - (r - w/2.0)*cos(theta),
				0.0);
		}
		steps[i+1].footPos[swg] = R*dp + steps[i].footPos[sup];
		steps[i+1].footOri[swg] = steps[i].footOri[sup] + steps[i].turn;
	}

	int i = (int)steps.size() - 1;

	// set final step's state
	steps[i].cop = (steps[i].footPos[0] + steps[i].footPos[1])/2.0;
	steps[i].icp = (steps[i].footPos[0] + steps[i].footPos[1])/2.0;
	i--;

	// calc N-1 to 0 step's state
	for( ; i >= 0; i--) {
		int sup =  steps[i].side;
		steps[i].cop = steps[i].footPos[sup];
		steps[i].icp = steps[i].cop + exp(-steps[i].duration/cap->T)*(steps[i+1].icp - steps[i].cop);
	}
}

}
