#include "footstep.h"
#include "capturability.h"
#include "swing.h"

namespace Capt {

void Footstep::Step::Read(Scenebuilder::XMLNode* node){
	node->Get(pos, ".pos");
	node->Get(ori, ".ori");
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
	int i = (int)steps.size() - 1;

	// set final step's state
	steps[i].cop = steps[i].pos;
	steps[i].icp = steps[i].pos;
	i--;

	// calc N-1 to 0 step's state
	for( ; i >= 0; i--) {
		if(i > 0)
			 swing->Set(steps[i-1].pos, steps[i-1].ori, steps[i+1].pos, steps[i+1].ori);
		else swing->Set(steps[i+1].pos, steps[i+1].ori, steps[i+1].pos, steps[i+1].ori);

		steps[i].cop = steps[i].pos;
		steps[i].icp = steps[i].cop + exp(-swing->GetDuration()/cap->T)*(steps[i+1].icp - steps[i].cop);
	}
}

}
