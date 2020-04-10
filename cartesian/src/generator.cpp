#include "generator.h"

namespace Capt {

Generator::Generator(Model *model){
  swing = new Swing(model);

  model->read(&omega, "omega");
}

Generator::~Generator(){
}

void Generator::calc(Footstep& footstep){
  int i = (int)footstep.size() - 1;

  // set final step's state
  footstep[i].cop = footstep[i].pos;
  footstep[i].icp = footstep[i].pos;
  i--;

  // calc N-1 to 0 step's state
  while(i >= 0) {
    if(i > 0)
         swing->set(footstep[i-1].pos, footstep[i+1].pos);
    else swing->set(footstep[i+1].pos, footstep[i+1].pos);
    footstep[i].cop = footstep[i].pos;
    footstep[i].icp = footstep[i].cop + exp(-omega * swing->getDuration())*(footstep[i + 1].icp - footstep[i].cop);
    i--;
  }
}

} // namespace Capt