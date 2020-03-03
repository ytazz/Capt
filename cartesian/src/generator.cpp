#include "generator.h"

namespace Capt {

Generator::Generator(Model *model, Param *param){
  swing = new Swing(model, param);

  model->read(&omega, "omega");
}

Generator::~Generator(){
}

void Generator::calc(Footstep *footstep){
  int i = (int)footstep->size() - 1;

  // set final step's state
  ( *footstep )[i].cop = ( *footstep )[i].pos;
  ( *footstep )[i].icp = ( *footstep )[i].pos;
  i--;

  // calc N-1 to 0 step's state
  while(i > 0) {
    swing->set( ( *footstep )[i + 1].pos, ( *footstep )[i - 1].pos);
    ( *footstep )[i].cop = ( *footstep )[i].pos;
    ( *footstep )[i].icp = ( *footstep )[i].cop + exp(-omega * swing->getDuration() ) * ( ( *footstep )[i + 1].icp - ( *footstep )[i].cop );
    // printf("%d\n", i);
    // printf("  dt : %1.3lf\n", swing->getDuration() );
    // printf("  cop: %1.3lf, %1.3lf\n", ( *footstep )[i].cop.x(), ( *footstep )[i].cop.y() );
    // printf("  icp: %1.3lf, %1.3lf\n", ( *footstep )[i].icp.x(), ( *footstep )[i].icp.y() );
    i--;
  }
}

} // namespace Capt