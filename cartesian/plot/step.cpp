#include "Capt.h"
#include <iostream>

using namespace std;
using namespace Capt;

int main(int argc, char const *argv[]) {
  Model *model = new Model("data/valkyrie.xml");
  Param *param = new Param("data/valkyrie_xy.xml");
  Grid  *grid  = new Grid(param);

  double foot_vel_max, step_time_min, omega;
  model->read(&omega, "omega");
  model->read(&step_time_min, "step_time_min");
  model->read(&foot_vel_max, "foot_vel_max");

  int   stateId, inputId;
  State state;
  Input input;
  std::cout << "stateId: ";
  std::cin >> stateId;
  std::cout << "inputId: ";
  std::cin >> inputId;

  state = grid->getState(stateId);
  input = grid->getInput(inputId);

  double dist = ( input.swf - state.swf ).norm();
  double tau  = max(0, step_time_min / 2 - state.elp) +
                dist / foot_vel_max +
                step_time_min / 2;

  Pendulum pendulum(model);
  pendulum.setCop(input.cop);
  pendulum.setIcp(state.icp);
  vec2_t icp = pendulum.getIcp(tau);

  State state_;
  state_.icp << -input.swf.x() + icp.x(), input.swf.y() - icp.y();
  state_.swf << -input.swf.x(), input.swf.y();
  state_.elp = 0.0;

  int next_id = grid->roundState(state_).id;

  state.print();
  input.print();
  printf("omega: %+1.3lf\n", omega);
  printf("t_min: %+1.3lf\n", step_time_min);
  printf("v_max: %+1.3lf\n", foot_vel_max);

  printf("tau  : %+1.3lf\n", tau);
  printf("dist : %+1.3lf\n", dist);
  printf("cop  : %+1.3lf, %+1.3lf\n", input.cop.x(), input.cop.y() );
  printf("icp  : %+1.3lf, %+1.3lf\n", state.icp.x(), state.icp.y() );
  printf("icp' : %+1.3lf, %+1.3lf\n", icp.x(), icp.y() );

  printf("----------\n");
  state_.print();
  printf("next_id: %d\n", next_id);
  grid->roundState(state_).state.print();

  return 0;
}