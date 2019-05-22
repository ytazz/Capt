#ifndef __CAPTURABILITY_H__
#define __CAPTURABILITY_H__

#include "grid.h"
#include "input.h"
#include "model.h"
#include "param.h"
#include "pendulum.h"
#include "state.h"
#include "swing_foot.h"
#include "vector.h"
#include <vector>

namespace CA {

struct CaptureState {
  GridState grid_state;
  std::vector<GridInput> grid_input;
  std::vector<GridState> next_grid_state;
  std::vector<int> n_step_capturable;

  int size() { return grid_input.size(); }
};

class Capturability {
public:
  Capturability();
  ~Capturability();

  // for analyzing capturability
  void setState(State state);
  void setInput(Input input);

  Vector2 getCop();
  Vector2 getDcm(float dt);
  GridState getNextState(float dt); // state after step
  std::vector<Vector2> getSupportRegion();

  bool inPolygon(Vector2 point_, std::vector<Vector2> support_region_);

  void setCapturable(bool is_exist);
  bool getCapturable(int state_id_);

  // for using pre-computed capture region data
  void readDatabase(std::string file_path);
  CaptureState getCaptureRegion(int grid_state_id_, int n_step_capturable_);

private:
  Grid grid;

  // for analyzing capturability
  State state;
  Input input;
  std::vector<std::vector<bool>> capturable;

  // for using pre-computed capture region data
  std::vector<CaptureState> capture_state;
  void setCaptureRegion(GridState grid_state_, GridInput grid_input_,
                        GridState next_grid_state_, int n_step_capturable_);
};

} // namespace CA

#endif // __CAPTURABILITY_H__
