#include "grid.h"

namespace Capt {

Grid::Grid(Param *param) : param(param), epsilon(0.001) {
  state.clear();
  input.clear();

  state_num = 0;
  input_num = 0;

  icp_x_num = 0;
  icp_y_num = 0;
  swf_x_num = 0;
  swf_y_num = 0;
  swf_z_num = 0;
  cop_x_num = 0;
  cop_y_num = 0;

  create();
}

Grid::~Grid() {
}

void Grid::create() {
  using CaptEnum::MAX;
  using CaptEnum::MIN;
  using CaptEnum::STP;

  // icp
  param->read(&icp_x[MIN], "icp_x_min");
  param->read(&icp_x[MAX], "icp_x_max");
  param->read(&icp_x[STP], "icp_x_stp");
  param->read(&icp_y[MIN], "icp_y_min");
  param->read(&icp_y[MAX], "icp_y_max");
  param->read(&icp_y[STP], "icp_y_stp");
  // swf
  param->read(&swf_x[MIN], "swf_x_min");
  param->read(&swf_x[MAX], "swf_x_max");
  param->read(&swf_x[STP], "swf_x_stp");
  param->read(&swf_y[MIN], "swf_y_min");
  param->read(&swf_y[MAX], "swf_y_max");
  param->read(&swf_y[STP], "swf_y_stp");
  param->read(&swf_z[MIN], "swf_z_min");
  param->read(&swf_z[MAX], "swf_z_max");
  param->read(&swf_z[STP], "swf_z_stp");
  // exc
  param->read(&exc_x[MIN], "exc_x_min");
  param->read(&exc_x[MAX], "exc_x_max");
  param->read(&exc_y[MIN], "exc_y_min");
  param->read(&exc_y[MAX], "exc_y_max");
  // cop
  param->read(&cop_x[MIN], "cop_x_min");
  param->read(&cop_x[MAX], "cop_x_max");
  param->read(&cop_x[STP], "cop_x_stp");
  param->read(&cop_y[MIN], "cop_y_min");
  param->read(&cop_y[MAX], "cop_y_max");
  param->read(&cop_y[STP], "cop_y_stp");

  // num of each grid
  param->read(&icp_x_num, "icp_x_num");
  param->read(&icp_y_num, "icp_y_num");
  param->read(&swf_x_num, "swf_x_num");
  param->read(&swf_y_num, "swf_y_num");
  param->read(&swf_z_num, "swf_z_num");
  param->read(&cop_x_num, "cop_x_num");
  param->read(&cop_y_num, "cop_y_num");

  // current grid value
  double icp_x_ = icp_x[MIN], icp_y_ = icp_y[MIN];
  double swf_x_ = swf_x[MIN], swf_y_ = swf_y[MIN], swf_z_ = swf_z[MIN];
  double cop_x_ = icp_x[MIN], cop_y_ = icp_y[MIN];

  // state
  state_num = 0;
  for (int i = 0; i < icp_x_num; i++) {
    icp_x_ = icp_x[MIN] + icp_x[STP] * i;
    for (int j = 0; j < icp_y_num; j++) {
      icp_y_ = icp_y[MIN] + icp_y[STP] * j;
      for (int k = 0; k < swf_x_num; k++) {
        swf_x_ = swf_x[MIN] + swf_x[STP] * k;
        for (int l = 0; l < swf_y_num; l++) {
          swf_y_ = swf_y[MIN] + swf_y[STP] * l;
          for (int m = 0; m < swf_z_num; m++) {
            swf_z_ = swf_z[MIN] + swf_z[STP] * m;
            setState(icp_x_, icp_y_, swf_x_, swf_y_, swf_z_);
          }
        }
      }
    }
  }

  FILE *fp_state = NULL;
  if ( ( fp_state = fopen("state_table.csv", "w") ) == NULL) {
    printf("Error: couldn't open state_table.csv\n");
    exit(EXIT_FAILURE);
  }
  fprintf(fp_state, "index, icp_x, icp_y, swf_x, swf_y, swf_z\n");
  for (int i = 0; i < max(icp_x_num, icp_y_num, swf_x_num, swf_y_num, swf_z_num); i++) {
    fprintf(fp_state, "%d", i);
    if (i < icp_x_num) {
      icp_x_ = icp_x[MIN] + icp_x[STP] * i;
      fprintf(fp_state, ",%lf", icp_x_);
    } else {
      fprintf(fp_state, ",");
    }
    if (i < icp_y_num) {
      icp_y_ = icp_y[MIN] + icp_y[STP] * i;
      fprintf(fp_state, ",%lf", icp_y_);
    } else {
      fprintf(fp_state, ",");
    }
    if (i < swf_x_num) {
      swf_x_ = swf_x[MIN] + swf_x[STP] * i;
      fprintf(fp_state, ",%lf", swf_x_);
    } else {
      fprintf(fp_state, ",");
    }
    if (i < swf_y_num) {
      swf_y_ = swf_y[MIN] + swf_y[STP] * i;
      fprintf(fp_state, ",%lf", swf_y_);
    } else {
      fprintf(fp_state, ",");
    }
    if (i < swf_z_num) {
      swf_z_ = swf_z[MIN] + swf_z[STP] * i;
      fprintf(fp_state, ",%lf", swf_z_);
    } else {
      fprintf(fp_state, ",");
    }
    fprintf(fp_state, "\n");
  }
  fclose(fp_state);

  // input
  input_num = 0;
  for (int i = 0; i < cop_x_num; i++) {
    cop_x_ = cop_x[MIN] + cop_x[STP] * i;
    for (int j = 0; j < cop_y_num; j++) {
      cop_y_ = cop_y[MIN] + cop_y[STP] * j;
      for (int k = 0; k < swf_x_num; k++) {
        swf_x_ = swf_x[MIN] + swf_x[STP] * k;
        for (int l = 0; l < swf_y_num; l++) {
          swf_y_ = swf_y[MIN] + swf_y[STP] * l;
          setInput(cop_x_, cop_y_, swf_x_, swf_y_);
        }
      }
    }
  }

  FILE *fp_input = NULL;
  if ( ( fp_input = fopen("input_table.csv", "w") ) == NULL) {
    printf("Error: couldn't open input_table.csv\n");
    exit(EXIT_FAILURE);
  }
  fprintf(fp_input, "index,cop_x,cop_y,swf_x,swf_y\n");
  for (int i = 0; i < max(cop_x_num, cop_y_num, swf_x_num, swf_y_num); i++) {
    fprintf(fp_input, "%d", i);
    if (i < cop_x_num) {
      cop_x_ = cop_x[MIN] + cop_x[STP] * i;
      fprintf(fp_input, ",%lf", cop_x_);
    } else {
      fprintf(fp_input, ",");
    }
    if (i < cop_y_num) {
      cop_y_ = cop_y[MIN] + cop_y[STP] * i;
      fprintf(fp_input, ",%lf", cop_y_);
    } else {
      fprintf(fp_input, ",");
    }
    if (i < swf_x_num) {
      swf_x_ = swf_x[MIN] + swf_x[STP] * i;
      fprintf(fp_input, ",%lf", swf_x_);
    } else {
      fprintf(fp_input, ",");
    }
    if (i < swf_y_num) {
      swf_y_ = swf_y[MIN] + swf_y[STP] * i;
      fprintf(fp_input, ",%lf", swf_y_);
    } else {
      fprintf(fp_input, ",");
    }
    fprintf(fp_input, "\n");
  }
  fclose(fp_input);
}

int Grid::max(int val1, int val2) {
  int max_val = 0;

  if (val1 >= val2)
    max_val = val1;
  else
    max_val = val2;

  return max_val;
}

int Grid::max(int val1, int val2, int val3, int val4) {
  int max_val = 0;

  max_val = max(max(val1, val2), max(val3, val4) );

  return max_val;
}

int Grid::max(int val1, int val2, int val3, int val4, int val5) {
  int max_val = 0;

  max_val = max(max(val1, val2), max(max(val3, val4), val5) );

  return max_val;
}

vec2_t Grid::roundVec(vec2_t vec){
  using CaptEnum::MAX;
  using CaptEnum::MIN;
  using CaptEnum::STP;
  int    idx = round( vec.x() / icp_x[STP]);
  int    idy = round( vec.y() / icp_y[STP]);
  vec2_t vec_(icp_x[STP] * idx, icp_y[STP] * idy);
  return vec_;
}

GridState Grid::roundState(State state_) {
  using CaptEnum::MAX;
  using CaptEnum::MIN;
  using CaptEnum::STP;

  int       state_id = -1;
  GridState gs;

  int icp_x_id = 0, icp_y_id = 0;
  int swf_x_id = 0, swf_y_id = 0, swf_z_id = 0;

  icp_x_id = round( ( state_.icp.x() - icp_x[MIN] ) / icp_x[STP]);
  icp_y_id = round( ( state_.icp.y() - icp_y[MIN] ) / icp_y[STP]);
  swf_x_id = round( ( state_.swf.x() - swf_x[MIN] ) / swf_x[STP]);
  swf_y_id = round( ( state_.swf.y() - swf_y[MIN] ) / swf_y[STP]);
  swf_z_id = round( ( state_.swf.z() - swf_z[MIN] ) / swf_z[STP]);

  if(0 <= icp_x_id && icp_x_id < icp_x_num &&
     0 <= icp_y_id && icp_y_id < icp_y_num &&
     0 <= swf_x_id && swf_x_id < swf_x_num &&
     0 <= swf_y_id && swf_y_id < swf_y_num &&
     0 <= swf_z_id && swf_z_id < swf_z_num )
    state_id = getStateIndex(icp_x_id, icp_y_id, swf_x_id, swf_y_id, swf_z_id);

  gs.id    = state_id;
  gs.state = getState(state_id);

  return gs;
}

int Grid::getStateIndex(State state_) {
  return roundState(state_).id;
}

void Grid::setState(double icp_x, double icp_y, double swf_x, double swf_y, double swf_z) {
  State state_(icp_x, icp_y, swf_x, swf_y, swf_z);
  state.push_back(state_);
  state_num++;
}

void Grid::setInput(double cop_x, double cop_y, double swf_x, double swf_y) {
  Input input_(cop_x, cop_y, swf_x, swf_y);
  input.push_back(input_);
  input_num++;
}

bool Grid::existState(int state_id) {
  bool is_exist = false;
  if (0 <= state_id && state_id < state_num)
    is_exist = true;

  return is_exist;
}

bool Grid::existState(State state_) {
  using CaptEnum::MAX;
  using CaptEnum::MIN;
  using CaptEnum::STP;

  bool flag = false;

  bool flag_icp_x = false, flag_icp_y = false;
  bool flag_swf_x = false, flag_swf_y = false;
  bool flag_swf_z = false;

  // icp_x
  if (state_.icp.x() >= icp_x[MIN] - icp_x[STP] / 2.0 &&
      state_.icp.x() < icp_x[MAX] + icp_x[STP] / 2.0) {
    flag_icp_x = true;
  }
  // icp_y
  if (state_.icp.y() >= icp_y[MIN] - icp_y[STP] / 2.0 &&
      state_.icp.y() < icp_y[MAX] + icp_y[STP] / 2.0) {
    flag_icp_y = true;
  }
  // swf_x
  if (state_.swf.x() >= swf_x[MIN] - swf_x[STP] / 2.0 &&
      state_.swf.x() < swf_x[MAX] + swf_x[STP] / 2.0) {
    flag_swf_x = true;
  }
  // swf_y
  if (state_.swf.y() >= swf_y[MIN] - swf_y[STP] / 2.0 &&
      state_.swf.y() < swf_y[MAX] + swf_y[STP] / 2.0) {
    flag_swf_y = true;
  }
  // swf_z
  if (state_.swf.z() >= swf_z[MIN] - swf_z[STP] / 2.0 &&
      state_.swf.z() < swf_z[MAX] + swf_z[STP] / 2.0) {
    flag_swf_z = true;
  }

  flag = flag_icp_x * flag_icp_y * flag_swf_x * flag_swf_y * flag_swf_z;

  return flag;
}

bool Grid::existInput(int input_id) {
  bool is_exist = false;
  if (0 <= input_id && input_id < input_num)
    is_exist = true;

  return is_exist;
}

int Grid::getStateIndex(int icp_x_id, int icp_y_id, int swf_x_id, int swf_y_id, int swf_z_id) {
  int index = 0;
  index = swf_y_num * swf_x_num * icp_y_num * swf_z_num * icp_x_id +
          swf_y_num * swf_x_num * swf_z_num * icp_y_id +
          swf_y_num * swf_z_num * swf_x_id +
          swf_z_num * swf_y_id +
          swf_z_id;
  return index;
}

State Grid::getState(int index) {
  State state_;
  if (existState(index) ) {
    state_ = state[index];
  }else{
    state_.set(-1, -1, -1, -1, -1);
  }

  return state_;
}

Input Grid::getInput(int index) {
  Input input_;
  if (existInput(index) ) {
    input_ = input[index];
  }else{
    input_.set(-1, -1, -1, -1);
  }
  return input_;
}

vec2_t Grid::getIcp(int index){
  vec2_t icp_(-1, -1);

  int idx = index / icp_y_num;
  int idy = index % icp_y_num;
  if(0 <= idx && idx < icp_x_num && 0 <= idy && idy < icp_y_num) {
    double icp_x_ = icp_x[CaptEnum::MIN] + icp_x[CaptEnum::STP] * idx;
    double icp_y_ = icp_y[CaptEnum::MIN] + icp_y[CaptEnum::STP] * idy;
    icp_ << icp_x_, icp_y_;
  }

  return icp_;
}

vec2_t Grid::getSwf(int index){
  vec2_t swf_(-1, -1);

  int idx = index / swf_y_num;
  int idy = index % swf_y_num;
  if(0 <= idx && idx < swf_x_num && 0 <= idy && idy < swf_y_num) {
    double swf_x_ = swf_x[CaptEnum::MIN] + swf_x[CaptEnum::STP] * idx;
    double swf_y_ = swf_y[CaptEnum::MIN] + swf_y[CaptEnum::STP] * idy;
    swf_ << swf_x_, swf_y_;
  }

  return swf_;
}

vec2_t Grid::getCop(int index){
  vec2_t cop_(-1, -1);

  int idx = index / cop_y_num;
  int idy = index % cop_y_num;
  if(0 <= idx && idx < cop_x_num && 0 <= idy && idy < cop_y_num) {
    double cop_x_ = cop_x[CaptEnum::MIN] + cop_x[CaptEnum::STP] * idx;
    double cop_y_ = cop_y[CaptEnum::MIN] + cop_y[CaptEnum::STP] * idy;
    cop_ << cop_x_, cop_y_;
  }

  return cop_;
}

int Grid::indexIcp(vec2_t icp){
  int id = -1;

  int icp_x_id = round( ( icp.x() - icp_x[CaptEnum::MIN] ) / icp_x[CaptEnum::STP]);
  int icp_y_id = round( ( icp.y() - icp_y[CaptEnum::MIN] ) / icp_y[CaptEnum::STP]);
  if(0 <= icp_x_id && icp_x_id < icp_x_num &&
     0 <= icp_y_id && icp_y_id < icp_y_num) {
    id = icp_y_num * icp_x_id + icp_y_id;
  }

  return id;
}

int Grid::indexSwf(vec2_t swf){
  int id = -1;

  int swf_x_id = round( ( swf.x() - swf_x[CaptEnum::MIN] ) / swf_x[CaptEnum::STP]);
  int swf_y_id = round( ( swf.y() - swf_y[CaptEnum::MIN] ) / swf_y[CaptEnum::STP]);
  // int swf_z_id = round( ( swf.z() - swf_z[CaptEnum::MIN] ) / swf_z[CaptEnum::STP]);
  int swf_z_id = 0;
  if(0 <= swf_x_id && swf_x_id < swf_x_num &&
     0 <= swf_y_id && swf_y_id < swf_y_num &&
     0 <= swf_z_id && swf_z_id < swf_z_num ) {
    id = swf_z_num * swf_y_num * swf_x_id + swf_z_num * swf_y_id + swf_z_id;
  }

  return id;
}

int Grid::indexCop(vec2_t cop){
  int id = -1;

  int cop_x_id = round( ( cop.x() - cop_x[CaptEnum::MIN] ) / cop_x[CaptEnum::STP]);
  int cop_y_id = round( ( cop.y() - cop_y[CaptEnum::MIN] ) / cop_y[CaptEnum::STP]);
  if(0 <= cop_x_id && cop_x_id < cop_x_num &&
     0 <= cop_y_id && cop_y_id < cop_y_num) {
    id = cop_y_num * cop_x_id + cop_y_id;
  }

  return id;
}

bool Grid::isSteppable(vec2_t swf){
  using CaptEnum::MAX;
  using CaptEnum::MIN;

  bool flag_x = true, flag_y = true;
  if(exc_x[MIN] - epsilon <= swf.x() && swf.x() <= exc_x[MAX] + epsilon)
    flag_x = false;
  if(exc_y[MIN] - epsilon <= swf.y() && swf.y() <= exc_y[MAX] + epsilon)
    flag_y = false;

  return ( flag_x || flag_y );
}

int Grid::getNumState() {
  return state_num;
}

int Grid::getNumInput() {
  return input_num;
}

int Grid::getNumGrid() {
  return state_num * input_num;
}

} // namespace Capt