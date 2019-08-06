#include "grid.h"

namespace Capt {

Grid::Grid(Param param) : param(param) {
  state.clear();
  input.clear();

  num_state = 0;
  num_input = 0;

  num_icp_r = 0.0;
  num_icp_th = 0.0;
  num_swft_r = 0.0;
  num_swft_th = 0.0;
  num_swft_r = 0.0;
  num_swft_th = 0.0;

  num_icp_x = 0.0;
  num_icp_y = 0.0;
  num_swft_x = 0.0;
  num_swft_y = 0.0;
  num_swft_x = 0.0;
  num_swft_y = 0.0;

  create();
}

Grid::~Grid() {}

void Grid::create() {
  using GridSpace::MAX;
  using GridSpace::MIN;
  using GridSpace::STEP;

  std::string coord = param.getStr("coordinate", "type");
  if (strcmp(coord.c_str(), "polar") == 0) {
    icp_r[MIN] = param.getVal("icp_r", "min");
    icp_r[MAX] = param.getVal("icp_r", "max");
    icp_r[STEP] = param.getVal("icp_r", "step");
    icp_th[MIN] = param.getVal("icp_th", "min");
    icp_th[MAX] = param.getVal("icp_th", "max");
    icp_th[STEP] = param.getVal("icp_th", "step");
    swft_r[MIN] = param.getVal("swft_r", "min");
    swft_r[MAX] = param.getVal("swft_r", "max");
    swft_r[STEP] = param.getVal("swft_r", "step");
    swft_th[MIN] = param.getVal("swft_th", "min");
    swft_th[MAX] = param.getVal("swft_th", "max");
    swft_th[STEP] = param.getVal("swft_th", "step");

    // num of each grid
    num_icp_r = round((icp_r[MAX] - icp_r[MIN]) / icp_r[STEP]) + 1;
    num_icp_th = round((icp_th[MAX] - icp_th[MIN]) / icp_th[STEP]) + 1;
    num_swft_r = round((swft_r[MAX] - swft_r[MIN]) / swft_r[STEP]) + 1;
    num_swft_th = round((swft_th[MAX] - swft_th[MIN]) / swft_th[STEP]) + 1;

    // current grid value
    float icp_r_ = icp_r[MIN], icp_th_ = icp_th[MIN];
    float swft_r_ = icp_r[MIN], swft_th_ = icp_th[MIN];

    // state
    num_state = 0;
    for (int i = 0; i < num_icp_r; i++) {
      icp_r_ = icp_r[MIN] + icp_r[STEP] * i;
      for (int j = 0; j < num_icp_th; j++) {
        icp_th_ = icp_th[MIN] + icp_th[STEP] * j;
        for (int k = 0; k < num_swft_r; k++) {
          swft_r_ = swft_r[MIN] + swft_r[STEP] * k;
          for (int l = 0; l < num_swft_th; l++) {
            swft_th_ = swft_th[MIN] + swft_th[STEP] * l;
            setStatePolar(icp_r_, icp_th_, swft_r_, swft_th_);
          }
        }
      }
    }

    // printf("state = %d\n", num_state);
    // printf("num_icp_r   = %d\n", num_icp_r);
    // printf("num_icp_th  = %d\n", num_icp_th);
    // printf("num_swft_r  = %d\n", num_swft_r);
    // printf("num_swft_th = %d\n", num_swft_th);

    FILE *fp_state = NULL;
    if ((fp_state = fopen("state_table.csv", "w")) == NULL) {
      printf("Error: couldn't open state_table.csv\n");
      exit(EXIT_FAILURE);
    }
    fprintf(fp_state, "index,icp_r,icp_th,swft_r,swft_th\n");
    for (int i = 0; i < max(num_icp_r, num_icp_th, num_swft_r, num_swft_th);
         i++) {
      fprintf(fp_state, "%d", i);
      if (i < num_icp_r) {
        icp_r_ = icp_r[MIN] + icp_r[STEP] * i;
        fprintf(fp_state, ",%lf", icp_r_);
      } else {
        fprintf(fp_state, ",");
      }
      if (i < num_icp_th) {
        icp_th_ = icp_th[MIN] + icp_th[STEP] * i;
        fprintf(fp_state, ",%lf", icp_th_);
      } else {
        fprintf(fp_state, ",");
      }
      if (i < num_swft_r) {
        swft_r_ = swft_r[MIN] + swft_r[STEP] * i;
        fprintf(fp_state, ",%lf", swft_r_);
      } else {
        fprintf(fp_state, ",");
      }
      if (i < num_swft_th) {
        swft_th_ = swft_th[MIN] + swft_th[STEP] * i;
        fprintf(fp_state, ",%lf", swft_th_);
      } else {
        fprintf(fp_state, ",");
      }
      fprintf(fp_state, "\n");
    }
    fclose(fp_state);

    // input
    num_input = 0;
    for (int k = 0; k < num_swft_r; k++) {
      swft_r_ = swft_r[MIN] + swft_r[STEP] * k;
      for (int l = 0; l < num_swft_th; l++) {
        swft_th_ = swft_th[MIN] + swft_th[STEP] * l;
        setInputPolar(swft_r_, swft_th_);
      }
    }

    FILE *fp_input = NULL;
    if ((fp_input = fopen("input_table.csv", "w")) == NULL) {
      printf("Error: couldn't open input_table.csv\n");
      exit(EXIT_FAILURE);
    }
    fprintf(fp_input, "index,swft_r,swft_th\n");
    for (int i = 0; i < max(num_swft_r, num_swft_th); i++) {
      fprintf(fp_input, "%d", i);
      if (i < num_swft_r) {
        swft_r_ = swft_r[MIN] + swft_r[STEP] * i;
        fprintf(fp_input, ",%lf", swft_r_);
      } else {
        fprintf(fp_input, ",");
      }
      if (i < num_swft_th) {
        swft_th_ = swft_th[MIN] + swft_th[STEP] * i;
        fprintf(fp_input, ",%lf", swft_th_);
      } else {
        fprintf(fp_input, ",");
      }
      fprintf(fp_input, "\n");
    }
    fclose(fp_input);
  }
}

int Grid::round(float value) {
  int result = (int)value;

  float decimal = value - (int)value;
  if (decimal >= 0.5) {
    result += 1;
  }

  return result;
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

  max_val = max(max(val1, val2), max(val3, val4));

  return max_val;
}

GridState Grid::roundState(State state_) {
  using GridSpace::MAX;
  using GridSpace::MIN;
  using GridSpace::STEP;

  int icp_r_id = 0, icp_th_id = 0;
  int swft_r_id = 0, swft_th_id = 0;

  icp_r_id = round((state_.icp.r - icp_r[MIN]) / icp_r[STEP]);
  icp_th_id = round((state_.icp.th - icp_th[MIN]) / icp_th[STEP]);
  swft_r_id = round((state_.swft.r - swft_r[MIN]) / swft_r[STEP]);
  swft_th_id = round((state_.swft.th - swft_th[MIN]) / swft_th[STEP]);

  int state_id = 0;
  state_id = getStateIndex(icp_r_id, icp_th_id, swft_r_id, swft_th_id);
  GridState gs;
  gs.id = state_id;
  gs.state = getState(state_id);

  return gs;
}

int Grid::getStateIndex(State state_) {
  int id = -1;

  if (existState(state_)) {
    GridState gs;
    gs = roundState(state_);
    id = gs.id;
  }

  return id;
}

void Grid::setStatePolar(float icp_r, float icp_th, float swft_r,
                         float swft_th) {
  State state_;
  state_.icp.setPolar(icp_r, icp_th);
  state_.swft.setPolar(swft_r, swft_th);

  state.push_back(state_);
  num_state++;
}

void Grid::setStateCartesian(float icp_x, float icp_y, float swft_x,
                             float swft_y) {
  State state_;
  state_.icp.setCartesian(icp_x, icp_y);
  state_.swft.setCartesian(swft_x, swft_y);

  state.push_back(state_);
  num_state++;
}

void Grid::setInputPolar(float swft_r, float swft_th) {
  Input input_;
  input_.swft.setPolar(swft_r, swft_th);

  input.push_back(input_);
  num_input++;
}

void Grid::setInputCartesian(float swft_x, float swft_y) {
  Input input_;
  input_.swft.setCartesian(swft_x, swft_y);

  input.push_back(input_);
  num_input++;
}

bool Grid::existState(int state_id) {
  bool is_exist = false;
  if (0 <= state_id && state_id <= num_state)
    is_exist = true;

  return is_exist;
}

bool Grid::existState(State state_) {
  using GridSpace::MAX;
  using GridSpace::MIN;
  using GridSpace::STEP;

  bool flag_icp_r = false, flag_icp_th = false;
  bool flag_swft_r = false, flag_swft_th = false;

  // icp_r
  if (state_.icp.r >= icp_r[MIN] - icp_r[STEP] / 2.0 &&
      state_.icp.r < icp_r[MAX] + icp_r[STEP] / 2.0) {
    flag_icp_r = true;
  }
  // icp_th
  if (state_.icp.th >= icp_th[MIN] - icp_th[STEP] / 2.0 &&
      state_.icp.th < icp_th[MAX] + icp_th[STEP] / 2.0) {
    flag_icp_th = true;
  }
  // swft_r
  if (state_.swft.r >= swft_r[MIN] - swft_r[STEP] / 2.0 &&
      state_.swft.r < swft_r[MAX] + swft_r[STEP] / 2.0) {
    flag_swft_r = true;
  }
  // swft_th
  if (state_.swft.th >= swft_th[MIN] - swft_th[STEP] / 2.0 &&
      state_.swft.th < swft_th[MAX] + swft_th[STEP] / 2.0) {
    flag_swft_th = true;
  }

  bool flag = flag_icp_r * flag_icp_th * flag_swft_r * flag_swft_th;
  return flag;
}

bool Grid::existInput(int input_id) {
  bool is_exist = false;
  if (0 <= input_id && input_id <= num_input)
    is_exist = true;

  return is_exist;
}

State Grid::getState(int index) {
  if (!existState(index)) {
    // printf("Error: state id(%d) is larger than number of states(%d)\n",
    // index,
    //        num_state);
    // exit(EXIT_FAILURE);
  }

  return state[index];
}

int Grid::getStateIndex(int icp_r_id, int icp_th_id, int swft_r_id,
                        int swft_th_id) {
  int index = 0;
  index = num_swft_th * num_swft_r * num_icp_th * icp_r_id +
          num_swft_th * num_swft_r * icp_th_id + num_swft_th * swft_r_id +
          swft_th_id;
  return index;
}

State Grid::getState(int icp_r_id, int icp_th_id, int swft_r_id,
                     int swft_th_id) {
  int index = getStateIndex(icp_r_id, icp_th_id, swft_r_id, swft_th_id);
  return getState(index);
}

Input Grid::getInput(int index) {
  if (!existInput(index)) {
    // printf("Error: input id(%d) is larger than number of inputs(%d)\n",
    // index,
    //        num_input);
    // exit(EXIT_FAILURE);
  }

  return input[index];
}

Input Grid::getInput(int swft_r_id, int swft_th_id) {
  int index = 0;
  index = num_swft_th * swft_r_id + swft_th_id;

  return getInput(index);
}

int Grid::getNumState() { return num_state; }

int Grid::getNumInput() { return num_input; }

} // namespace Capt