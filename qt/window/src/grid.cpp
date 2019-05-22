#include "grid.h"

namespace CA {

Grid::Grid(Param param) {
  this->param = param;

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
    fprintf(fp_state, "index");
    for (int i = 0; i < max(num_icp_r, num_icp_th, num_swft_r, num_swft_th);
         i++) {
      fprintf(fp_state, ",%d", i);
    }
    fprintf(fp_state, "\nicp_r");
    for (int i = 0; i < num_icp_r; i++) {
      icp_r_ = icp_r[MIN] + icp_r[STEP] * i;
      fprintf(fp_state, ",%lf", icp_r_);
    }
    fprintf(fp_state, "\nicp_th");
    for (int j = 0; j < num_icp_th; j++) {
      icp_th_ = icp_th[MIN] + icp_th[STEP] * j;
      fprintf(fp_state, ",%lf", icp_th_);
    }
    fprintf(fp_state, "\nswft_r");
    for (int k = 0; k < num_swft_r; k++) {
      swft_r_ = swft_r[MIN] + swft_r[STEP] * k;
      fprintf(fp_state, ",%lf", swft_r_);
    }
    fprintf(fp_state, "\nswft_th");
    for (int l = 0; l < num_swft_th; l++) {
      swft_th_ = swft_th[MIN] + swft_th[STEP] * l;
      fprintf(fp_state, ",%lf", swft_th_);
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
    fprintf(fp_input, "index");
    for (int i = 0; i < max(num_swft_r, num_swft_th); i++) {
      fprintf(fp_input, ",%d", i);
    }
    fprintf(fp_input, "\nswft_r");
    for (int k = 0; k < num_swft_r; k++) {
      swft_r_ = swft_r[MIN] + swft_r[STEP] * k;
      fprintf(fp_input, ",%lf", swft_r_);
    }
    fprintf(fp_input, "\nswft_th");
    for (int l = 0; l < num_swft_th; l++) {
      swft_th_ = swft_th[MIN] + swft_th[STEP] * l;
      fprintf(fp_input, ",%lf", swft_th_);
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

State Grid::getState(int index) {
  if (index > num_state) {
    printf("Error: state id(%d) is larger than number of states(%d)\n", index,
           num_state);
    exit(EXIT_FAILURE);
  }

  return state[index];
}

State Grid::getState(int icp_r_id, int icp_th_id, int swft_r_id,
                     int swft_th_id) {
  int index = 0;
  index = num_swft_th * num_swft_r * num_icp_th * icp_r_id +
          num_swft_th * num_swft_r * icp_th_id + num_swft_th * swft_r_id +
          swft_th_id;

  return getState(index);
}

Input Grid::getInput(int index) {
  if (index > num_input) {
    printf("Error: input id(%d) is larger than number of inputs(%d)\n", index,
           num_input);
    exit(EXIT_FAILURE);
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

} // namespace CA