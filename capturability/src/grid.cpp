#include "grid.h"

namespace Capt {

Grid::Grid(Param param) : param(param), coord(param.getStr("coordinate", "type") ) {
  state.clear();
  input.clear();

  num_state = 0;
  num_input = 0;

  num_icp_r  = 0.0;
  num_icp_th = 0.0;
  num_swf_r  = 0.0;
  num_swf_th = 0.0;
  num_swf_r  = 0.0;
  num_swf_th = 0.0;

  num_icp_x = 0.0;
  num_icp_y = 0.0;
  num_swf_x = 0.0;
  num_swf_y = 0.0;
  num_swf_x = 0.0;
  num_swf_y = 0.0;

  create();
}

Grid::~Grid() {
}

void Grid::create() {
  using GridSpace::MAX;
  using GridSpace::MIN;
  using GridSpace::STEP;

  if (strcmp(coord.c_str(), "cartesian") == 0) {
    icp_x[MIN]  = param.getVal("icp_x", "min");
    icp_x[MAX]  = param.getVal("icp_x", "max");
    icp_x[STEP] = param.getVal("icp_x", "step");
    icp_y[MIN]  = param.getVal("icp_y", "min");
    icp_y[MAX]  = param.getVal("icp_y", "max");
    icp_y[STEP] = param.getVal("icp_y", "step");
    swf_x[MIN]  = param.getVal("swf_x", "min");
    swf_x[MAX]  = param.getVal("swf_x", "max");
    swf_x[STEP] = param.getVal("swf_x", "step");
    swf_y[MIN]  = param.getVal("swf_y", "min");
    swf_y[MAX]  = param.getVal("swf_y", "max");
    swf_y[STEP] = param.getVal("swf_y", "step");

    // num of each grid
    num_icp_x = round( ( icp_x[MAX] - icp_x[MIN] ) / icp_x[STEP]) + 1;
    num_icp_y = round( ( icp_y[MAX] - icp_y[MIN] ) / icp_y[STEP]) + 1;
    num_swf_x = round( ( swf_x[MAX] - swf_x[MIN] ) / swf_x[STEP]) + 1;
    num_swf_y = round( ( swf_y[MAX] - swf_y[MIN] ) / swf_y[STEP]) + 1;

    // current grid value
    float icp_x_ = icp_x[MIN], icp_y_ = icp_y[MIN];
    float swf_x_ = icp_x[MIN], swf_y_ = icp_y[MIN];

    // state
    num_state = 0;
    for (int i = 0; i < num_icp_x; i++) {
      icp_x_ = icp_x[MIN] + icp_x[STEP] * i;
      for (int j = 0; j < num_icp_y; j++) {
        icp_y_ = icp_y[MIN] + icp_y[STEP] * j;
        for (int k = 0; k < num_swf_x; k++) {
          swf_x_ = swf_x[MIN] + swf_x[STEP] * k;
          for (int l = 0; l < num_swf_y; l++) {
            swf_y_ = swf_y[MIN] + swf_y[STEP] * l;
            setStateCartesian(icp_x_, icp_y_, swf_x_, swf_y_);
          }
        }
      }
    }

    FILE *fp_state = NULL;
    if ( ( fp_state = fopen("state_table.csv", "w") ) == NULL) {
      printf("Error: couldn't open state_table.csv\n");
      exit(EXIT_FAILURE);
    }
    fprintf(fp_state, "index,icp_x,icp_y,swf_x,swf_y\n");
    for (int i = 0; i < max(num_icp_x, num_icp_y, num_swf_x, num_swf_y);
         i++) {
      fprintf(fp_state, "%d", i);
      if (i < num_icp_x) {
        icp_x_ = icp_x[MIN] + icp_x[STEP] * i;
        fprintf(fp_state, ",%lf", icp_x_);
      } else {
        fprintf(fp_state, ",");
      }
      if (i < num_icp_y) {
        icp_y_ = icp_y[MIN] + icp_y[STEP] * i;
        fprintf(fp_state, ",%lf", icp_y_);
      } else {
        fprintf(fp_state, ",");
      }
      if (i < num_swf_x) {
        swf_x_ = swf_x[MIN] + swf_x[STEP] * i;
        fprintf(fp_state, ",%lf", swf_x_);
      } else {
        fprintf(fp_state, ",");
      }
      if (i < num_swf_y) {
        swf_y_ = swf_y[MIN] + swf_y[STEP] * i;
        fprintf(fp_state, ",%lf", swf_y_);
      } else {
        fprintf(fp_state, ",");
      }
      fprintf(fp_state, "\n");
    }
    fclose(fp_state);

    // input
    num_input = 0;
    for (int k = 0; k < num_swf_x; k++) {
      swf_x_ = swf_x[MIN] + swf_x[STEP] * k;
      for (int l = 0; l < num_swf_y; l++) {
        swf_y_ = swf_y[MIN] + swf_y[STEP] * l;
        setInputCartesian(swf_x_, swf_y_);
      }
    }

    FILE *fp_input = NULL;
    if ( ( fp_input = fopen("input_table.csv", "w") ) == NULL) {
      printf("Error: couldn't open input_table.csv\n");
      exit(EXIT_FAILURE);
    }
    fprintf(fp_input, "index,swf_x,swf_y\n");
    for (int i = 0; i < max(num_swf_x, num_swf_y); i++) {
      fprintf(fp_input, "%d", i);
      if (i < num_swf_x) {
        swf_x_ = swf_x[MIN] + swf_x[STEP] * i;
        fprintf(fp_input, ",%lf", swf_x_);
      } else {
        fprintf(fp_input, ",");
      }
      if (i < num_swf_y) {
        swf_y_ = swf_y[MIN] + swf_y[STEP] * i;
        fprintf(fp_input, ",%lf", swf_y_);
      } else {
        fprintf(fp_input, ",");
      }
      fprintf(fp_input, "\n");
    }
    fclose(fp_input);
  }else if (strcmp(coord.c_str(), "polar") == 0) {
    icp_r[MIN]   = param.getVal("icp_r", "min");
    icp_r[MAX]   = param.getVal("icp_r", "max");
    icp_r[STEP]  = param.getVal("icp_r", "step");
    icp_th[MIN]  = param.getVal("icp_th", "min");
    icp_th[MAX]  = param.getVal("icp_th", "max");
    icp_th[STEP] = param.getVal("icp_th", "step");
    swf_r[MIN]   = param.getVal("swf_r", "min");
    swf_r[MAX]   = param.getVal("swf_r", "max");
    swf_r[STEP]  = param.getVal("swf_r", "step");
    swf_th[MIN]  = param.getVal("swf_th", "min");
    swf_th[MAX]  = param.getVal("swf_th", "max");
    swf_th[STEP] = param.getVal("swf_th", "step");

    // num of each grid
    num_icp_r  = round( ( icp_r[MAX] - icp_r[MIN] ) / icp_r[STEP]) + 1;
    num_icp_th = round( ( icp_th[MAX] - icp_th[MIN] ) / icp_th[STEP]) + 1;
    num_swf_r  = round( ( swf_r[MAX] - swf_r[MIN] ) / swf_r[STEP]) + 1;
    num_swf_th = round( ( swf_th[MAX] - swf_th[MIN] ) / swf_th[STEP]) + 1;

    // current grid value
    float icp_r_ = icp_r[MIN], icp_th_ = icp_th[MIN];
    float swf_r_ = icp_r[MIN], swf_th_ = icp_th[MIN];

    // state
    num_state = 0;
    for (int i = 0; i < num_icp_r; i++) {
      icp_r_ = icp_r[MIN] + icp_r[STEP] * i;
      for (int j = 0; j < num_icp_th; j++) {
        icp_th_ = icp_th[MIN] + icp_th[STEP] * j;
        for (int k = 0; k < num_swf_r; k++) {
          swf_r_ = swf_r[MIN] + swf_r[STEP] * k;
          for (int l = 0; l < num_swf_th; l++) {
            swf_th_ = swf_th[MIN] + swf_th[STEP] * l;
            setStatePolar(icp_r_, icp_th_, swf_r_, swf_th_);
          }
        }
      }
    }

    FILE *fp_state = NULL;
    if ( ( fp_state = fopen("state_table.csv", "w") ) == NULL) {
      printf("Error: couldn't open state_table.csv\n");
      exit(EXIT_FAILURE);
    }
    fprintf(fp_state, "index,icp_r,icp_th,swf_r,swf_th\n");
    for (int i = 0; i < max(num_icp_r, num_icp_th, num_swf_r, num_swf_th);
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
      if (i < num_swf_r) {
        swf_r_ = swf_r[MIN] + swf_r[STEP] * i;
        fprintf(fp_state, ",%lf", swf_r_);
      } else {
        fprintf(fp_state, ",");
      }
      if (i < num_swf_th) {
        swf_th_ = swf_th[MIN] + swf_th[STEP] * i;
        fprintf(fp_state, ",%lf", swf_th_);
      } else {
        fprintf(fp_state, ",");
      }
      fprintf(fp_state, "\n");
    }
    fclose(fp_state);

    // input
    num_input = 0;
    for (int k = 0; k < num_swf_r; k++) {
      swf_r_ = swf_r[MIN] + swf_r[STEP] * k;
      for (int l = 0; l < num_swf_th; l++) {
        swf_th_ = swf_th[MIN] + swf_th[STEP] * l;
        setInputPolar(swf_r_, swf_th_);
      }
    }

    FILE *fp_input = NULL;
    if ( ( fp_input = fopen("input_table.csv", "w") ) == NULL) {
      printf("Error: couldn't open input_table.csv\n");
      exit(EXIT_FAILURE);
    }
    fprintf(fp_input, "index,swf_r,swf_th\n");
    for (int i = 0; i < max(num_swf_r, num_swf_th); i++) {
      fprintf(fp_input, "%d", i);
      if (i < num_swf_r) {
        swf_r_ = swf_r[MIN] + swf_r[STEP] * i;
        fprintf(fp_input, ",%lf", swf_r_);
      } else {
        fprintf(fp_input, ",");
      }
      if (i < num_swf_th) {
        swf_th_ = swf_th[MIN] + swf_th[STEP] * i;
        fprintf(fp_input, ",%lf", swf_th_);
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

  max_val = max(max(val1, val2), max(val3, val4) );

  return max_val;
}

GridState Grid::roundState(State state_) {
  using GridSpace::MAX;
  using GridSpace::MIN;
  using GridSpace::STEP;

  int       state_id = -1;
  GridState gs;

  if (strcmp(coord.c_str(), "cartesian") == 0) {
    int icp_x_id = 0, icp_y_id = 0;
    int swf_x_id = 0, swf_y_id = 0;

    icp_x_id = round( ( state_.icp.x - icp_x[MIN] ) / icp_x[STEP]);
    icp_y_id = round( ( state_.icp.y - icp_y[MIN] ) / icp_y[STEP]);
    swf_x_id = round( ( state_.swf.x - swf_x[MIN] ) / swf_x[STEP]);
    swf_y_id = round( ( state_.swf.y - swf_y[MIN] ) / swf_y[STEP]);

    state_id = getStateIndexCartesian(icp_x_id, icp_y_id, swf_x_id, swf_y_id);
  }else if (strcmp(coord.c_str(), "polar") == 0) {
    int icp_r_id = 0, icp_th_id = 0;
    int swf_r_id = 0, swf_th_id = 0;

    icp_r_id  = round( ( state_.icp.r - icp_r[MIN] ) / icp_r[STEP]);
    icp_th_id = round( ( state_.icp.th - icp_th[MIN] ) / icp_th[STEP]);
    swf_r_id  = round( ( state_.swf.r - swf_r[MIN] ) / swf_r[STEP]);
    swf_th_id = round( ( state_.swf.th - swf_th[MIN] ) / swf_th[STEP]);

    state_id = getStateIndexPolar(icp_r_id, icp_th_id, swf_r_id, swf_th_id);
  }
  if(existState(state_id) )
    gs.id = state_id;
  else
    gs.id = -1;
  gs.state = getState(state_id);

  return gs;
}

int Grid::getStateIndex(State state_) {
  return roundState(state_).id;
}

void Grid::setStatePolar(float icp_r, float icp_th, float swf_r,
                         float swf_th) {
  State state_;
  state_.icp.setPolar(icp_r, icp_th);
  state_.swf.setPolar(swf_r, swf_th);

  state.push_back(state_);
  num_state++;
}

void Grid::setStateCartesian(float icp_x, float icp_y, float swf_x,
                             float swf_y) {
  State state_;
  state_.icp.setCartesian(icp_x, icp_y);
  state_.swf.setCartesian(swf_x, swf_y);

  state.push_back(state_);
  num_state++;
}

void Grid::setInputPolar(float swf_r, float swf_th) {
  Input input_;
  input_.swf.setPolar(swf_r, swf_th);

  input.push_back(input_);
  num_input++;
}

void Grid::setInputCartesian(float swf_x, float swf_y) {
  Input input_;
  input_.swf.setCartesian(swf_x, swf_y);

  input.push_back(input_);
  num_input++;
}

bool Grid::existState(int state_id) {
  bool is_exist = false;
  if (0 <= state_id && state_id < num_state)
    is_exist = true;

  return is_exist;
}

bool Grid::existState(State state_) {
  using GridSpace::MAX;
  using GridSpace::MIN;
  using GridSpace::STEP;

  bool flag = false;

  if (strcmp(coord.c_str(), "cartesian") == 0) {
    bool flag_icp_x = false, flag_icp_y = false;
    bool flag_swf_x = false, flag_swf_y = false;

    // icp_x
    if (state_.icp.x >= icp_x[MIN] - icp_x[STEP] / 2.0 &&
        state_.icp.x < icp_x[MAX] + icp_x[STEP] / 2.0) {
      flag_icp_x = true;
    }
    // icp_y
    if (state_.icp.y >= icp_y[MIN] - icp_y[STEP] / 2.0 &&
        state_.icp.y < icp_y[MAX] + icp_y[STEP] / 2.0) {
      flag_icp_y = true;
    }
    // swf_x
    if (state_.swf.x >= swf_x[MIN] - swf_x[STEP] / 2.0 &&
        state_.swf.x < swf_x[MAX] + swf_x[STEP] / 2.0) {
      flag_swf_x = true;
    }
    // swf_y
    if (state_.swf.y >= swf_y[MIN] - swf_y[STEP] / 2.0 &&
        state_.swf.y < swf_y[MAX] + swf_y[STEP] / 2.0) {
      flag_swf_y = true;
    }
    flag = flag_icp_x * flag_icp_y * flag_swf_x * flag_swf_y;
  }else if (strcmp(coord.c_str(), "polar") == 0) {
    bool flag_icp_r = false, flag_icp_th = false;
    bool flag_swf_r = false, flag_swf_th = false;

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
    // swf_r
    if (state_.swf.r >= swf_r[MIN] - swf_r[STEP] / 2.0 &&
        state_.swf.r < swf_r[MAX] + swf_r[STEP] / 2.0) {
      flag_swf_r = true;
    }
    // swf_th
    if (state_.swf.th >= swf_th[MIN] - swf_th[STEP] / 2.0 &&
        state_.swf.th < swf_th[MAX] + swf_th[STEP] / 2.0) {
      flag_swf_th = true;
    }
    flag = flag_icp_r * flag_icp_th * flag_swf_r * flag_swf_th;
  }
  return flag;
}

bool Grid::existInput(int input_id) {
  bool is_exist = false;
  if (0 <= input_id && input_id <= num_input)
    is_exist = true;

  return is_exist;
}

int Grid::getStateIndexCartesian(int icp_x_id, int icp_y_id, int swf_x_id,
                                 int swf_y_id) {
  int index = 0;
  index = num_swf_y * num_swf_x * num_icp_y * icp_x_id +
          num_swf_y * num_swf_x * icp_y_id + num_swf_y * swf_x_id +
          swf_y_id;
  return index;
}

int Grid::getStateIndexPolar(int icp_r_id, int icp_th_id, int swf_r_id,
                             int swf_th_id) {
  int index = 0;
  index = num_swf_th * num_swf_r * num_icp_th * icp_r_id +
          num_swf_th * num_swf_r * icp_th_id + num_swf_th * swf_r_id +
          swf_th_id;
  return index;
}

State Grid::getState(int index) {
  State state_;
  if (existState(index) ) {
    state_ = state[index];
  }else{
    state_.icp.setCartesian(-1, -1);
    state_.swf.setCartesian(-1, -1);
  }

  return state_;
}

Input Grid::getInput(int index) {
  Input input_;
  if (existInput(index) ) {
    input_ = input[index];
  }else{
    input_.swf.setCartesian(-1, -1);
  }

  return input_;
}

int Grid::getNumState() {
  return num_state;
}

int Grid::getNumInput() {
  return num_input;
}

int Grid::getNumGrid() {
  return num_state * num_input;
}

} // namespace Capt