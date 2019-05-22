#include "capturability.h"

namespace CA {

Capturability::Capturability() {
  capture_state.clear();

  for (int i = 0; i < grid.sizeState(); i++) {
    CaptureState cs;
    capture_state.push_back(cs);
  }

  capturable.resize(5);
}

Capturability::~Capturability() {}

void Capturability::setState(State state) { this->state = state; }

void Capturability::setInput(Input input) { this->input = input; }

Vector2 Capturability::getCop() {
  Vector2 cop_ = {state.dcm.x, state.dcm.y};
  Vector2 cop;

  Polygon polygon;
  std::vector<Vector2> foot_su;

  foot_su = polygon.getFootVertex({0.0, 0.0}, "right");
  polygon.setVertex(foot_su);
  foot_su = polygon.getConvexHull();
  bool in_suFt = polygon.inPolygon(cop_, foot_su);

  if (in_suFt) {
    cop = {0.0, 0.0};
    // cop = cop_;
  } else {
    std::vector<Vector2> v;
    v.push_back({-0.04, -0.03});
    v.push_back({-0.03, -0.04});
    v.push_back({+0.01, -0.03});
    v.push_back({+0.04, -0.04});
    v.push_back({+0.06, -0.04});
    v.push_back({+0.09, -0.02});
    v.push_back({+0.09, +0.02});
    v.push_back({+0.06, +0.03});
    v.push_back({+0.01, +0.02});
    v.push_back({-0.04, +0.02});

    double rad;
    if (atan2(cop_.y, cop_.x) < 0) {
      rad = atan2(cop_.y, cop_.x) + 2 * M_PI;
    } else {
      rad = atan2(cop_.y, cop_.x);
    }

    int l = 0;
    if (0 <= rad && rad < 0.219) {
      l = 6;
    } else if (rad <= 0.464) {
      l = 7;
    } else if (rad <= 1.107) {
      l = 8;
    } else if (rad <= 2.678) {
      l = 9;
    } else if (rad <= 3.875) {
      l = 0;
    } else if (rad <= 4.069) {
      l = 1;
    } else if (rad <= 5.034) {
      l = 2;
    } else if (rad <= 5.498) {
      l = 3;
    } else if (rad <= 5.695) {
      l = 4;
    } else if (rad <= 6.065) {
      l = 5;
    } else {
      l = 6;
    }

    double a, b, c, d;
    if (l == 0 || l == 6) {
      c = cop_.y / cop_.x;
      d = 0;
      cop.x = v[l].x;
      cop.y = c * cop.x;
    } else {
      a = (v[l].y - v[l - 1].y) / (v[l].x - v[l - 1].x);
      b = -a * v[l - 1].x + v[l - 1].y;
      if (fabs(cop_.x) <= 0.001) {
        cop.x = 0.0;
        cop.y = b;
      } else {
        c = cop_.y / cop_.x;
        d = 0;
        cop.x = (d - b) / (a - c);
        cop.y = (a * d - b * c) / (a - c);
      }
    }
  }

  return cop;
}

Vector2 Capturability::getDcm(float dt) {
  Pendulum pendulum;

  pendulum.setDcm(state.dcm);
  Vector2 cop;
  cop = getCop();
  pendulum.setVrp({cop.x, cop.y, input.vrp.z});

  Vector3 dcm3;
  Vector2 dcm2;

  dcm3 = pendulum.getDcm(dt);
  dcm2.x = dcm3.x;
  dcm2.y = dcm3.y;

  return dcm2;
}

std::vector<Vector2> Capturability::getSupportRegion() {
  Polygon polygon;
  std::vector<Vector2> foot_su, foot_sw;

  foot_su = polygon.getFootVertex({0.0, 0.0}, "right");
  foot_sw = polygon.getFootVertex({input.foot.x, input.foot.y}, "left");
  polygon.setVertex(foot_su);
  polygon.setVertex(foot_sw);

  return polygon.getConvexHull();
}

bool Capturability::inPolygon(Vector2 point_,
                              std::vector<Vector2> support_region_) {
  Polygon polygon;

  return polygon.inPolygon(point_, support_region_);
}

GridState Capturability::getNextState(float dt) {
  State state;
  Vector2 dcm;

  state.foot = {-input.foot.x, input.foot.y, 0.0};
  state.dcm = {-input.foot.x + dcm.x, input.foot.y - dcm.y, 0.3};

  GridState grid_state = grid.mappingState(state);

  return grid_state;
}

void Capturability::setCapturable(bool is_exist) {
  capturable[0].push_back(is_exist);
}

bool Capturability::getCapturable(int state_id_) {
  return capturable[0][state_id_];
}

void Capturability::readDatabase(std::string file_path) {
  FILE *fp;
  int i = 0;

  if ((fp = fopen(file_path.c_str(), "r")) == NULL) {
    printf("Error: cannnot find the file \"%s\"\n", file_path.c_str());
    exit(EXIT_FAILURE);
  } else {
    double buf[9];
    while (fscanf(fp, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", &buf[0], &buf[1],
                  &buf[2], &buf[3], &buf[4], &buf[5], &buf[6], &buf[7],
                  &buf[8]) != EOF) {
      GridState grid_state;
      GridInput grid_input;
      GridState next_grid_state;

      grid_state.id = buf[0];
      grid_state.state = grid.getState(grid_state.id);

      grid_input.id = buf[5];
      grid_input.input = grid.getInput(grid_input.id);

      next_grid_state.id = buf[8];
      next_grid_state.state = grid.getState(next_grid_state.id);

      setCaptureRegion(grid_state, grid_input, next_grid_state, 1);
      i++;
    }
    fclose(fp);
  }

  printf("Read success! (%d datas)\n", i);
}

void Capturability::setCaptureRegion(GridState grid_state_,
                                     GridInput grid_input_,
                                     GridState next_grid_state_,
                                     int n_step_capturable_) {
  capture_state[grid_state_.id].grid_state = grid_state_;
  capture_state[grid_state_.id].grid_input.push_back(grid_input_);
  capture_state[grid_state_.id].next_grid_state.push_back(next_grid_state_);
  capture_state[grid_state_.id].n_step_capturable.push_back(n_step_capturable_);
}

CaptureState Capturability::getCaptureRegion(int grid_state_id_,
                                             int n_step_capturable_) {
  if (grid_state_id_ > grid.sizeState()) {
    printf("Error: id %d is out of \"state\" datas (> %d)\n", grid_state_id_,
           grid.sizeState());
    exit(EXIT_FAILURE);
  }

  return capture_state[grid_state_id_];
}

} // namespace CA
