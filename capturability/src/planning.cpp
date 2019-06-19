#include "planning.h"

namespace CA {

Planning::Planning(Model model) : polygon(), pendulum(model) {
  this->step_seq.clear();
  this->com_state.clear();
}

Planning::~Planning() {}

void Planning::setStepSeq(std::vector<StepSeq> step_seq) {
  this->step_seq.clear();
  this->step_seq = step_seq;
}

void Planning::setComState(vec2_t com, vec2_t com_vel) {
  this->com_state.clear();
  ComState com_state_;
  com_state_.pos = com;
  com_state_.vel = com_vel;
  this->com_state.push_back(com_state_);
}

void Planning::calcRef() {
  float point[2]; // landing
  float piece[2], piece_vel[2];
  float com_des[2], com_vel_des[2];
  float c, s;
  float Tc = sqrt(0.25 / 9.80665);
  const float a = 1, b = 0.1;

  for (size_t i = 0; i < step_seq.size() - 1; i++) {
    // step 3
    pendulum.setCop(step_seq[i].cop);
    pendulum.setCom(com_state[i].pos);
    pendulum.setComVel(com_state[i].vel);

    ComState com_state_;
    com_state_.pos = pendulum.getCom(step_seq[i].step_time);
    com_state_.vel = pendulum.getComVel(step_seq[i].step_time);
    this->com_state.push_back(com_state_);

    // step 5
    point[0] = step_seq[i].footstep.x();
    point[1] = step_seq[i].footstep.y();

    // step 6
    piece[0] = (step_seq[i + 1].footstep.x() - step_seq[i].footstep.x()) / 2.0;
    piece[1] = (step_seq[i + 1].footstep.y() - step_seq[i].footstep.y()) / 2.0;

    c = cosh(step_seq[i].step_time / Tc);
    s = sinh(step_seq[i].step_time / Tc);
    piece_vel[0] = piece[0] * (c + 1) / (Tc * s);
    piece_vel[1] = piece[1] * (c + 1) / (Tc * s);
    // printf("%f,%f\n", piece[0], piece[1]);

    // step 7
    com_des[0] = point[0] + piece[0];
    com_des[1] = point[1] + piece[1];
    com_vel_des[0] = piece_vel[0];
    com_vel_des[1] = piece_vel[1];

    // step 8
    float d = a * (c - 1) * (c - 1) + b * (s / Tc) * (s / Tc);
    point[0] = -a * (c - 1) *
                   (com_des[0] - c * com_state[i + 1].pos.x -
                    Tc * s * com_state[i + 1].vel.x) /
                   d -
               b * s *
                   (com_vel_des[0] - s * com_state[i].pos.x / Tc -
                    c * com_state[i].vel.x) /
                   (Tc * d);
    point[1] = -a * (c - 1) *
                   (com_des[1] - c * com_state[i + 1].pos.y -
                    Tc * s * com_state[i + 1].vel.y) /
                   d -
               b * s *
                   (com_vel_des[1] - s * com_state[i].pos.y / Tc -
                    c * com_state[i].vel.y) /
                   (Tc * d);

    step_seq[i + 1].footstep.x() = point[0];
    step_seq[i + 1].footstep.y() = point[1];
    step_seq[i + 1].cop.x = point[0];
    step_seq[i + 1].cop.y = point[1];
  }
}

float Planning::getPlanningTime() {
  float step_time_sum = 0.0;
  for (size_t i = 0; i < step_seq.size(); i++) {
    step_time_sum += step_seq[i].step_time;
  }
  return step_time_sum;
}

vec2_t Planning::getFootstep(int num_step) {
  vec2_t foot;
  foot.setCartesian(step_seq[num_step].footstep.x(),
                    step_seq[num_step].footstep.y());
  return foot;
}

int Planning::getNumStep(float time) {
  int num_step = 0;
  float step_time_sum = 0.0;

  if (time > getPlanningTime()) {
    printf("Error: time(%lf) > planning time(%lf)\n", time, getPlanningTime());
    exit(EXIT_FAILURE);
  } else {
    for (size_t i = 0; i < step_seq.size(); i++) {
      step_time_sum += step_seq[i].step_time;
      if (time <= step_time_sum)
        break;
      num_step++;
    }
  }
  return num_step;
}

vec2_t Planning::getCom(float time) { return com_state[getNumStep(time)].pos; }

void Planning::printStepSeq() {
  printf("---------------------------------------------------------\n");
  printf("|num\t|foot_x\t|foot_y\t|cop_x\t|cop_y\t|time\t|suft\t|\n");
  printf("---------------------------------------------------------\n");

  for (size_t i = 0; i < step_seq.size(); i++) {
    printf("|%d", (int)i);
    printf("\t|%1.3lf", step_seq[i].footstep.x());
    printf("\t|%1.3lf", step_seq[i].footstep.y());
    printf("\t|%1.3lf", step_seq[i].cop.x);
    printf("\t|%1.3lf", step_seq[i].cop.y);
    printf("\t|%1.3lf", step_seq[i].step_time);
    printf("\t|");
    switch (step_seq[i].e_suft) {
    case FOOT_R:
      printf("%s", "r_foot");
      break;
    case FOOT_L:
      printf("%s", "l_foot");
      break;
    default:
      break;
    }
    printf("\t|");
    printf("\n");
  }
  printf("---------------------------------------------------------\n");
}

} // namespace CA