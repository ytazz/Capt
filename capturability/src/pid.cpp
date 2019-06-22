#include "../include/pid.h"

PID::PID() {
  dt = 0.0;
  t = 0.0;

  k_p = 0.0;
  k_i = 0.0;
  k_d = 0.0;

  u_min = 0.0;
  u_max = 0.0;
}

PID::~PID() {}

void PID::setTimestep(float dt) { this->dt = dt; }

void PID::setGain(float k_p, float k_i, float k_d) {
  this->k_p = k_p;
  this->k_i = k_i;
  this->k_d = k_d;
}

void PID::setLimit(float u_min, float u_max) {
  this->u_min = u_min;
  this->u_max = u_max;
}

void PID::showParam() {
  printf("----- PID parameters -----\n");
  printf("  dt   \t:%1.4lf\n", dt);
  printf("  kp   \t:%5.3lf\n", k_p);
  printf("  ki   \t:%5.3lf\n", k_i);
  printf("  kd   \t:%5.3lf\n", k_d);
}

void PID::control(cnoid::BodyPtr ioBody, std::vector<float> qref) {
  int num_joint = ioBody->numJoints();

  if (num_joint != qref.size()) {
    printf("Error: number of controllable joint (%d) and number of input joint "
           "(%d) aren't match\n",
           num_joint, (int)qref.size());
    exit(EXIT_FAILURE);
  }

  if (qold.empty()) {
    qold = qref;
  }
  if (qrefold.empty()) {
    qrefold = qref;
  }

  if (integral.empty()) {
    for (int i = 0; i < num_joint; ++i) {
      integral.push_back(0.0);
    }
  }

  for (int i = 0; i < num_joint; ++i) {
    float q = ioBody->joint(i)->q();
    float dq = (q - qold[i]) / dt;
    float dqref = (qref[i] - qrefold[i]) / dt;
    integral[i] += (qref[i] - q) * dt;
    float u = k_p * (qref[i] - q) + k_i * integral[i] + k_d * (dqref - dq);
    qold[i] = q;
    qrefold[i] = q;

    // if (u > u_max) {
    //   u = u_max;
    // } else if (u < u_min) {
    //   u = u_min;
    // }

    ioBody->joint(i)->dq_target() = u;
  }

  t += dt;
}
