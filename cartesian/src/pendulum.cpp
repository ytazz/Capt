#include "pendulum.h"

namespace Capt {

Pendulum::Pendulum(Model *model) {
  model->read(&g, "gravity");
  model->read(&h, "com_height");
  omega = sqrt(g / h);
}

Pendulum::~Pendulum(){
};

void Pendulum::setCom(const vec2_t com) {
  this->com = com;
}

void Pendulum::setCom(const vec3_t com) {
  this->com << com.x(), com.y();
}

void Pendulum::setComVel(const vec2_t com_vel) {
  this->com_vel = com_vel;
}

void Pendulum::setComVel(const vec3_t com_vel) {
  this->com_vel << com_vel.x(), com_vel.y();
}

void Pendulum::setIcp(const vec2_t icp) {
  this->icp = icp;
}

void Pendulum::setIcp(const vec3_t icp) {
  this->icp << icp.x(), icp.y();
}

void Pendulum::setCop(const vec2_t cop) {
  this->cop = cop;
}

void Pendulum::setCop(const vec3_t cop) {
  this->cop << cop.x(), cop.y();
}

vec2_t Pendulum::getCop(double dt) {
  return this->cop;
}

vec2_t Pendulum::getCom(double dt) {
  vec2_t com_;
  com_ = cop + ( com - cop ) * cosh(omega * dt) + com_vel * sinh(omega * dt) / omega;
  return com_;
}

vec2_t Pendulum::getComVel(double dt) {
  vec2_t com_vel_;
  com_vel_ = ( com - cop ) * omega * sinh(omega * dt) + com_vel * cosh(omega * dt);
  return com_vel_;
}

vec2_t Pendulum::getIcp(double dt) {
  vec2_t icp_;
  icp_ = ( icp - cop ) * exp(omega * dt) + cop;
  return icp_;
}

vec2_t Pendulum::getIcpVel(double dt) {
  vec2_t icp_vel_;
  icp_vel_ = omega * ( icp - cop ) * exp(omega * dt);
  return icp_vel_;
}

vec2_t Pendulum::invCop(vec2_t icp, vec2_t hat_icp, double dt){
  vec2_t cop_;
  cop_ = ( hat_icp - icp * exp(omega * dt) ) / ( 1 - exp(omega * dt) );
  return cop_;
}

vec2_t Pendulum::invIcp(vec2_t cop, vec2_t hat_icp, double dt){
  vec2_t icp_;
  icp_ = cop + ( hat_icp - cop ) / exp(omega * dt);
  return icp_;
}

} // namespace Capt