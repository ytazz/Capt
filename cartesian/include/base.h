#ifndef __BASE_H__
#define __BASE_H__

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

namespace Capt {

#define EPSILON 0.001

// Eigen typedefs
typedef Eigen::Vector2f vec2_t;
typedef Eigen::Vector2i vec2i_t;
typedef Eigen::Vector3f vec3_t;
typedef std::vector<vec2_t> arr2_t;
typedef std::vector<vec3_t> arr3_t;

// linear algebra
float dot(vec2_t v1, vec2_t v2);
float cross(vec2_t v1, vec2_t v2);
vec2_t normal(vec2_t v);

// coordinate transformation with left and right of support2 foot
// [ x, y, z ] -> mirror -> [ x, -y, z ]
vec2_t mirror(vec2_t v);
vec3_t mirror(vec3_t v);

// round to the nearest integer
int round(float val);

// vector dimension transformation between R^2 and R^3
// R^3 to R^2 : [ x, y, z ] -> [ x, y ]
// R^2 to R^3 : [ x, y ] -> [ x, y, 0 ]
vec2_t vec3Tovec2(vec3_t vec3);
vec3_t vec2Tovec3(vec2_t vec2);

// 不要
enum OccupancyType {
  NONE,
  EMPTY,
  EXIST,
  CLOSED,
  GOAL
};

// foot enum
enum Foot { FOOT_NONE, FOOT_R, FOOT_L };

// availability of local planning
// SUCCESS: planning succeeded successfully
// FAIL   : couldn't find a solution
// FINISH : reached the final step
enum Status { SUCCESS, FAIL, FINISH };

// trajectory decision variables
struct Step {
  Foot   suf;
  vec3_t pos;
  vec3_t cop;
  vec3_t icp;
  // vec3_t com;
};

typedef std::vector<Step> Footstep;

// 不要 stepと同じでは?
struct Sequence {
  Foot   suf;
  vec3_t pos;
  vec3_t icp;
  vec3_t cop;

  void substitute(Foot suf, vec2_t pos, vec2_t icp, vec2_t cop){
    this->suf     = suf;
    this->pos.x() = pos.x();
    this->pos.y() = pos.y();
    this->pos.z() = 0.0f;
    this->icp.x() = icp.x();
    this->icp.y() = icp.y();
    this->icp.z() = 0.0f;
    this->cop.x() = cop.x();
    this->cop.y() = cop.y();
    this->cop.z() = 0.0f;
  }
};

struct EnhancedState {
  Capt::Footstep footstep;
  Capt::vec3_t   icp;
  Capt::vec3_t   rfoot;
  Capt::vec3_t   lfoot;
  Capt::Foot     s_suf;

  void operator=(const EnhancedState &eState) {
    this->footstep = eState.footstep;
    this->icp      = eState.icp;
    this->rfoot    = eState.rfoot;
    this->lfoot    = eState.lfoot;
    this->s_suf    = eState.s_suf;
  }

  void print(){
    printf("EnhancedState\n");
    if(s_suf == Foot::FOOT_R) {
      printf("  support: Right\n");
    }else{
      printf("  support: Left\n");
    }
    printf("  icp    : %1.3lf, %1.3lf, %1.3lf\n", icp.x(), icp.y(), icp.z() );
    printf("  rfoot  : %1.3lf, %1.3lf, %1.3lf\n", rfoot.x(), rfoot.y(), rfoot.z() );
    printf("  lfoot  : %1.3lf, %1.3lf, %1.3lf\n", lfoot.x(), lfoot.y(), lfoot.z() );
  }
};

struct EnhancedInput {
  double       duration; // remained step duration
  Capt::vec3_t cop;
  Capt::vec3_t icp;
  Capt::vec3_t suf;
  Capt::vec3_t swf;
  Capt::vec3_t land;

  void operator=(const EnhancedInput &eInput) {
    this->duration = eInput.duration;
    this->cop      = eInput.cop;
    this->icp      = eInput.icp;
    this->suf      = eInput.suf;
    this->swf      = eInput.swf;
    this->land     = eInput.land;
  }

  void print(){
    printf("EnhancedInput\n");
    printf("  duration: %1.4lf\n", duration);
    printf("  cop     : %1.3lf, %1.3lf, %1.3lf\n", cop.x(), cop.y(), cop.z() );
    printf("  icp     : %1.3lf, %1.3lf, %1.3lf\n", icp.x(), icp.y(), icp.z() );
    printf("  suf     : %1.3lf, %1.3lf, %1.3lf\n", suf.x(), suf.y(), suf.z() );
    printf("  swf     : %1.3lf, %1.3lf, %1.3lf\n", swf.x(), swf.y(), swf.z() );
    printf("  land    : %1.3lf, %1.3lf, %1.3lf\n", land.x(), land.y(), land.z() );
  }
};

} // namespace Capt

#endif // __BASE_H__