#ifndef __BASE_H__
#define __BASE_H__

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include <cmath>

#include <sbxml.h>

namespace Capt {

// Eigen typedefs
typedef float           real_t;
typedef Eigen::Vector2i vec2i_t;
typedef Eigen::Vector2f vec2_t;
typedef Eigen::Vector3f vec3_t;
typedef Eigen::Vector4f vec4_t;
typedef std::vector<vec2_t> arr2_t;
typedef std::vector<vec3_t> arr3_t;

//#define EPSILON 0.001
const real_t pi = 3.14159265358979f;

//
inline real_t wrapradian(real_t theta){
	while(theta >  pi) theta -= 2*pi;
	while(theta < -pi) theta += 2*pi;
	return theta;
}

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

// foot enum
enum Foot { FOOT_NONE, FOOT_R, FOOT_L };

// availability of local planning
// SUCCESS: planning succeeded successfully
// FAIL   : couldn't find a solution
// FINISH : reached the final step
enum Status { SUCCESS, MODIFIED, FAIL };

// trajectory decision variables
struct Step {
  Foot   s_suf;
  vec3_t pos;
  vec3_t cop;
  vec3_t icp;
  // vec3_t com;
};
struct Footstep : public std::vector<Step>{
  int cur;  //< current footstep index
};

struct EnhancedState {
  Footstep footstep;
  vec3_t   suf;
  vec3_t   swf;
  vec3_t   icp;
  //Capt::vec3_t   rfoot;
  //Capt::vec3_t   lfoot;
  Foot     s_suf;

  void updateFootstepIndex();  //< update current footstep index based on support foot position

  /*
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
  */
};

struct EnhancedInput {
  double       duration; // remained step duration
  Capt::vec3_t cop;
  Capt::vec3_t icp;
  //Capt::vec3_t suf;
  //Capt::vec3_t swf;
  Capt::vec3_t land;
  /*
  void operator=(const EnhancedInput &eInput) {
    this->duration = eInput.duration;
    this->cop      = eInput.cop;
    //this->icp      = eInput.icp;
    //this->suf      = eInput.suf;
    //this->swf      = eInput.swf;
    this->land     = eInput.land;
  }
  */
  /*
  void print(){
    printf("EnhancedInput\n");
    printf("  duration: %1.4lf\n", duration);
    printf("  cop     : %1.3lf, %1.3lf, %1.3lf\n", cop.x(), cop.y(), cop.z() );
    //printf("  icp     : %1.3lf, %1.3lf, %1.3lf\n", icp.x(), icp.y(), icp.z() );
    //printf("  suf     : %1.3lf, %1.3lf, %1.3lf\n", suf.x(), suf.y(), suf.z() );
    //printf("  swf     : %1.3lf, %1.3lf, %1.3lf\n", swf.x(), swf.y(), swf.z() );
    printf("  land    : %1.3lf, %1.3lf, %1.3lf\n", land.x(), land.y(), land.z() );
  }
  */
};

} // namespace Capt

#endif // __BASE_H__