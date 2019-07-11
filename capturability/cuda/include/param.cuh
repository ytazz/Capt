#ifndef __PARAM_CUH__
#define __PARAM_CUH__

#include "loader.cuh"
#include <math.h>

namespace GPGPU {

namespace Pa {
enum ParamElement {
  NOELEMENT,
  COORDINATE,
  UNIT,
  ICP,
  SWING,
};

enum Coordinate { NOCOORD, POLAR, CARTESIAN };

enum Axis { NOAXIS, RADIUS, ANGLE, X, Y };
} // namespace Pa

class Param : public Loader {

public:
  __device__ explicit Param(const std::string &name = "");
  __device__ ~Param();

  __device__ void callbackElement(const std::string &name,
                                  const bool is_start) override;
  __device__ void callbackAttribute(const std::string &name,
                                    const std::string &value) override;

  __device__ float getVal(const char *element_name, const char *attribute_name);
  __device__ std::string getStr(const char *element_name,
                                const char *attribute_name);

private:
  Pa::ParamElement element;
  Pa::Coordinate coordinate;
  Pa::Axis axis;

  // unit
  float unit_length, unit_angle;
  // polar
  float icp_r_min, icp_r_max, icp_r_step;
  float icp_th_min, icp_th_max, icp_th_step;
  float swft_r_min, swft_r_max, swft_r_step;
  float swft_th_min, swft_th_max, swft_th_step;
  // cartesian
  float icp_x_min, icp_x_max, icp_x_step;
  float icp_y_min, icp_y_max, icp_y_step;
  float swft_x_min, swft_x_max, swft_x_step;
  float swft_y_min, swft_y_max, swft_y_step;

  float pi;
};

} // namespace GPGPU

#endif // __PARAM_CUH__