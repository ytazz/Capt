#ifndef __LOADER_CUH__
#define __LOADER_CUH__

#include "vector.cuh"
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
// #include <eigen3/Eigen/Core>
#include <expat.h>
#include <string.h>

namespace GPGPU {

typedef Eigen::Vector3f vec3_t;

class Loader {
public:
  __device__ Loader(const std::string &name);
  __device__ ~Loader();

  __device__ static void start(void *data, const char *el, const char **attr);
  __device__ static void end(void *data, const char *el);

  __device__ void parse();

  __device__ void start_element(const std::string &name);
  __device__ void end_element(const std::string &name);
  __device__ void get_attribute(const std::string &name,
                                const std::string &value);

  __device__ virtual void callbackElement(const std::string &name,
                                          const bool is_start) = 0;
  __device__ virtual void callbackAttribute(const std::string &name,
                                            const std::string &value) = 0;

  __device__ bool equalStr(const char *chr1, const char *chr2);
  __device__ bool equalStr(const std::string &str1, const char *chr2);
  __device__ bool equalStr(const char *chr1, const std::string &str2);
  __device__ bool equalStr(const std::string &str1, const std::string &str2);
  __device__ Vector2 convertStrToVec(const std::string &str);
  __device__ vec3_t convertStrToVec3(const std::string &str);

protected:
  std::string name;
  XML_Parser parser;
  int depth;
};

} // namespace GPGPU

#endif // __LOADER_CUH__