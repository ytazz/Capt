#ifndef __XML_LOADER__
#define __XML_LOADER__

#include "vector.h"
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
// #include <eigen3/Eigen/Core>
#include <expat.h>
#include <string.h>

namespace CA {

typedef Eigen::Vector3f vec3_t;

class Loader {
public:
  Loader(const std::string &name);
  ~Loader();

  static void start(void *data, const char *el, const char **attr);
  static void end(void *data, const char *el);

  void parse();

  void start_element(const std::string &name);
  void end_element(const std::string &name);
  void get_attribute(const std::string &name, const std::string &value);

  virtual void callbackElement(const std::string &name,
                               const bool is_start) = 0;
  virtual void callbackAttribute(const std::string &name,
                                 const std::string &value) = 0;

  bool equalStr(const char *chr1, const char *chr2);
  bool equalStr(const std::string &str1, const char *chr2);
  bool equalStr(const char *chr1, const std::string &str2);
  bool equalStr(const std::string &str1, const std::string &str2);
  Vector2 convertStrToVec(const std::string &str);
  vec3_t convertStrToVec3(const std::string &str);

protected:
  std::string name;
  XML_Parser parser;
  int depth;
};

} // namespace CA

#endif // __XML_LOADER__