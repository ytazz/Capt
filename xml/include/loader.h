#ifndef __XML_LOADER__
#define __XML_LOADER__

#include <expat.h>
#include <stdint.h>
#include <string>
#include <vector>

class Loader {
public:
  Loader(const std::string &name);
  ~Loader();

  static void start(void *data, const char *el, const char **attr);
  static void end(void *data, const char *el);

  void parse(void);

private:
  void start_element(const std::string &name);
  void get_attribute(const std::string &name, const std::string &value);
  void end_element(const std::string &name);

  std::string name;
  XML_Parser parser;
  uint32_t depth;

  std::vector<std::vector<std::string>> data;
  int num_element, num_attribute;
};

#endif // __XML_LOADER__
