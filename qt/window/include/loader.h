#ifndef __XML_LOADER__
#define __XML_LOADER__

#include <expat.h>
#include <string>

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

protected:
  std::string name;
  XML_Parser parser;
  int depth;
};

#endif // __XML_LOADER__
