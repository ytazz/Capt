#ifndef __MODEL__
#define __MODEL__

#include "loader.h"
#include <iostream>

class Model : public Loader {

public:
  explicit Model(const std::string &name);
  ~Model();

  void callbackElement(const std::string &name, const bool is_start) override;
  void callbackAttribute(const std::string &name,
                         const std::string &value) override;
};

#endif // __MODEL__
