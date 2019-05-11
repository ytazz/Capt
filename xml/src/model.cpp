#include "model.h"

Model::Model(const std::string &name) : Loader(name) {
  // initialize
}

Model::~Model() {}

void Model::callbackElement(const std::string &name, const bool is_start) {
  std::cout << "find" << std::endl;
}

void Model::callbackAttribute(const std::string &name,
                              const std::string &value) {
  std::cout << "find" << std::endl;
}
