#include "loader.h"
#include <assert.h>
#include <fstream>
#include <iostream>

Loader::Loader(const std::string &name)
    : name(name), parser(XML_ParserCreate(NULL)), depth(0) {
  if (!parser) {
    std::cout << "Couldn't allocate memory for parser" << std::endl;
    exit(-1);
  }

  num_element = 0;
  num_attribute = 0;

  XML_SetUserData(parser, this);
  XML_SetElementHandler(parser, start, end);
}

Loader::~Loader() {}

void Loader::parse(void) {
  std::ifstream l_file(name.c_str());
  if (!l_file) {
    std::cout << "ERROR : Could not open file \"" << name << "\"" << std::endl;
    exit(-1);
  }
  const uint32_t l_size = 10;
  char l_buf[l_size];
  bool l_end = false;
  while (!(l_end = l_file.eof())) {
    l_file.read(l_buf, l_size);
    if (!XML_Parse(parser, l_buf, l_file.gcount(), l_end)) {
      std::cout << "ERROR : Parse error at line "
                << XML_GetCurrentLineNumber(parser) << " :"
                << XML_ErrorString(XML_GetErrorCode(parser)) << std::endl;
      exit(-1);
    }
  }
  l_file.close();
  std::cout << "PARSE SUCCESSFULL" << std::endl;
}

void Loader::start(void *data, const char *el, const char **attr) {
  Loader *l_parser = static_cast<Loader *>(data);
  assert(l_parser);
  l_parser->start_element(el);
  for (uint32_t i = 0; attr[i]; i += 2) {
    l_parser->get_attribute(attr[i], attr[i + 1]);
  }
}

void Loader::end(void *data, const char *el) {
  Loader *l_parser = static_cast<Loader *>(data);
  assert(l_parser);
  l_parser->end_element(el);
}

void Loader::start_element(const std::string &name) {
  for (uint32_t l_index = 0; l_index < depth; ++l_index) {
    std::cout << "  ";
  }
  // std::cout << "Starting element \"" << name << "\"" << std::endl;
  std::cout << "Starting element \"" << name << "\"";
  num_element++;
  ++depth;
  std::cout << "\t" << num_element << std::endl;
}

void Loader::end_element(const std::string &name) {
  --depth;
  for (uint32_t l_index = 0; l_index < depth; ++l_index) {
    std::cout << "  ";
  }
  std::cout << "Ending element \"" << name << "\"" << std::endl;
}

void Loader::get_attribute(const std::string &name, const std::string &value) {
  for (uint32_t l_index = 0; l_index < depth; ++l_index) {
    std::cout << "  ";
  }
  std::cout << "attribute[\"" << name << "\"] = \"" << value << "\""
            << std::endl;
  num_attribute++;
}
