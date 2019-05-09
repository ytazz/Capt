#ifndef __READER_H__
#define __READER_H__

#include "vector.h"
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <iostream>
#include <string>

namespace CA {

class Reader {
public:
  Reader();
  ~Reader();

  void read(std::string file_name);
  void show();
  float getVal(std::string param_name);
  std::string getStr(std::string param_name);
  std::vector<Vector2> getVec(std::string param_name);

private:
  bool loadVal(std::string element_name, std::string attribute_name,
               float *val);
  bool loadStr(std::string element_name, std::string attribute_name,
               std::string *str);
  bool loadVec(std::string element_name, std::string attribute_name,
               std::vector<Vector2> *vec);
  bool check(std::string param_name);

  boost::property_tree::ptree pt;

  bool load_name;
  bool load_mass;
  bool load_com_height;
  bool load_step_time_min;
  bool load_foot_vel_max;
  bool load_foot_r, load_foot_l;

  std::string name;
  float mass;
  float com_height;
  float step_time_min;
  float foot_vel_max;
  std::vector<Vector2> points_r, points_l;
};

} // namespace CA

#endif // __READER_H__
