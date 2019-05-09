#include "reader.h"

namespace CA {

Reader::Reader() {
  load_mass = false;
  load_com_height = false;
  load_step_time_min = false;
  load_foot_vel_max = false;
  load_foot_r = false;
  load_foot_l = false;

  name = "";
  mass = 0.0;
  com_height = 0.0;
  step_time_min = 0.0;
  foot_vel_max = 0.0;
  points_r.clear();
  points_l.clear();
}

Reader::~Reader() {}

bool Reader::loadVal(std::string element_name, std::string attribute_name,
                     float *val) {
  bool flag = false;
  std::string param_name = element_name + ".<xmlattr>." + attribute_name;

  if (boost::optional<std::string> s =
          pt.get_optional<std::string>(param_name)) {
    printf("%s = %s\n", attribute_name.c_str(), s.get().c_str());
    *val = std::stof(s.get());
    flag = true;
  } else {
    printf("Error: the element(%s) or attribute(%s) doesn't exist !\n",
           element_name.c_str(), attribute_name.c_str());
    exit(EXIT_FAILURE);
  }

  return flag;
}

bool Reader::loadStr(std::string element_name, std::string attribute_name,
                     std::string *str) {
  bool flag = false;
  std::string param_name = element_name + ".<xmlattr>." + attribute_name;

  if (boost::optional<std::string> s =
          pt.get_optional<std::string>(param_name)) {
    printf("%s = %s\n", attribute_name.c_str(), s.get().c_str());
    *str = s.get();
    flag = true;
  } else {
    printf("Error: the　element(%s) or attribute(%s) doesn't exist !\n",
           element_name.c_str(), attribute_name.c_str());
    exit(EXIT_FAILURE);
  }

  return flag;
}

bool Reader::loadVec(std::string element_name, std::string attribute_name,
                     std::vector<Vector2> *vec) {
  bool flag = false;
  std::string param_name = element_name;

  int count = 0;
  BOOST_FOREACH (const boost::property_tree::ptree::value_type &child,
                 pt.get_child(param_name)) {
    flag = true;
    // const int value = boost::lexical_cast<int>(child.second.data());
    std::cout << child.second.data() << " , count = " << count << std::endl;
    count += 1;
  }

  // if (boost::optional<std::string> s =
  //         pt.get_optional<std::string>(param_name)) {
  //   printf("%s = %s\n", attribute_name.c_str(), s.get().c_str());
  //   // *val = std::stof(s.get());
  //   flag = true;
  // } else {
  //   printf("Error: the element(%s) or attribute(%s) doesn't exist !\n",
  //          element_name.c_str(), attribute_name.c_str());
  //   exit(EXIT_FAILURE);
  // }

  return flag;
}

void Reader::read(std::string file_name) {
  boost::property_tree::read_xml(file_name, pt);

  load_name = loadStr("robot", "name", &name);
  load_mass = loadVal("robot.physics", "mass", &mass);
  load_com_height = loadVal("robot.physics", "com_height", &com_height);
  load_step_time_min =
      loadVal("robot.physics", "step_time_min", &step_time_min);
  load_foot_vel_max = loadVal("robot.physics", "foot_vel_max", &foot_vel_max);
  load_foot_r = loadVec("robot.link.shape", "point", &points_r);
  // load_foot_l = loadVec("robot.link.shape.polygon", "point", &points_l);
}

} // namespace CA
