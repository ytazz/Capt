#ifndef __GRAPH_H__
#define __GRAPH_H__

#include "loader.h"
#include <math.h>

namespace CA {

namespace Gr {
enum GraphElement { NOELEMENT, COORDINATE, UNIT, RADIUS, ANGLE, X, Y };

enum Coordinate { NOCOORD, POLAR, CARTESIAN };
} // namespace Gr

class Graph : public Loader {

public:
  explicit Graph(const std::string &name = NULL);
  ~Graph();

  void callbackElement(const std::string &name, const bool is_start) override;
  void callbackAttribute(const std::string &name,
                         const std::string &value) override;

  void print();

private:
  Gr::GraphElement element;
  Gr::Coordinate coordinate;

  float unit_length, unit_angle;
  float radius_min, radius_max, radius_step, radius_tick;
  float angle_min, angle_max, angle_step, angle_tick;

  const float pi;
};

} // namespace CA

#endif // __GRAPH_H__