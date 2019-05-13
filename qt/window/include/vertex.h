#ifndef __VERTEX_H__
#define __VERTEX_H__

#include <QVector3D>

class Vertex {
public:
  Vertex();
  Vertex(const QVector3D &position) : m_position(position) {}
  static int positionOffset() { return offsetof(Vertex, m_position); }
  static int stride() { return sizeof(Vertex); }

private:
  QVector3D m_position;
};
#endif // __VERTEX_H__