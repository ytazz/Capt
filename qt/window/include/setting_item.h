#ifndef __SETTING_ITEM_H__
#define __SETTING_ITEM_H__

#include "base.h"
#include "graph.h"
#include "model.h"
#include "param.h"
#include "pendulum.h"
#include "section.h"
#include "vector.h"
#include <QtGui>
#include <QtWidgets>

namespace CA {

class SettingItem : public QWidget {
  Q_OBJECT

public:
  explicit SettingItem(QWidget *parent = nullptr, item_t item_name = 0);
  ~SettingItem();

  QLabel *label_file_name;
  QLabel *label_coordinate;
  QLabel *label_r_min, *label_r_max, *label_r_step, *label_r_tick;
  QLabel *label_t_min, *label_t_max, *label_t_step, *label_t_tick;

  QToolButton *button_file;

signals:
  // set point & line parameters
  // common
  void setPoint(Vector2 point, const char *color);
  void setPoints(std::vector<Vector2> point, const char *color);
  void setLine(std::vector<Vector2> vertex, const char *color);
  void setCircle(Vector2 center, float radius, const char *color);
  void setArc(Vector2 center, float radius, float start_angle, float end_angle,
              const char *color);
  // polar
  void setPolarGridRadius(float min, float max, float step, const char *color);
  void setPolarGridAngle(float min, float max, float step, const char *color);
  // cartesian
  void setCartesianGridX(float min, float max, float step, const char *color);
  void setCartesianGridY(float min, float max, float step, const char *color);

  // paint function
  void paint();
  // reset all parameters
  void reset();

public slots:
  void openFile();

private:
  int windowWidth, windowHeight;
  QColor windowColor;

  void setWindowColor();

  void createPage(item_t item_name);
  void createGraphPage();
  void createAnalysisPage();
  void createSearchPage();
  void createHelpPage();

  Section *section[4];

  void createConnection(item_t item_name);
};

} // namespace CA

#endif // __SETTING_ITEM_H__