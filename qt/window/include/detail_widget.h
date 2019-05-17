#ifndef __DETAIL_WIDGET_H__
#define __DETAIL_WIDGET_H__

#include <QtGui>
#include <QtWidgets>
#include <base.h>

namespace CA {

struct Legend {
  char *name;
  char *type;
  char *color;
};

class DetailWidget : public QWidget {
  Q_OBJECT

public:
  explicit DetailWidget(QWidget *parent = nullptr, int width = 0,
                        int height = 0);
  ~DetailWidget();

  void setWindowColor(QColor color);

  void setLegend(char *name, char *mark, char *color);
  void updateLayout();

private:
  int width, height;
  int width_name, width_mark, height_name, height_mark;

  QGridLayout *layout;
  std::vector<Legend> legend;
};

} // namespace CA

#endif // __DETAIL_WIDGET_H__