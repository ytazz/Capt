#ifndef __DETAIL_WIDGET_H__
#define __DETAIL_WIDGET_H__

#include <QtGui>
#include <QtWidgets>
#include <base.h>

namespace CA {

class DetailWidget : public QWidget {
  Q_OBJECT

public:
  explicit DetailWidget(QWidget *parent = nullptr, int width = 0,
                        int height = 0);
  ~DetailWidget();

  void setWindowColor(QColor color);
  void layout();

private:
  int width, height;
};

} // namespace CA

#endif // __DETAIL_WIDGET_H__