#ifndef __SETTING_WIDGET_H__
#define __SETTING_WIDGET_H__

#include "base.h"
#include "section.h"
#include "setting_item.h"
#include <QtGui>
#include <QtWidgets>

namespace CA {

class SettingWidget : public QWidget {
  Q_OBJECT

public:
  explicit SettingWidget(QWidget *parent = nullptr, int width = 0,
                         int height = 0);
  ~SettingWidget();

  void setWindowColor();

  SettingItem *page[NumberOfItem];

public slots:
  void pageGraph();
  void pageAnalysis();
  void pageSearch();
  void pageHelp();

private:
  int windowWidth, windowHeight;
  QColor windowColor;

  QStackedWidget *stackedWidget;

  int numPage;
};

} // namespace CA

#endif // __SETTING_WIDGET_H__
