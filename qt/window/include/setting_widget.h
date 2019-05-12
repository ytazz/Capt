#ifndef SETTING_WIDGET_H
#define SETTING_WIDGET_H

#include "base.h"
#include "section.h"
#include "setting_item.h"
#include <QtGui>
#include <QtWidgets>

class SettingWidget : public QWidget {
  Q_OBJECT

public:
  SettingWidget(int width, int height);
  ~SettingWidget();

  void setWindowColor();

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
  QWidget *page[NumberOfItem];
};

#endif // SETTING_WIDGET_H
