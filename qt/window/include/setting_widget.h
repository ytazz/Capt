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
  explicit SettingWidget(QWidget *parent = nullptr, int width = 0,
                         int height = 0);
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
