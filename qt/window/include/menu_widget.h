#ifndef WIDGET_MENU_H
#define WIDGET_MENU_H

#include "menu_button.h"
#include "section.h"
#include <QtGui>
#include <QtWidgets>

class MenuWidget : public QWidget {
  Q_OBJECT

public:
  explicit MenuWidget(QWidget *parent = nullptr, int width = 0, int height = 0);
  ~MenuWidget();

  void setWindowColor();
  void connectSignalSlot();

  menuButton *getButton(int index);
  menuButton *buttonGraph;
  menuButton *buttonAnalysis;
  menuButton *buttonSearch;
  menuButton *buttonHelp;

  void offOthers(menuButton *button);

public slots:
  void pressedGraph();
  void pressedAnalysis();
  void pressedSearch();
  void pressedHelp();

private:
  int menuWidth, menuHeight;

  QColor windowColor;

  int numButton;
};

#endif // WIDGET_MENU_H
