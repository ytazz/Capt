#ifndef WIDGET_MENU_H
#define WIDGET_MENU_H

#include <QtGui>
#include <QtWidgets>
#include "menu_button.h"

class MenuWidget : public QWidget
{
    Q_OBJECT

public:
    MenuWidget(int width, int height);
    ~MenuWidget();

    void setWindowColor();
    void connectSignalSlot();

    menuButton* getButton(int index);

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
    menuButton *buttonGraph;
    menuButton *buttonAnalysis;
    menuButton *buttonSearch;
    menuButton *buttonHelp;
};

#endif // WIDGET_MENU_H
