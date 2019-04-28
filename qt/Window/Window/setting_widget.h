#ifndef SETTING_WIDGET_H
#define SETTING_WIDGET_H

#include <QtGui>
#include <QtWidgets>
#include "setting_graph.h"
#include "section.h"

class SettingWidget : public QWidget
{
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
    SettingGraph *page[4];
};

#endif // SETTING_WIDGET_H
