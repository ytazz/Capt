#ifndef SETTING_GRAPH_H
#define SETTING_GRAPH_H

#include <QtGui>
#include <QtWidgets>
#include "section.h"

class SettingGraph : public QWidget
{
    Q_OBJECT

public:
    explicit SettingGraph(QWidget *parent = nullptr);
    ~SettingGraph();

private:
    int windowWidth, windowHeight;
    QColor windowColor;

    void setWindowColor();

    void open();
    void createConnect();

    QVBoxLayout* pageLayout;
    Section* section[3];
};

#endif // SETTING_GRAPH_H
