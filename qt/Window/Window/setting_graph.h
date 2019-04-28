#ifndef SETTING_GRAPH_H
#define SETTING_GRAPH_H

#include <QtGui>
#include <QtWidgets>

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
};

#endif // SETTING_GRAPH_H
