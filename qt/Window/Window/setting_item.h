#ifndef SETTING_GRAPH_H
#define SETTING_ITEM_H

#include <QtGui>
#include <QtWidgets>
#include "section.h"
#include "base.h"

class SettingItem : public QWidget
{
    Q_OBJECT

public:
    explicit SettingItem(QWidget *parent = nullptr, item_t item_name = 0);
    ~SettingItem();

private:
    int windowWidth, windowHeight;
    QColor windowColor;

    void setWindowColor();

    void createPage(item_t item_name);
    void createGraphPage();
    void createAnalysisPage();
    void createSearchPage();
    void createHelpPage();

    void open();
    void createConnect();
};

#endif // SETTING_ITEM_H
