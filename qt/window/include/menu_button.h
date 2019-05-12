#ifndef MENU_BUTTON_H
#define MENU_BUTTON_H

#include <QtGui>
#include <QtWidgets>
#include <base.h>

class menuButton : public QToolButton
{
    Q_OBJECT

public:
    explicit menuButton();
    explicit menuButton(QWidget *parent = nullptr, item_t item_name = 0,
                        int x=0, int y=0);
    ~menuButton();

    void setButtonColor(QColor windowColor);
    void setButtonDefaultColor();

    void setDefaultIcon();
    void setPressIcon();

public:
    QString getName(item_t item_name);
    int getIndex();

public slots:
    void setButtonPressed();
    void setButtonReleased();

private:
    item_t buttonName;

    int buttonX, buttonY;
    int buttonWidth, buttonHeight;

    QColor buttonColor;
    QColor buttonTextColor;
    QColor buttonFocusColor;
    QColor buttonPressColor;
};

#endif // MENU_BUTTON_H
