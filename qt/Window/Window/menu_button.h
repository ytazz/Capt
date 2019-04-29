#ifndef MENU_BUTTON_H
#define MENU_BUTTON_H

#include <QtGui>
#include <QtWidgets>

class menuButton : public QToolButton
{
    Q_OBJECT

public:
    explicit menuButton();
    explicit menuButton(const QString &text, QWidget *parent = nullptr,
                        int x=0, int y=0);
    ~menuButton();

    void setButtonColor(QColor windowColor);
    void setButtonDefaultColor();

    void setDefaultIcon();
    void setPressIcon();

public:
    QString getName();
    int getId();

public slots:
    void setButtonPressed();
    void setButtonReleased();

private:
    // QToolButton *button;

    QString buttonName;

    int buttonId;

    int buttonX, buttonY;
    int buttonWidth, buttonHeight;

    QColor buttonColor;
    QColor buttonTextColor;
    QColor buttonFocusColor;
    QColor buttonPressColor;
};

#endif // MENU_BUTTON_H
