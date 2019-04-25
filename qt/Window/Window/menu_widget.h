#ifndef WIDGET_MENU_H
#define WIDGET_MENU_H

#include <QtGui>
#include <QtWidgets>

class menuButton : public QPushButton
{
    Q_OBJECT

public:
    menuButton(const QString &text, QWidget *parent,
               int x,int y,
               int width, int height);
    ~menuButton();

    void setButtonColor(QColor windowColor, QColor textColor);
    void setButtonDefaultColor();
//    void setButtonFocusColor();

private slots:
    void setButtonPressed();
    void setButtonReleased();

private:
    QPushButton *button;

    QString buttonName;

    int buttonX, buttonY;
    int buttonWidth, buttonHeight;

    QColor buttonColor;
    QColor buttonTextColor;
    QColor buttonFocusColor;
    QColor buttonPressColor;
    QColor buttonPressTextColor;

//protected:
//    void focusInEvent(QFocusEvent* e);
//    void focusOutEvent(QFocusEvent* e);
};

class MenuWidget : public QWidget
{
    Q_OBJECT

public:
    MenuWidget(int width, int height);
    ~MenuWidget();

    void setWindowColor();
    void connectSignalSlot();

public slots:
    void buttonFocus();

private:
    int menuWidth, menuHeight;

    QColor windowColor;

    menuButton *buttonGraph;
    menuButton *buttonAnalysis;
    menuButton *buttonSearch;
    menuButton *buttonHelp;
};

#endif // WIDGET_MENU_H
