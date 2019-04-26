#ifndef WIDGET_MENU_H
#define WIDGET_MENU_H

#include <QtGui>
#include <QtWidgets>

class menuButton : public QToolButton
{
    Q_OBJECT

public:
    menuButton(const QString &text, QWidget *parent,
               int x,int y,
               int width, int height);
    ~menuButton();

    void setButtonColor(QColor windowColor);
    void setButtonDefaultColor();

    void setIcon();
    void setPressIcon();

public:
    QString getName();
    int getId();

public slots:
    void setButtonPressed();
    void setButtonReleased();

private:
    QToolButton *button;

    QString buttonName;

    int buttonId;

    int buttonX, buttonY;
    int buttonWidth, buttonHeight;

    QColor buttonColor;
    QColor buttonTextColor;
    QColor buttonFocusColor;
    QColor buttonPressColor;
};

class MenuWidget : public QWidget
{
    Q_OBJECT

public:
    MenuWidget(int width, int height);
    ~MenuWidget();

    void setWindowColor();
    void connectSignalSlot();

    menuButton* getButton(int index);

public slots:
    void offOthers(menuButton *button);

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
