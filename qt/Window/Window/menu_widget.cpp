#include "menu_widget.h"

menuButton::menuButton(const QString &text, QWidget *parent,
                       int x,int y,
                       int width, int height)
{
    // generate button
    button = new QPushButton(text, parent);
    //buttonName = new QString(text);
    buttonName = text;

    // position
    buttonX = x;
    buttonY = y;

    // size
    buttonWidth  = width;
    buttonHeight = height;

    // arrangement
    button->setGeometry(QRect(QPoint(buttonX,buttonY),
                              QSize(buttonWidth, buttonHeight)));

    // color
    buttonColor         = QColor("#404244");
    buttonTextColor     = QColor("#BCBEBF");
    buttonFocusColor    = QColor("#595B5D");
    buttonPressColor    = QColor("#262829");
    QColor color = QColor("#98A9EE");
    if(!QString::compare(buttonName, tr("Graph")   , Qt::CaseInsensitive))
        color = QColor("#7FC242"); // green
    if(!QString::compare(buttonName, tr("Analysis"), Qt::CaseInsensitive))
        color = QColor("#98A9EE"); // blue
    if(!QString::compare(buttonName, tr("Search")  , Qt::CaseInsensitive))
        color = QColor("#BA6000"); // red
    if(!QString::compare(buttonName, tr("Help")    , Qt::CaseInsensitive))
        color = QColor("#F3BD04"); // orange
    buttonPressTextColor= color;
    setButtonDefaultColor();

    // signal & slot
    connect(button, SIGNAL (pressed()), this, SLOT (setButtonPressed()));
    connect(button, SIGNAL (released()), this, SLOT (setButtonReleased()));

    setButtonPressed();
}

menuButton::~menuButton(){}

void menuButton::setButtonPressed()
{
    setButtonColor(buttonPressColor,buttonPressTextColor);
    button->setDisabled(true);
    printf("a\n");
}

void menuButton::setButtonReleased()
{
    setButtonColor(buttonColor,buttonTextColor);
    button->setDisabled(false);
    printf("b\n");
}

//void menuButton::focusInEvent(QFocusEvent* e)
//{
//    setButtonFocusColor();
//    QPushButton::focusInEvent(e);
//}

//void menuButton::focusOutEvent(QFocusEvent* e)
//{
//    setButtonDefaultColor();
//    QPushButton::focusOutEvent(e);
//}

void menuButton::setButtonColor(QColor windowColor, QColor textColor)
{
    button->setFlat(true);
    QPalette palette = button->palette();
    palette.setColor(QPalette::Button, windowColor);
    palette.setColor(QPalette::ButtonText, textColor);
    button->setPalette(palette);
    button->setAutoFillBackground(true);
    button->setStyleSheet("font-weight: bold");
}

void menuButton::setButtonDefaultColor()
{
    setButtonColor(buttonColor, buttonTextColor);
}

//void menuButton::setButtonFocusColor()
//{
//    setButtonColor(buttonFocusColor, buttonTextColor);
//}

MenuWidget::MenuWidget(int width, int height)
{
    // size
    menuWidth  = width;
    menuHeight = height;
    setFixedSize(menuWidth, menuHeight);

    // color
    windowColor         = QColor("#404244");
    setWindowColor();

    // generate buttons
    buttonGraph    = new menuButton(tr("Graph")   ,this,0,0*menuWidth,menuWidth,menuWidth);
    buttonAnalysis = new menuButton(tr("Analysis"),this,0,1*menuWidth,menuWidth,menuWidth);
    buttonSearch   = new menuButton(tr("Search")  ,this,0,2*menuWidth,menuWidth,menuWidth);
    buttonHelp     = new menuButton(tr("Help")    ,this,0,3*menuWidth,menuWidth,menuWidth);
}

MenuWidget::~MenuWidget(){}

void MenuWidget::setWindowColor()
{
    QLabel *plabel=new QLabel(this);
    plabel->setFixedSize(menuWidth, menuHeight);
    QPalette palette = plabel->palette();
    palette.setColor(plabel->backgroundRole(), windowColor);
    palette.setColor(plabel->foregroundRole(), windowColor);
    plabel->setPalette(palette);
    plabel->setAutoFillBackground(true);
}

void MenuWidget::connectSignalSlot()
{
    //connect(this, buttonGraph->focusInEvent(),this,buttonFocus());
}

void MenuWidget::buttonFocus()
{
    printf("hello");
}
