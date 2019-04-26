#include "menu_button.h"

menuButton::menuButton(const QString &text, QWidget *parent,
                       int x,int y,
                       int width, int height)
{
    // generate button
    button = new QToolButton(parent);
    buttonName = text;
    button->setText(buttonName);
    setIcon();

    // set button index
    if(!QString::compare(buttonName, tr("Graph")   , Qt::CaseInsensitive))
        buttonId = 0;
    if(!QString::compare(buttonName, tr("Analysis"), Qt::CaseInsensitive))
        buttonId = 1;
    if(!QString::compare(buttonName, tr("Search")  , Qt::CaseInsensitive))
        buttonId = 2;
    if(!QString::compare(buttonName, tr("Help")    , Qt::CaseInsensitive))
        buttonId = 3;

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
    setButtonDefaultColor();

    // signal & slot
    // connect(button, SIGNAL (pressed()), this, SLOT (setButtonPressed()));
}

menuButton::~menuButton(){}

void menuButton::setIcon()
{
    QIcon icon;
    if(!QString::compare(buttonName, tr("Graph")   , Qt::CaseInsensitive))
        icon = QIcon(":/icons/graph.png");
    if(!QString::compare(buttonName, tr("Analysis"), Qt::CaseInsensitive))
        icon = QIcon(":/icons/analysis.png");
    if(!QString::compare(buttonName, tr("Search")  , Qt::CaseInsensitive))
        icon = QIcon(":/icons/search.png");
    if(!QString::compare(buttonName, tr("Help")    , Qt::CaseInsensitive))
        icon = QIcon(":/icons/help.png");

    button->setIcon(icon);
    button->setIconSize(QSize(40, 40));
    button->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
}

void menuButton::setPressIcon()
{
    QIcon icon;
    if(!QString::compare(buttonName, tr("Graph")   , Qt::CaseInsensitive))
        icon = QIcon(":/icons/graph_.png");
    if(!QString::compare(buttonName, tr("Analysis"), Qt::CaseInsensitive))
        icon = QIcon(":/icons/analysis_.png");
    if(!QString::compare(buttonName, tr("Search")  , Qt::CaseInsensitive))
        icon = QIcon(":/icons/search_.png");
    if(!QString::compare(buttonName, tr("Help")    , Qt::CaseInsensitive))
        icon = QIcon(":/icons/help_.png");

    button->setIcon(icon);
    button->setIconSize(QSize(40, 40));
    button->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
}

void menuButton::setButtonPressed()
{
    setButtonColor(buttonPressColor);
    setPressIcon();
    //button->setDisabled(true);
}

void menuButton::setButtonReleased()
{
    setButtonColor(buttonColor);
    setIcon();
    //button->setDisabled(false);
}

void menuButton::setButtonColor(QColor windowColor)
{
    button->setAutoRaise(true);
    QPalette palette = button->palette();
    palette.setColor(QPalette::Button, windowColor);
    palette.setColor(QPalette::ButtonText, buttonTextColor);
    button->setPalette(palette);
    button->setAutoFillBackground(true);
    button->setStyleSheet("font-weight: bold");
}

void menuButton::setButtonDefaultColor()
{
    setButtonColor(buttonColor);
}

QString menuButton::getName()
{
    return buttonName;
}

int menuButton::getId()
{
    return buttonId;
}
