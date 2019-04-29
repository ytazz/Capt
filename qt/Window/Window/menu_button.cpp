#include "menu_button.h"

menuButton::menuButton(){}

menuButton::menuButton(const QString &text, QWidget *parent,
                       int x,int y):
    QToolButton(parent)
{
    // generate button
    buttonName = text;
    this->setText(buttonName);
    setDefaultIcon();

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
    buttonWidth  = parent->width();
    buttonHeight = parent->width();

    // arrangement
    this->setGeometry(QRect(QPoint(buttonX,buttonY),
                            QSize(buttonWidth, buttonHeight)));

    // color
    buttonColor         = QColor("#404244");
    buttonTextColor     = QColor("#BCBEBF");
    buttonFocusColor    = QColor("#595B5D");
    buttonPressColor    = QColor("#262829");
    setButtonDefaultColor();

    // signal & slot
    // connect(this, SIGNAL (pressed()), this, SLOT (setButtonPressed()));
}

menuButton::~menuButton(){}

void menuButton::setDefaultIcon()
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

    this->setIcon(icon);
    this->setIconSize(QSize(40, 40));
    this->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
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

    this->setIcon(icon);
    this->setIconSize(QSize(40, 40));
    this->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
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
    setDefaultIcon();
    //button->setDisabled(false);
}

void menuButton::setButtonColor(QColor windowColor)
{
    this->setStyleSheet("font-weight: bold");
    this->setAutoRaise(true);
    QPalette palette = this->palette();
    palette.setColor(QPalette::Button, windowColor);
    palette.setColor(QPalette::ButtonText, buttonTextColor);
    this->setPalette(palette);
    this->setAutoFillBackground(true);
}

void menuButton::setButtonDefaultColor()
{
    this->setButtonColor(buttonColor);
}

QString menuButton::getName()
{
    return buttonName;
}

int menuButton::getId()
{
    return buttonId;
}
