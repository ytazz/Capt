#include "menu_button.h"

menuButton::menuButton(){}

menuButton::menuButton(QWidget *parent, item_t item_name,
                       int x,int y):
    QToolButton(parent)
{
    // generate button
    buttonName = item_name;
    this->setText(getName(buttonName));
    setDefaultIcon();

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
    if(buttonName == GRAPH)
        icon = QIcon(":/icons/graph.png");
    if(buttonName == ANALYSIS)
        icon = QIcon(":/icons/analysis.png");
    if(buttonName == SEARCH)
        icon = QIcon(":/icons/search.png");
    if(buttonName == HELP)
        icon = QIcon(":/icons/help.png");

    this->setIcon(icon);
    this->setIconSize(QSize(40, 40));
    this->setToolButtonStyle(Qt::ToolButtonTextUnderIcon);
}

void menuButton::setPressIcon()
{
    QIcon icon;
    if(buttonName == GRAPH)
        icon = QIcon(":/icons/graph_.png");
    if(buttonName == ANALYSIS)
        icon = QIcon(":/icons/analysis_.png");
    if(buttonName == SEARCH)
        icon = QIcon(":/icons/search_.png");
    if(buttonName == HELP)
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

QString menuButton::getName(item_t item_name)
{
    QString name = tr("");

    if(item_name == GRAPH)
        name = tr("Graph");
    if(item_name == ANALYSIS)
        name = tr("Analysis");
    if(item_name == SEARCH)
        name = tr("Search");
    if(item_name == HELP)
        name = tr("Help");

    return name;
}

int menuButton::getIndex()
{
    return buttonName;
}
