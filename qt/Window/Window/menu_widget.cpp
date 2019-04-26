#include "menu_widget.h"

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

    //setButtonPressed();
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
    button->setDisabled(true);
    setPressIcon();
    printf("pressed");
}

void menuButton::setButtonReleased()
{
    setButtonColor(buttonColor);
    button->setDisabled(false);
}

void menuButton::setButtonColor(QColor windowColor)
{
    button->setAutoRaise(true);
    QPalette palette = button->palette();
    palette.setColor(QPalette::Button, windowColor);
    palette.setColor(QPalette::Background, windowColor);
    palette.setColor(QPalette::Window, windowColor);
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

MenuWidget::MenuWidget(int width, int height)
{
    // number of buttons
    numButton = 4;

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

    connectSignalSlot();
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

menuButton* MenuWidget::getButton(int index)
{
    if(index == 0) return buttonGraph;
    if(index == 1) return buttonAnalysis;
    if(index == 2) return buttonSearch;
    if(index == 3) return buttonHelp;
}

void MenuWidget::connectSignalSlot()
{
    connect(buttonGraph, &menuButton::pressed, buttonGraph, &menuButton::setButtonPressed);
//    connect(this, SIGNAL (buttonGraph->pressed()), this, SLOT(offOthers(buttonGraph)));
//    connect(this, SIGNAL (buttonGraph->pressed()), this, SLOT(offOthers(buttonGraph)));
//    connect(this, SIGNAL (buttonGraph->pressed()), this, SLOT(offOthers(buttonGraph)));
}

void MenuWidget::offOthers(menuButton *button)
{
//    int nowButton = button->getId();
//    printf("button %d is pressed!\n",nowButton);

//    for(int i=0;i<numButton;i++){
//        if(i==nowButton){
//            button->setButtonPressed();
//        }else{
//            getButton(i)->setButtonReleased();
//        }
//    }

    printf("hello");
}
