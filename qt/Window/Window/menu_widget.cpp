#include "menu_widget.h"

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
    buttonGraph    = new menuButton(tr("Graph")   ,this,0,0*menuWidth);
    buttonAnalysis = new menuButton(tr("Analysis"),this,0,1*menuWidth);
    buttonSearch   = new menuButton(tr("Search")  ,this,0,2*menuWidth);
    buttonHelp     = new menuButton(tr("Help")    ,this,0,3*menuWidth);

    connectSignalSlot();
    pressedGraph();
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
    menuButton* button = new menuButton(tr(""),this,0,5*menuWidth);
    if(index == 0) button = buttonGraph;
    if(index == 1) button = buttonAnalysis;
    if(index == 2) button = buttonSearch;
    if(index == 3) button = buttonHelp;

    return button;
}

void MenuWidget::connectSignalSlot()
{
    connect(buttonGraph   , &menuButton::pressed, this, &MenuWidget::pressedGraph);
    connect(buttonAnalysis, &menuButton::pressed, this, &MenuWidget::pressedAnalysis);
    connect(buttonSearch  , &menuButton::pressed, this, &MenuWidget::pressedSearch);
    connect(buttonHelp    , &menuButton::pressed, this, &MenuWidget::pressedHelp);
}

void MenuWidget::offOthers(menuButton *button)
{
    int nowButton = button->getId();

    for(int i=0;i<numButton;i++){
        if(i==nowButton){
            getButton(i)->setButtonPressed();
        }else{
            getButton(i)->setButtonReleased();
        }
    }
}

void MenuWidget::pressedGraph()
{
    offOthers(buttonGraph);
}

void MenuWidget::pressedAnalysis()
{
    offOthers(buttonAnalysis);
}

void MenuWidget::pressedSearch()
{
    offOthers(buttonSearch);
}

void MenuWidget::pressedHelp()
{
    offOthers(buttonHelp);
}
