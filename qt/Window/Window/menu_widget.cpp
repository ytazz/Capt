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
    buttonGraph    = new menuButton(tr("Graph")   ,this,0,0*menuWidth,menuWidth,menuWidth);
    buttonAnalysis = new menuButton(tr("Analysis"),this,0,1*menuWidth,menuWidth,menuWidth);
    buttonSearch   = new menuButton(tr("Search")  ,this,0,2*menuWidth,menuWidth,menuWidth);
    buttonHelp     = new menuButton(tr("Help")    ,this,0,3*menuWidth,menuWidth,menuWidth);

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
    if(index == 0) return buttonGraph;
    if(index == 1) return buttonAnalysis;
    if(index == 2) return buttonSearch;
    if(index == 3) return buttonHelp;
}

void MenuWidget::connectSignalSlot()
{
    connect(buttonGraph   ->button, &menuButton::pressed, this, &MenuWidget::pressedGraph);
    connect(buttonAnalysis->button, &menuButton::pressed, this, &MenuWidget::pressedAnalysis);
    connect(buttonSearch  ->button, &menuButton::pressed, this, &MenuWidget::pressedSearch);
    connect(buttonHelp    ->button, &menuButton::pressed, this, &MenuWidget::pressedHelp);
}

void MenuWidget::offOthers(menuButton *button)
{
    int nowButton = button->getId();

    for(int i=0;i<numButton;i++){
        if(i==nowButton){
            button->setButtonPressed();
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
