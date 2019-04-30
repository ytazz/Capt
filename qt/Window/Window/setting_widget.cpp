#include "setting_widget.h"

SettingWidget::SettingWidget(int width, int height)
{
    // number of pages
    numPage = NumberOfItem;

    // size
    windowWidth  = width;
    windowHeight = height;
    setFixedSize(windowWidth, windowHeight);

    // color
    windowColor = QColor("#FFFFFF");
    setWindowColor();

    // generate stacked widget
    stackedWidget = new QStackedWidget(this);
    stackedWidget->setFixedSize(windowWidth, windowHeight);

    // generate each pages
    page[Graph]    = new SettingGraph(stackedWidget);
    page[Analysis] = new SettingGraph(stackedWidget);
    page[Search]   = new SettingGraph(stackedWidget);
    page[Help]     = new SettingGraph(stackedWidget);

    // register pages to stacked widget
    for(int i=0; i<numPage; i++){
        stackedWidget->addWidget(page[i]);
    }
}

SettingWidget::~SettingWidget(){}

void SettingWidget::setWindowColor()
{
    QLabel *plabel=new QLabel(this);
    plabel->setFixedSize(windowWidth, windowHeight);
    QPalette palette = plabel->palette();
    palette.setColor(plabel->backgroundRole(), windowColor);
    palette.setColor(plabel->foregroundRole(), windowColor);
    plabel->setPalette(palette);
    plabel->setAutoFillBackground(true);
}

void SettingWidget::pageGraph()
{
    stackedWidget->setCurrentWidget(page[Graph]);
    printf("page 0\n");
}

void SettingWidget::pageAnalysis()
{
    stackedWidget->setCurrentWidget(page[Analysis]);
    printf("page 1\n");
}

void SettingWidget::pageSearch()
{
    stackedWidget->setCurrentWidget(page[Search]);
    printf("page 2\n");
}

void SettingWidget::pageHelp()
{
    stackedWidget->setCurrentWidget(page[Help]);
    printf("page 3\n");
}
