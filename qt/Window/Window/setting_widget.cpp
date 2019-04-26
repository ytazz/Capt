#include "setting_widget.h"

SettingWidget::SettingWidget(int width, int height)
{
    // number of pages
    numPage = 4;

    // size
    windowWidth  = width;
    windowHeight = height;
    setFixedSize(windowWidth, windowHeight);

    // color
    windowColor = QColor("#FFFFFF");
    setWindowColor();

    QLabel *plabel[4];
    // generate stacked widget
    stackedWidget = new QStackedWidget(this);
    stackedWidget->setFixedSize(windowWidth, windowHeight);
    for(int i=0; i<numPage; i++){
        page[i] = new QWidget(stackedWidget);
        page[i]->setFixedSize(windowWidth, windowHeight);
        stackedWidget->addWidget(page[i]);
        plabel[i]=new QLabel(page[i]);
        plabel[i]->setFixedSize(windowWidth, windowHeight);
        plabel[i]->setText(QString::number(i));
    }

    stackedWidget->setCurrentWidget(page[0]);
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
    stackedWidget->setCurrentWidget(page[0]);
    printf("page 0\n");
}

void SettingWidget::pageAnalysis()
{
    stackedWidget->setCurrentWidget(page[1]);
    printf("page 1\n");
}

void SettingWidget::pageSearch()
{
    stackedWidget->setCurrentWidget(page[2]);
    printf("page 2\n");
}

void SettingWidget::pageHelp()
{
    stackedWidget->setCurrentWidget(page[3]);
    printf("page 3\n");
}
