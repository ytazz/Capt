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

    // generate stacked widget
    stackedWidget = new QStackedWidget(this);
    stackedWidget->setFixedSize(windowWidth, windowHeight);
//    for(int i=0; i<numPage; i++){
//        page[i] = new SettingGraph(stackedWidget);
//        page[i]->setFixedSize(windowWidth, windowHeight);
//        stackedWidget->addWidget(page[i]);
//    }

//    stackedWidget->setCurrentWidget(page[0]);
    QWidget* widget = new SettingGraph(stackedWidget);
    widget->setFixedSize(windowWidth, windowHeight);
    stackedWidget->addWidget(widget);

    QVBoxLayout* layout = new QVBoxLayout(widget);
    Section* section = new Section("Section", 300, widget);
    layout->addWidget(section);
    Section* section2 = new Section("Section2", 300, widget);
    layout->addWidget(section2);
    layout->addStretch();

    QVBoxLayout* anyLayout = new QVBoxLayout();
    anyLayout->addWidget(new QLabel("Some Text in Section", section));
    anyLayout->addWidget(new QPushButton("Button in Section", section));

    QVBoxLayout* anyLayout2 = new QVBoxLayout();
    anyLayout2->addWidget(new QLabel("Some Text in Section", section2));
    anyLayout2->addWidget(new QLabel("Some Text in Section", section2));
    anyLayout2->addWidget(new QLabel("Some Text in Section", section2));
    anyLayout2->addWidget(new QPushButton("Button in Section", section2));

    section->setContentLayout(*anyLayout);
    section2->setContentLayout(*anyLayout2);
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
