#include "setting_graph.h"

SettingGraph::SettingGraph(QWidget* parent):
    QWidget(parent)
{
    // size
    windowWidth = parent->width();
    windowHeight = parent->height();
    this->setFixedSize(windowWidth, windowHeight);

    // color
    windowColor = QColor("#FFFFFF");
    setWindowColor();

    // set page layout
    pageLayout = new QVBoxLayout(this);
    section[0] = new Section("Setting File", 300, this);
    section[1] = new Section("Coordinate", 300, this);
    section[2] = new Section("Axis", 300, this);
    pageLayout->addWidget(section[0]);
    pageLayout->addWidget(section[1]);
    pageLayout->addWidget(section[2]);
    pageLayout->addStretch();

    QVBoxLayout* layoutSetting = new QVBoxLayout();
    layoutSetting->addWidget(new QLabel("Some Text in Section", section[0]));
    layoutSetting->addWidget(new QPushButton("Button in Section", section[0]));

    QVBoxLayout* anyLayout2 = new QVBoxLayout();
    anyLayout2->addWidget(new QLabel("Some Text in Section", section[1]));
    anyLayout2->addWidget(new QLabel("Some Text in Section", section[1]));
    anyLayout2->addWidget(new QLabel("Some Text in Section", section[1]));
    anyLayout2->addWidget(new QPushButton("Button in Section", section[1]));

    section[0]->setContentLayout(*layoutSetting);
    section[1]->setContentLayout(*anyLayout2);
    section[2]->setContentLayout(*anyLayout2);

    // connect signal & slot
    createConnect();
}

SettingGraph::~SettingGraph(){}

void SettingGraph::createConnect()
{
//    connect()
}

void SettingGraph::setWindowColor()
{
    QLabel *plabel=new QLabel(this);
    plabel->setFixedSize(windowWidth, windowHeight);
    QPalette palette = plabel->palette();
    palette.setColor(plabel->backgroundRole(), windowColor);
    palette.setColor(plabel->foregroundRole(), windowColor);
    plabel->setPalette(palette);
    plabel->setAutoFillBackground(true);
}

void SettingGraph::open()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"), "",
        tr("Text Files (*.txt);;C++ Files (*.cpp *.h)"));

    if (!fileName.isEmpty()) {
        QFile file(fileName);
        if (!file.open(QIODevice::ReadOnly)) {
            QMessageBox::critical(this, tr("Error"), tr("Could not open file"));
            return;
        }
        QTextStream in(&file);
        //textEdit->setText(in.readAll());
        file.close();
    }
}
