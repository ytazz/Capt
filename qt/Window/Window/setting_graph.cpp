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
