#include "setting_item.h"

SettingItem::SettingItem(QWidget* parent, item_t item_name):
    QWidget(parent)
{
    // size
    windowWidth = parent->width();
    windowHeight = parent->height();
    this->setFixedSize(windowWidth, windowHeight);
    printf("%d\n",item_name);

    // color
    windowColor = QColor("#FFFFFF");
    setWindowColor();

    // create page
    createPage(item_name);

    // initialize labels


    // connect signal & slot
    createConnect();
}

SettingItem::~SettingItem(){}

void SettingItem::createConnect()
{
//    connect()
}

void SettingItem::createPage(item_t item_name)
{
    if(item_name == GRAPH)    createGraphPage();
    if(item_name == ANALYSIS) createAnalysisPage();
    if(item_name == SEARCH)   createSearchPage();
    if(item_name == HELP)     createHelpPage();
}

void SettingItem::createGraphPage()
{
    // set page layout
    QVBoxLayout* pageLayout = new QVBoxLayout(this);

    // sections
    Section* section[4];
    section[0] = new Section("Setting File", 300, this);
    section[1] = new Section("Coordinate", 300, this);
    section[2] = new Section("Axis (r)", 300, this);
    section[3] = new Section("Axis (theta)", 300, this);
    pageLayout->addWidget(section[0]);
    pageLayout->addWidget(section[1]);
    pageLayout->addWidget(section[2]);
    pageLayout->addWidget(section[3]);
    pageLayout->addStretch();

    // layout for each sections
    // [0] Setting File
    QVBoxLayout* layout_set = new QVBoxLayout();
    layout_set->addWidget(new QLabel(tr("file name"), section[0]));
    layout_set->addWidget(new QPushButton(tr("open file"), section[0]));

    // [1] Coordinate
    QVBoxLayout* layout_coo = new QVBoxLayout();
    layout_coo->addWidget(new QLabel(tr("file name"), section[1]));

    // [2] Axis
    QGridLayout* layout_axi[2];

    // 1st axis (r or x)
    // initialize labels
    label_r_min =new QLabel("0.0",section[2]);
    label_r_max =new QLabel("0.0",section[2]);
    label_r_step=new QLabel("0.0",section[2]);
    label_r_tick=new QLabel("0.0",section[2]);
    // layout
    layout_axi[0] = new QGridLayout();
    layout_axi[0]->addWidget(new QLabel("min", section[2]),0,0);
    layout_axi[0]->addWidget(new QLabel("max", section[2]),1,0);
    layout_axi[0]->addWidget(new QLabel("step", section[2]),2,0);
    layout_axi[0]->addWidget(new QLabel("tick", section[2]),3,0);
    layout_axi[0]->addWidget(label_r_min, 0,1);
    layout_axi[0]->addWidget(label_r_max, 1,1);
    layout_axi[0]->addWidget(label_r_step,2,1);
    layout_axi[0]->addWidget(label_r_tick,3,1);

    // 2nd axis (theta or y)
    // initialize labels
    label_t_min =new QLabel("0.0",section[3]);
    label_t_max =new QLabel("0.0",section[3]);
    label_t_step=new QLabel("0.0",section[3]);
    label_t_tick=new QLabel("0.0",section[3]);
    // layout
    layout_axi[1] = new QGridLayout();
    layout_axi[1]->addWidget(new QLabel("min", section[3]),0,0);
    layout_axi[1]->addWidget(new QLabel("max", section[3]),1,0);
    layout_axi[1]->addWidget(new QLabel("step", section[3]),2,0);
    layout_axi[1]->addWidget(new QLabel("tick", section[3]),3,0);
    layout_axi[1]->addWidget(label_t_min, 0,1);
    layout_axi[1]->addWidget(label_t_max, 1,1);
    layout_axi[1]->addWidget(label_t_step,2,1);
    layout_axi[1]->addWidget(label_t_tick,3,1);

    // register each layouts to section
    section[0]->setContentLayout(*layout_set);
    section[1]->setContentLayout(*layout_coo);
    section[2]->setContentLayout(*layout_axi[0]);
    section[3]->setContentLayout(*layout_axi[1]);
}

void SettingItem::createAnalysisPage()
{
    QLabel *label=new QLabel(this);
    label->setFixedSize(windowWidth, windowHeight);
    label->setText("Sorry, this page (Analysis) doesn't exist ...");
}

void SettingItem::createSearchPage()
{
    QLabel *label=new QLabel(this);
    label->setFixedSize(windowWidth, windowHeight);
    label->setText("Sorry, this page (Search) doesn't exist ...");
}

void SettingItem::createHelpPage()
{
    QLabel *label=new QLabel(this);
    label->setFixedSize(windowWidth, windowHeight);
    label->setText("Sorry, this page (Help) doesn't exist ...");
}

void SettingItem::setWindowColor()
{
    QLabel *plabel=new QLabel(this);
    plabel->setFixedSize(windowWidth, windowHeight);
    QPalette palette = plabel->palette();
    palette.setColor(plabel->backgroundRole(), windowColor);
    palette.setColor(plabel->foregroundRole(), windowColor);
    plabel->setPalette(palette);
    plabel->setAutoFillBackground(true);
}

void SettingItem::open()
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
//        textEdit->setText(in.readAll());
        file.close();
    }
}
