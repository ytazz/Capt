#include "setting_item.h"

using namespace CA;

SettingItem::SettingItem(QWidget *parent, item_t item_name) : QWidget(parent) {
  // size
  windowWidth = parent->width();
  windowHeight = parent->height();
  this->setFixedSize(windowWidth, windowHeight);

  // color
  windowColor = QColor("#FFFFFF");
  setWindowColor();

  // create page
  createPage(item_name);

  // connect signal & slot
  createConnection(item_name);
}

SettingItem::~SettingItem() {}

void SettingItem::createConnection(item_t item_name) {
  if (item_name == GRAPH)
    QObject::connect(button_file, &QToolButton::pressed, this,
                     &SettingItem::openFile);
}

void SettingItem::createPage(item_t item_name) {
  if (item_name == GRAPH)
    createGraphPage();
  if (item_name == ANALYSIS)
    createAnalysisPage();
  if (item_name == SEARCH)
    createSearchPage();
  if (item_name == HELP)
    createHelpPage();
}

void SettingItem::createGraphPage() {
  // set page layout
  QVBoxLayout *pageLayout = new QVBoxLayout(this);

  // sections
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
  label_file_name = new QLabel(tr("file name"), section[0]);
  label_file_name->setFixedSize(windowWidth - 80, 70);
  button_file = new QToolButton(section[0]);
  button_file->setIcon(
      QApplication::style()->standardIcon(QStyle::SP_DialogOpenButton));
  button_file->setIconSize(QSize(30, 30));
  QHBoxLayout *layout_set = new QHBoxLayout();
  layout_set->addWidget(label_file_name);
  pageLayout->addStretch();
  layout_set->addWidget(button_file);

  // [1] Coordinate
  label_coordinate = new QLabel(tr("-"), section[1]);
  QVBoxLayout *layout_coo = new QVBoxLayout();
  layout_coo->addWidget(label_coordinate);

  // [2] Axis
  QGridLayout *layout_axi[2];

  // 1st axis (r or x)
  // initialize labels
  label_r_min = new QLabel("-", section[2]);
  label_r_max = new QLabel("-", section[2]);
  label_r_step = new QLabel("-", section[2]);
  label_r_tick = new QLabel("-", section[2]);
  // layout
  layout_axi[0] = new QGridLayout();
  layout_axi[0]->addWidget(new QLabel("min", section[2]), 0, 0);
  layout_axi[0]->addWidget(new QLabel("max", section[2]), 1, 0);
  layout_axi[0]->addWidget(new QLabel("step", section[2]), 2, 0);
  layout_axi[0]->addWidget(new QLabel("tick", section[2]), 3, 0);
  layout_axi[0]->addWidget(label_r_min, 0, 1);
  layout_axi[0]->addWidget(label_r_max, 1, 1);
  layout_axi[0]->addWidget(label_r_step, 2, 1);
  layout_axi[0]->addWidget(label_r_tick, 3, 1);

  // 2nd axis (theta or y)
  // initialize labels
  label_t_min = new QLabel("-", section[3]);
  label_t_max = new QLabel("-", section[3]);
  label_t_step = new QLabel("-", section[3]);
  label_t_tick = new QLabel("-", section[3]);
  // layout
  layout_axi[1] = new QGridLayout();
  layout_axi[1]->addWidget(new QLabel("min", section[3]), 0, 0);
  layout_axi[1]->addWidget(new QLabel("max", section[3]), 1, 0);
  layout_axi[1]->addWidget(new QLabel("step", section[3]), 2, 0);
  layout_axi[1]->addWidget(new QLabel("tick", section[3]), 3, 0);
  layout_axi[1]->addWidget(label_t_min, 0, 1);
  layout_axi[1]->addWidget(label_t_max, 1, 1);
  layout_axi[1]->addWidget(label_t_step, 2, 1);
  layout_axi[1]->addWidget(label_t_tick, 3, 1);

  // register each layouts to section
  section[0]->setContentLayout(*layout_set);
  section[1]->setContentLayout(*layout_coo);
  section[2]->setContentLayout(*layout_axi[0]);
  section[3]->setContentLayout(*layout_axi[1]);
}

void SettingItem::createAnalysisPage() {
  QLabel *label = new QLabel(this);
  label->setFixedSize(windowWidth, windowHeight);
  label->setText("Sorry, this page (Analysis)\n doesn't exist ...");
}

void SettingItem::createSearchPage() {
  QLabel *label = new QLabel(this);
  label->setFixedSize(windowWidth, windowHeight);
  label->setText("Sorry, this page (Search)\n doesn't exist ...");
}

void SettingItem::createHelpPage() {
  QLabel *label = new QLabel(this);
  label->setFixedSize(windowWidth, windowHeight);
  label->setText("Sorry, this page (Help)\n doesn't exist ...");
}

void SettingItem::setWindowColor() {
  QLabel *plabel = new QLabel(this);
  plabel->setFixedSize(windowWidth, windowHeight);
  QPalette palette = plabel->palette();
  palette.setColor(plabel->backgroundRole(), windowColor);
  palette.setColor(plabel->foregroundRole(), windowColor);
  plabel->setPalette(palette);
  plabel->setAutoFillBackground(true);
}

void SettingItem::openFile() {
  QString file_name_qstr = QFileDialog::getOpenFileName(
      this, tr("Open Setting File"), "", tr("XML Files (*.xml)"));

  if (!file_name_qstr.isEmpty()) {
    std::string file_name = file_name_qstr.toStdString();
    Graph graph(file_name);
    graph.parse();

    label_file_name->setText(QString::fromStdString(file_name));
    label_file_name->setWordWrap(true);
    label_coordinate->setText(
        QString::fromStdString(graph.getStr("coordinate", "type")));
    label_r_min->setText(QString::number(graph.getVal("radius", "min")));
    label_r_max->setText(QString::number(graph.getVal("radius", "max")));
    label_r_step->setText(QString::number(graph.getVal("radius", "step")));
    label_r_tick->setText(QString::number(graph.getVal("radius", "tick")));
    label_t_min->setText(QString::number(graph.getVal("angle", "min")));
    label_t_max->setText(QString::number(graph.getVal("angle", "max")));
    label_t_step->setText(QString::number(graph.getVal("angle", "step")));
    label_t_tick->setText(QString::number(graph.getVal("angle", "tick")));

    setPolarGridRadius(graph.getVal("radius", "min"),
                       graph.getVal("radius", "max"),
                       graph.getVal("radius", "step"), "gray");
    setPolarGridAngle(graph.getVal("angle", "min"),
                      graph.getVal("angle", "max"),
                      graph.getVal("angle", "step"), "gray");

    Model model("nao.xml");
    model.parse();
    // setLine(model.getVec("link", "foot_r"), "black");
    // std::vector<Vector2> steppable;
    // Vector2 vec;
    // float radius, angle;
    // for (int i = 0; i <= 4; i++) {
    //   radius = 0.12 + 0.01 * i;
    //   for (int j = 0; j <= 7; j++) {
    //     angle = M_PI / 3.0 + M_PI / 18.0 * j;
    //     vec.setPolar(radius, angle);
    //     steppable.push_back(vec);
    //   }
    // }
    // setPoints(steppable, "blue");

    Param param("analysis.xml");
    param.parse();

    Vector2 center;
    center.setPolar(0.0, 0.0);
    setArc(center, 0.05, 20 * M_PI / 180, 160 * M_PI / 180, "red");
    // setCircle(center, 0.1, "red");

    Pendulum pendulum(model);
    Vector2 icp, icp_, cop;
    icp.setPolar(0.056, 2.64);
    cop.setPolar(0.04, 2.64);
    pendulum.setIcp(icp);
    pendulum.setCop(cop);
    setPoint(icp, "green");
    setPoint(cop, "black");

    float v_max = 1.0;
    float t_min = 0.1;
    float dt = 0.01;
    float r_foot = 0.04;
    float l_max = 0.22;
    Vector2 sw0;
    sw0.setPolar(0.158, 1.12);

    float t = 0.3;
    icp_ = pendulum.getIcp(t);
    setPoint(icp_, "blue");
    for (int j = 0; j < 360; j++) {
      float sw_r = v_max * (t - t_min);
      float sw_th = j * M_PI / 180.0;
      float sw_x = sw0.x + sw_r * cos(sw_th);
      float sw_y = sw0.y + sw_r * sin(sw_th);
      Vector2 swft;
      swft.setCartesian(sw_x, sw_y);
      setPoint(swft, "red");
      float norm = (swft - icp_).norm();
      printf("deg = \t%3.0d, \tnorm = \t%lf\n", j, norm);
      if (0.09 <= swft.r && swft.r <= 0.22 && 20.0 * M_PI / 180.0 <= swft.th &&
          swft.th <= 160.0 * M_PI / 180.0) {
        if (norm <= r_foot) {
        } else if (norm <= l_max * exp(-5.718 * t) + r_foot) {
        }
      }
    }

    paint();
  }
}
