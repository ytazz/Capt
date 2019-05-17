#include "main_window.h"
#include "ui_mainwindow.h"

namespace CA {

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow) {
  ui->setupUi(this);
  setWindowTitle(tr("Capturability Based Analysis"));

  createActions();
  createMenus();
  createToolBars();

  cw = new QWidget(this);
  this->setCentralWidget(cw);

  widgetMenu = new MenuWidget(this, 80, 890);
  widgetSetting = new SettingWidget(this, 240, 890);
  widgetScene = new GLWidget(this, 700, 700);
  widgetDetail = new DetailWidget(this, 250, 700);
  QWidget *widgetConsole = new QWidget();
  widgetDetail->updateLayout();

  QGridLayout *layout = new QGridLayout;
  layout->setSpacing(0);
  layout->setMargin(0);
  layout->setContentsMargins(0, 0, 0, 0);
  layout->addWidget(widgetMenu, 0, 0, 2, 1);
  layout->addWidget(widgetSetting, 0, 1, 2, 1);
  layout->addWidget(widgetScene, 0, 2);
  layout->addWidget(widgetDetail, 0, 3);
  layout->addWidget(widgetConsole, 1, 2, 1, 2);
  this->centralWidget()->setLayout(layout);

  widgetConsole->setFixedSize(950, 186);

  setWindowColor(widgetConsole, 950, 186, QColor("#FFFFFF"));

  createConnection();
}

MainWindow::~MainWindow() { delete ui; }

void MainWindow::createConnection() {
  connect(widgetMenu->buttonGraph, &menuButton::pressed, widgetSetting,
          &SettingWidget::pageGraph);
  connect(widgetMenu->buttonAnalysis, &menuButton::pressed, widgetSetting,
          &SettingWidget::pageAnalysis);
  connect(widgetMenu->buttonSearch, &menuButton::pressed, widgetSetting,
          &SettingWidget::pageSearch);
  connect(widgetMenu->buttonHelp, &menuButton::pressed, widgetSetting,
          &SettingWidget::pageHelp);
  connect(widgetSetting->page[GRAPH], &SettingItem::setPolarGridRadius,
          widgetScene, &GLWidget::setPolarGridRadius);
  connect(widgetSetting->page[GRAPH], &SettingItem::setPolarGridAngle,
          widgetScene, &GLWidget::setPolarGridAngle);
  connect(widgetSetting->page[GRAPH], &SettingItem::paint, widgetScene,
          &GLWidget::paint);
  connect(widgetSetting->page[GRAPH], &SettingItem::setPoint, widgetScene,
          &GLWidget::setPoint);
  connect(widgetSetting->page[GRAPH], &SettingItem::setPoints, widgetScene,
          &GLWidget::setPoints);
  connect(widgetSetting->page[GRAPH], &SettingItem::setPolygon, widgetScene,
          &GLWidget::setPolygon);
}

void MainWindow::createActions() {
  QIcon icon;

  // Open file
  icon = QApplication::style()->standardIcon(QStyle::SP_DialogOpenButton);
  openAct = new QAction(icon, tr("Open"), this);
  openAct->setStatusTip(tr("Open the project file"));
  // connect(openAct, SIGNAL(triggered()), this, SLOT(saveAs()));

  // Save File
  icon = QApplication::style()->standardIcon(QStyle::SP_DialogSaveButton);
  saveAsAct = new QAction(icon, tr("Save&As"), this);
  saveAsAct->setShortcuts(QKeySequence::SaveAs);
  saveAsAct->setStatusTip(tr("Save the project file under a new filename"));
  // connect(saveAsAct, SIGNAL(triggered()), this, SLOT(saveAs()));
}

void MainWindow::createMenus() {
  QMenu *fileMenu = menuBar()->addMenu(tr("File"));
  fileMenu->addAction(openAct);
  fileMenu->addAction(saveAsAct);
}

void MainWindow::createToolBars() {
  QToolBar *fileToolBar = addToolBar(tr("File"));
  fileToolBar->addAction(openAct);
  fileToolBar->addAction(saveAsAct);
}

void MainWindow::setWindowColor(QWidget *widget, int width, int height,
                                const QColor color) {
  QLabel *plabel = new QLabel(widget);
  plabel->setFixedSize(width, height);
  QPalette palette = plabel->palette();
  palette.setColor(plabel->backgroundRole(), color);
  palette.setColor(plabel->foregroundRole(), color);
  plabel->setPalette(palette);
  plabel->setAutoFillBackground(true);
}

} // namespace CA
