#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    createActions();
    createMenus();
    createToolBars();

    cw = new QWidget();
    setCentralWidget(cw);

    widgetMenu = new QWidget();
    widgetMenu->setGeometry(0, 0, 80, 900);
    QLabel *plabel=new QLabel(tr("label-1"), widgetMenu);
    plabel->setGeometry(0, 0, 80, 900);

    QPalette palette = plabel->palette();
    palette.setColor(plabel->backgroundRole(), Qt::red);
    palette.setColor(plabel->foregroundRole(), Qt::red);
    plabel->setPalette(palette);
    plabel->setAutoFillBackground(true);

    widgetSetting = new QWidget();
    setCentralWidget(widgetSetting);
    widgetSetting->setGeometry(80, 0, 120, 900);
    QLabel *plabel2=new QLabel(tr("label-1"), widgetSetting);
    plabel2->setGeometry(80, 0, 120, 900);

    QPalette palette2 = plabel2->palette();
    palette2.setColor(plabel2->backgroundRole(), Qt::blue);
    palette2.setColor(plabel2->foregroundRole(), Qt::blue);
    plabel2->setPalette(palette2);
    plabel2->setAutoFillBackground(true);
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::createActions()
{
    QIcon icon;

    // Open file
    icon = QApplication::style()->standardIcon( QStyle::SP_DialogOpenButton );
    openAct = new QAction(icon, tr("Open"), this);
    openAct->setStatusTip(tr("Open the project file"));
    connect(openAct, SIGNAL(triggered()), this, SLOT(saveAs()));

    // Save File
    icon = QApplication::style()->standardIcon( QStyle::SP_DialogSaveButton );
    saveAsAct = new QAction(icon, tr("Save&As"), this);
    saveAsAct->setShortcuts(QKeySequence::SaveAs);
    saveAsAct->setStatusTip(tr("Save the project file under a new filename"));
    connect(saveAsAct, SIGNAL(triggered()), this, SLOT(saveAs()));
}

void MainWindow::createMenus()
{
    QMenu *fileMenu = menuBar()->addMenu(tr("File"));
    fileMenu->addAction(openAct);
    fileMenu->addAction(saveAsAct);
}

void MainWindow::createToolBars()
{
    QToolBar *fileToolBar = addToolBar(tr("File"));
    fileToolBar->addAction(openAct);
    fileToolBar->addAction(saveAsAct);
}
