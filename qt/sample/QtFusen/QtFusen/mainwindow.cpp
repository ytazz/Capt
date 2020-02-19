///#include <QtGui>
#include <QtWidgets>
#include "mainwindow.h"
//#include <QSystemTrayIcon>
#include	"Fusen.h"

MainWindow::MainWindow(QWidget *parent,  Qt::WindowFlags flags)
	: QMainWindow(parent, flags)
{
	createActions();
	createTrayIcon();
	createMenus();
	createCentralWidget();
	m_systrayIcon->show();

	QSettings settings;
	//settings.setDefaultFormat(QSettings::IniFormat);
	const QString fontFamily = settings.value("fontFamily", "Arial").toString();
	const int fontSize = settings.value("fontSize", 12).toInt();
	m_font = QFont(fontFamily, fontSize);
	m_topMostAct->setChecked(settings.value("topMost", true).toBool());
	int nFusen = settings.value("nFusen", 0).toInt();
	if( !nFusen )
		newFusen();
	for(int ix = 0; ix < nFusen; ++ix) {
		Fusen *ptr = createFusen();
		//m_fusenList.push_back(ptr);
		ptr->move(settings.value(QString("fusen%1/pos").arg(ix+1), QPoint(100, 100)).toPoint());
		ptr->resize(settings.value(QString("fusen%1/size").arg(ix+1), QSize(200, 200)).toSize());
		ptr->setText(settings.value(QString("fusen%1/text").arg(ix+1), "").toString());
		int r = settings.value(QString("fusen%1/firstColorR").arg(ix+1), 255).toInt();
		int g = settings.value(QString("fusen%1/firstColorG").arg(ix+1), 255).toInt();
		int b = settings.value(QString("fusen%1/firstColorB").arg(ix+1), 255).toInt();
		int a = settings.value(QString("fusen%1/firstColorAlpha").arg(ix+1), 255).toInt();
		ptr->setFirstColor(QColor(r, g, b, a));
		r = settings.value(QString("fusen%1/secondColorR").arg(ix+1), 255).toInt();
		g = settings.value(QString("fusen%1/secondColorG").arg(ix+1), 255).toInt();
		b = settings.value(QString("fusen%1/secondColorB").arg(ix+1), 255).toInt();
		a = settings.value(QString("fusen%1/secondColorAlpha").arg(ix+1), 255).toInt();
		ptr->setSecondColor(QColor(r, g, b, a));
		ptr->setGradType(settings.value(QString("fusen%1/gradType").arg(ix+1), 0).toInt());
		const QString fontFamily = settings.value(QString("fusen%1/fontFamily").arg(ix+1), "Arial").toString();
		const int fontSize = settings.value(QString("fusen%1/fontSize").arg(ix+1), 12).toInt();
		ptr->setFont(QFont(fontFamily, fontSize));
		//ptr->setWindowIcon(QIcon(QPixmap(":QtFusen/Resources/rect2985.png")));
		ptr->show();
		ptr->hideStatusBarWidgets();
		addFusen(ptr);
	}
}

MainWindow::~MainWindow()
{
	QSettings settings;
	//settings.setDefaultFormat(QSettings::IniFormat);
	settings.setValue("fontFamily", m_font.family());
	settings.setValue("fontSize", m_font.pointSize());
	settings.setValue("topMost", m_topMostAct->isChecked());
	settings.setValue("nFusen", m_fusenList.size());
	for(int ix = 0; ix < m_fusenList.size(); ++ix) {
		const Fusen *ptr = m_fusenList[ix];
		settings.setValue(QString("fusen%1/pos").arg(ix+1), ptr->pos());
		settings.setValue(QString("fusen%1/size").arg(ix+1), ptr->size());
		settings.setValue(QString("fusen%1/text").arg(ix+1), ptr->text());
		int r, g, b, a;
		ptr->firstColor().getRgb(&r, &g, &b, &a);
		settings.setValue(QString("fusen%1/firstColorR").arg(ix+1), r);
		settings.setValue(QString("fusen%1/firstColorG").arg(ix+1), g);
		settings.setValue(QString("fusen%1/firstColorB").arg(ix+1), b);
		settings.setValue(QString("fusen%1/firstColorAlpha").arg(ix+1), a);
		ptr->secondColor().getRgb(&r, &g, &b, &a);
		settings.setValue(QString("fusen%1/secondColorR").arg(ix+1), r);
		settings.setValue(QString("fusen%1/secondColorG").arg(ix+1), g);
		settings.setValue(QString("fusen%1/secondColorB").arg(ix+1), b);
		settings.setValue(QString("fusen%1/secondColorAlpha").arg(ix+1), a);
		settings.setValue(QString("fusen%1/gradType").arg(ix+1), ptr->gradType());
		settings.setValue(QString("fusen%1/fontFamily").arg(ix+1), ptr->font().family());
		settings.setValue(QString("fusen%1/fontSize").arg(ix+1), ptr->font().pointSize());
	}
}

void MainWindow::onRecieved(const QString buf)
{
	newFusen();
}

Fusen *MainWindow::createFusen()
{
	Fusen *ptr = new Fusen(m_topMostAct->isChecked());
	connect(ptr, SIGNAL(onClose(Fusen *)), this, SLOT(onClose(Fusen *)));
	connect(ptr, SIGNAL(newFusenClicked()), this, SLOT(newFusen()));
	connect(ptr, SIGNAL(mouseEntered(Fusen *)), this, SLOT(onMouseEntered(Fusen *)));
	ptr->setFont(m_font);
	return ptr;
}

void MainWindow::onMouseEntered(Fusen *ptr)
{
	for(int ix = 0; ix < m_fusenList.size(); ++ix) {
		if( m_fusenList[ix] != ptr )
			m_fusenList[ix]->hideStatusBarWidgets();
	}
}

bool MainWindow::isDuplicatePos(const Fusen *ptr)
{
	QPoint p = ptr->pos();
	for(int ix = 0; ix < m_fusenList.size(); ++ix) {
		if( m_fusenList[ix] != ptr && m_fusenList[ix]->pos() == p )
			return true;
	}
	return false;
}

void MainWindow::newFusen()
{
	Fusen *ptr = createFusen();
	addFusen(ptr);
	//m_fusenList.push_back(ptr);
	ptr->show();
	ptr->hideStatusBarWidgets();
	while( isDuplicatePos(ptr) ) {
		QPoint p = ptr->pos();
		ptr->move(p += QPoint(16, 16));
	}
}

void MainWindow::onClose(Fusen *ptr)
{
	for(int ix = 0; ix < m_fusenList.size(); ++ix) {
		if( m_fusenList[ix] == ptr ) {
			m_fusenList.remove(ix);
			return;
		}
	}
}

void MainWindow::createActions()
{
	newFusenAct = new QAction(tr("&NewFusen"), this);
	//newFusenAct->setShortcuts(QKeySequence::New);
	connect(newFusenAct, SIGNAL(triggered()), this, SLOT(newFusen()));
	showAct = new QAction(tr("&MainWindow"), this);
	connect(showAct, SIGNAL(triggered()), this, SLOT(show()));
	quitAct = new QAction(tr("&Quit"), this);
	connect(quitAct, SIGNAL(triggered()), qApp, SLOT(quit()));
	fontAct = new QAction(tr("&Font..."), this);
	connect(fontAct, SIGNAL(triggered()), this, SLOT(font()));
	m_topMostAct = new QAction(tr("&TopMost"), this);
	m_topMostAct->setCheckable(true);
	connect(m_topMostAct, SIGNAL(toggled(bool)), this, SLOT(topMost(bool)));
}
void MainWindow::createTrayIcon()
{
	m_systrayMenu = new QMenu(this);
	m_systrayMenu->addAction(newFusenAct);
	//m_systrayMenu->addAction(showAct);
	m_systrayMenu->addAction(m_topMostAct);
	m_systrayMenu->addSeparator();
	m_systrayMenu->addAction(quitAct);

	m_systrayIcon = new QSystemTrayIcon(this);
	m_systrayIcon->setContextMenu(m_systrayMenu);
	m_systrayIcon->setToolTip(tr("QtFusen ver. 0.006\nright click to Menu"));
	m_systrayIcon->setIcon(QIcon(":QtFusen/Resources/rect2985.png"));
	connect(m_systrayIcon, SIGNAL(activated(QSystemTrayIcon::ActivationReason)),
			this, SLOT(systrayActivated(QSystemTrayIcon::ActivationReason)));
}
void MainWindow::createMenus()
{
	QMenu *fileMenu = menuBar()->addMenu(tr("&Fusen"));
	fileMenu->addAction(newFusenAct);
	m_systrayMenu->addSeparator();
	fileMenu->addAction(quitAct);

	QMenu *settingsMenu = menuBar()->addMenu(tr("&Settings"));
	settingsMenu->addAction(fontAct);
	settingsMenu->addAction(m_topMostAct);
}
void MainWindow::createCentralWidget()
{
	m_treeWidget = new QTreeWidget();
	QStringList sl;
	sl.push_back("text");
	m_treeWidget->setHeaderItem(new QTreeWidgetItem(sl));
	setCentralWidget(m_treeWidget);
}
void MainWindow::systrayActivated(QSystemTrayIcon::ActivationReason reason)
{
	switch( reason ) {
	case QSystemTrayIcon::DoubleClick:
		newFusen();
		break;
	}
}

void MainWindow::addFusen(Fusen *ptr)
{
	connect(ptr, SIGNAL(toOpenDialog()), this, SLOT(toOpenDialog()));
	connect(ptr, SIGNAL(closedDialog()), this, SLOT(closedDialog()));
	m_fusenList.push_back(ptr);
	QStringList sl;
	QString t = ptr->text();
	int ix = t.indexOf(QRegExp("[\\r\\n]"));		//	改行コード検索
	if( ix >= 0 ) t = t.left(ix);
	sl.push_back(t);
	m_treeWidget->addTopLevelItem(new QTreeWidgetItem(sl));
}

void MainWindow::font()
{
	bool ok;
	QFont font = QFontDialog::getFont(&ok, m_font, this);
	if( ok ) {
		m_font = font;
		for(int ix = 0; ix < m_fusenList.size(); ++ix) {
			m_fusenList[ix]->setFont(font);
		}
	}
}
void MainWindow::topMost(bool b)
{
	//QSettings settings;
	//settings.setValue("topMost", b);
	for(int ix = 0; ix < m_fusenList.size(); ++ix) {
		Fusen *ptr = m_fusenList[ix];
		ptr->setTopMost(b);
		ptr->setWindowIcon(QIcon(QPixmap(":QtFusen/Resources/rect2985.png")));
#if 0
		ptr->hide();
		QPoint pos = ptr->pos();
		if( b ) {
			//ptr->raise();
			ptr->setWindowFlags( Qt::WindowStaysOnTopHint | Qt::FramelessWindowHint );
		} else {
			//ptr->lower();
			ptr->setWindowFlags( Qt::FramelessWindowHint );
		}
		ptr->move(pos);
		ptr->show();
#endif
	}
}
void MainWindow::toOpenDialog()
{
	if( !m_topMostAct->isChecked() )
		return;
	topMost(false);
}
void MainWindow::closedDialog()
{
	if( !m_topMostAct->isChecked() )
		return;
	topMost(true);
}
