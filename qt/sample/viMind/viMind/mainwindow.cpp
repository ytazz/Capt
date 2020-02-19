///#include <QtGui>
#include <QtWidgets>
#include <QtXml>
#include "mainwindow.h"
#include "Scene.h"
#include "View.h"
#include "Node.h"
#include <Qdebug>

#define		VERSION_STR		"0.000"

MainWindow::MainWindow(QWidget *parent, Qt::WindowFlags flags)
	: QMainWindow(parent, flags)
{
	//m_isUntitled = true;

	m_scene = new Scene;
	m_scene->setSceneRect(-320, -240, 640, 480);	//	領域設定
	m_view = new View;
	m_view->setRenderHints(QPainter::Antialiasing);
	m_view->setScene(m_scene);		//	ビューとシーンを接続
	setCentralWidget(m_view);		//	ビューをセントラルウィジェットに指定

	m_view->setDragMode(QGraphicsView::ScrollHandDrag);		//	クリック＆ドラッグによるスクロール
	//m_scene->addItem(m_rootNode = new Node("Root"));
	//m_rootNode->setPos(0, 0);
	//m_scene->addEllipse(-2, -2, 4, 4);

#if 0
	View *view2 = new View;
	view2->setScene(m_scene);
	view2->show();
#endif

	//	シグナルを m_scene に接続するので、シーン生成後にコールすること
	createActions();
	createMenus();

#if 0
	QGraphicsTextItem *textItem = 
	m_scene->addText("Root");		//	中央に“Root”を表示
	//textItem->setFlags(QGraphicsItem::ItemIsSelectable);	//	選択可能
	textItem->setFlags(QGraphicsItem::ItemIsSelectable | QGraphicsItem::ItemIsMovable);		// 選択・移動可能
	//textItem->setFlags(QGraphicsItem::ItemIsSelectable |
	//					QGraphicsItem::ItemIsMovable |
	//					QGraphicsItem::ItemIsFocusable );
	//textItem->setTextInteractionFlags(Qt::TextEditorInteraction);
						//	選択、カーソル移動、文字入力、削除、undo/redo が可能になる
	textItem->setCursor(Qt::IBeamCursor);
	qDebug() << textItem->parentItem();
#endif
	updateWindowTitle();
}

MainWindow::~MainWindow()
{

}
void MainWindow::createActions()
{
    m_openAct = new QAction(QIcon(":viMind/Resources/open.png"), tr("&Open..."), this);
    m_openAct->setShortcuts(QKeySequence::Open);
    m_openAct->setToolTip(tr("open map"));
    m_openAct->setStatusTip(tr("Open an existing file"));
    connect(m_openAct, SIGNAL(triggered()), this, SLOT(open()));

    m_saveAct = new QAction(QIcon(":viMind/Resources/save.png"), tr("&Save"), this);
    m_saveAct->setShortcuts(QKeySequence::Save);
    m_saveAct->setToolTip(tr("save map"));
    m_saveAct->setStatusTip(tr("Save the map to disk"));
    connect(m_saveAct, SIGNAL(triggered()), this, SLOT(save()));

    m_printAct = new QAction(tr("&Print"), this);
    m_printAct->setShortcuts(QKeySequence::Print);
    m_printAct->setToolTip(tr("print map"));
    m_printAct->setStatusTip(tr("Print the map"));
    connect(m_printAct, SIGNAL(triggered()), this, SLOT(print()));

    m_printPreviewAct = new QAction(tr("Print Pre&view"), this);
    m_printPreviewAct->setToolTip(tr("print preview"));
    m_printPreviewAct->setStatusTip(tr("Print previrew the map"));
    connect(m_printPreviewAct, SIGNAL(triggered()), this, SLOT(printPreview()));

	m_undoAct = new QAction(QIcon(":viMind/Resources/undo.png"), tr("&Undo"), this);
    m_undoAct->setShortcuts(QKeySequence::Undo);
    m_undoAct->setToolTip(tr("undo"));
    m_undoAct->setStatusTip(tr("undo edit command"));
	//m_undoAct->setEnabled(false);
    connect(m_undoAct, SIGNAL(triggered()), m_scene, SLOT(undo()));
    connect(m_scene, SIGNAL(canUndoChanged(bool)), m_undoAct, SLOT(setEnabled(bool)));

	m_redoAct = new QAction(QIcon(":viMind/Resources/redo.png"), tr("&Redo"), this);
    m_redoAct->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_R));
    //m_redoAct->setShortcuts(QKeySequence::Redo);
    m_redoAct->setToolTip(tr("redo"));
    m_redoAct->setStatusTip(tr("redo edit commande"));
	//m_redoAct->setEnabled(false);
    connect(m_redoAct, SIGNAL(triggered()), m_scene, SLOT(redo()));
    connect(m_scene, SIGNAL(canRedoChanged(bool)), m_redoAct, SLOT(setEnabled(bool)));

    m_cutAct = new QAction(QIcon(":viMind/Resources/cut.png"), tr("Cu&t"), this);
    m_cutAct->setShortcuts(QKeySequence::Cut);
    m_cutAct->setToolTip(tr("cut"));
    m_cutAct->setStatusTip(tr("Cut the current selection's contents to the clipboard"));
    connect(m_cutAct, SIGNAL(triggered()), m_scene, SLOT(cut()));

    m_copyAct = new QAction(QIcon(":viMind/Resources/copy.png"), tr("&Copy"), this);
    m_copyAct->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_C));
    //m_copyAct->setShortcuts(QKeySequence::Copy);
    m_copyAct->setToolTip(tr("copy"));
    m_copyAct->setStatusTip(tr("Copy the current selection's contents to the clipboard"));
    connect(m_copyAct, SIGNAL(triggered()), m_scene, SLOT(copy()));

#if 0
    m_pasteAfterAct = new QAction(QIcon(":viMind/Resources/paste.png"), tr("&PasteAfter"), this);
    m_pasteAfterAct->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_V));
    m_pasteAfterAct->setToolTip(tr("paste"));
    m_pasteAfterAct->setStatusTip(tr("Paste the clipboard's contents into the current selection"));
    connect(m_pasteAfterAct, SIGNAL(triggered()), m_scene, SLOT(paste()));
#endif

    m_pasteChildAct = new QAction(tr("Pas&teChild"), this);
    m_pasteChildAct->setShortcut(QKeySequence(/*Qt::SHIFT +*/ Qt::CTRL + Qt::Key_V));
    m_pasteChildAct->setToolTip(tr("paste child"));
    m_pasteChildAct->setStatusTip(tr("Paste the clipboard's contents as current node's child"));
    connect(m_pasteChildAct, SIGNAL(triggered()), m_scene, SLOT(pasteChild()));

    m_deleteAct = new QAction(QIcon(":viMind/Resources/delete.png"), tr("&Delete"), this);
    m_deleteAct->setShortcuts(QKeySequence::Delete);
    m_deleteAct->setToolTip(tr("Delete"));
    m_deleteAct->setStatusTip(tr("Delete the current selected nodes"));
    connect(m_deleteAct, SIGNAL(triggered()), m_scene, SLOT(removeSelectedNode()));

    m_editNodeAct = new QAction(tr("&EditNodeText"), this);
    m_editNodeAct->setShortcut(QKeySequence(Qt::Key_F2));
    m_editNodeAct->setStatusTip(tr("Edit selected node's text"));
    connect(m_editNodeAct, SIGNAL(triggered()), m_scene, SLOT(editNode()));

    m_insertChildNodeAct = new QAction(tr("&ChildNode"), this);
    m_insertChildNodeAct->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_Insert));
    m_insertChildNodeAct->setStatusTip(tr("insert a child node"));
    connect(m_insertChildNodeAct, SIGNAL(triggered()), m_scene, SLOT(addChildNode()));

    m_collapseExpandAct = new QAction(tr("CollapseExpand"), this);
    m_collapseExpandAct->setShortcut(QKeySequence(Qt::Key_Space));
    m_collapseExpandAct->setStatusTip(tr("collapse or expand child nodes"));
    connect(m_collapseExpandAct, SIGNAL(triggered()), m_scene, SLOT(doCollapseExpand()));

    m_moveNodeLeftAct = new QAction(tr("&Left"), this);
    m_moveNodeLeftAct->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_Left));
    m_moveNodeLeftAct->setStatusTip(tr("move current node to left"));
    connect(m_moveNodeLeftAct, SIGNAL(triggered()), m_scene, SLOT(moveNodeLeft()));

    m_moveNodeRightAct = new QAction(tr("&Right"), this);
    m_moveNodeRightAct->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_Right));
    m_moveNodeRightAct->setStatusTip(tr("move current node to Right"));
    connect(m_moveNodeRightAct, SIGNAL(triggered()), m_scene, SLOT(moveNodeRight()));

    m_moveNodeUpAct = new QAction(tr("&Up"), this);
    m_moveNodeUpAct->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_Up));
    m_moveNodeUpAct->setStatusTip(tr("move current node to Up"));
    connect(m_moveNodeUpAct, SIGNAL(triggered()), m_scene, SLOT(moveNodeUp()));

    m_moveNodeDownAct = new QAction(tr("&Down"), this);
    m_moveNodeDownAct->setShortcut(QKeySequence(Qt::CTRL + Qt::Key_Down));
    m_moveNodeDownAct->setStatusTip(tr("move current node to Down"));
    connect(m_moveNodeDownAct, SIGNAL(triggered()), m_scene, SLOT(moveNodeDown()));

    m_formatForkAct = new QAction(tr("&Fork"), this);
    m_formatForkAct->setStatusTip(tr("Fork node style"));
    m_formatForkAct->setCheckable(true);
    connect(m_formatForkAct, SIGNAL(triggered()), m_scene, SLOT(formatFork()));

    m_formatRectAct = new QAction(tr("&Rect"), this);
    m_formatRectAct->setStatusTip(tr("Rect node style"));
    m_formatRectAct->setCheckable(true);
    connect(m_formatRectAct, SIGNAL(triggered()), m_scene, SLOT(formatRect()));

    m_formatRoundRectAct = new QAction(tr("&RoundRect"), this);
    m_formatRoundRectAct->setStatusTip(tr("RoundRect node style"));
    m_formatRoundRectAct->setCheckable(true);
    connect(m_formatRoundRectAct, SIGNAL(triggered()), m_scene, SLOT(formatRoundRect()));

    m_formatCircleRectAct = new QAction(tr("&CircleRect"), this);
    m_formatCircleRectAct->setStatusTip(tr("CircleRect node style"));
    m_formatCircleRectAct->setCheckable(true);
    connect(m_formatCircleRectAct, SIGNAL(triggered()), m_scene, SLOT(formatCircleRect()));

    m_nodeStyleGroup = new QActionGroup(this);
    m_nodeStyleGroup->setExclusive(true);
    m_nodeStyleGroup->addAction(m_formatForkAct);
    m_nodeStyleGroup->addAction(m_formatRectAct);
    m_nodeStyleGroup->addAction(m_formatRoundRectAct);
    m_nodeStyleGroup->addAction(m_formatCircleRectAct);

}
void MainWindow::createMenus()
{
    QMenu *fileMenu = menuBar()->addMenu(tr("&File"));
    fileMenu->addAction(m_openAct);
    fileMenu->addAction(m_saveAct);
    fileMenu->addSeparator();
    fileMenu->addAction(m_printAct);
    fileMenu->addAction(m_printPreviewAct);

    QMenu *editMenu = menuBar()->addMenu(tr("&Edit"));
    editMenu->addAction(m_undoAct);
    editMenu->addAction(m_redoAct);
    editMenu->addAction(m_cutAct);
    editMenu->addAction(m_copyAct);
    editMenu->addAction(m_pasteChildAct);
    editMenu->addAction(m_deleteAct);
    editMenu->addAction(m_editNodeAct);
    editMenu->addSeparator();
    QMenu *moveNodeMenu = editMenu->addMenu(tr("&MoveNode"));
	    moveNodeMenu->addAction(m_moveNodeLeftAct);
	    moveNodeMenu->addAction(m_moveNodeRightAct);
	    moveNodeMenu->addAction(m_moveNodeUpAct);
	    moveNodeMenu->addAction(m_moveNodeDownAct);

    QMenu *insertMenu = menuBar()->addMenu(tr("&Insert"));
    insertMenu->addAction(m_insertChildNodeAct);

    QMenu *formatMenu = menuBar()->addMenu(tr("Forma&t"));
    formatMenu->addAction(m_formatForkAct);
    formatMenu->addAction(m_formatRectAct);
    formatMenu->addAction(m_formatRoundRectAct);
    formatMenu->addAction(m_formatCircleRectAct);

    QMenu *naviMenu = menuBar()->addMenu(tr("&Navigation"));
    naviMenu->addAction(m_collapseExpandAct);
}
QString MainWindow::curDir() const
{
	QString curDir;
	if( !isUntitled() ) {
		QDir dir(m_curFullPath);
		dir.cdUp();
		curDir = dir.path();
	}
	return curDir;
}
void MainWindow::open()
{
	m_scene->setMode(Scene::COMMAND);
	QString fileName = QFileDialog::getOpenFileName(this, tr("Open File"), curDir(),
													tr("viMind(*.vmd *.mm);;All Files(*.*)"));
	if( fileName.isEmpty() ) return;
	loadFileToThisMainWindow(fileName);
}
bool MainWindow::save()
{
	m_scene->setMode(Scene::COMMAND);
    if( isUntitled() ) {
        return saveAs();
    } else {
        return saveFile(m_curFullPath);
    }
}
bool MainWindow::saveAs()
{
	m_scene->setMode(Scene::COMMAND);
	QString title = m_curFile;
	if( isUntitled() ) {
		Node *node = m_scene->rootNode();
		QString text = node->toPlainText();
		if( !text.isEmpty() && text != tr("Root") ) {
			text.replace("\n", "");
			title = text;
		}
	}
	QString cdir = curDir();
	if( !cdir.isEmpty() )
		QDir::setCurrent(cdir);
	QString fileName = QFileDialog::getSaveFileName(this, tr("Save File"), title,
													tr("viMind(*.vmd *.mm);;All Files(*.*)"));
	if( fileName.isEmpty() ) return false;
    setCurrentFile(fileName);
	return saveFile(fileName);
}

bool MainWindow::saveFile(const QString &fileName)
{
	QFile fp(fileName);
	if( !fp.open(QIODevice::WriteOnly) ) {
		return false;
	}
	QString buffer = m_scene->toXmlText(geometry());
	QTextCodec *codec = QTextCodec::codecForName("UTF-8");
	QByteArray ba = codec->fromUnicode(buffer);
	fp.write(ba);
	fp.close();
	showMessage(tr("written %1 byte").arg(ba.size()),
				5000);
    setWindowModified(false);
    //m_scene->setClean();
    updateWindowTitle();
	return true;
}
void MainWindow::setCurrentFile(const QString &fullPath)
{
    m_curFullPath = fullPath;
    if( isUntitled() ) {
        m_curFile.clear();
    } else {
        m_curFile = QFileInfo(fullPath).fileName();
    }
    updateCurFile();

#if 0
    setWindowModified(false);
    m_scene->setClean();
    updateWindowTitle();
#endif

#if 0
    QSettings settings;
    QStringList files = settings.value("recentFileList").toStringList();
    files.removeAll(fullPath);
    files.push_front(fullPath);
    while (files.size() > MaxRecentFiles)
        files.removeLast();
    settings.setValue("recentFileList", files);
    updateRecentFileActionsOfAllMainWindows();
#endif
}
void MainWindow::showMessage(const QString &mess, int limit)
{
	statusBar()->showMessage(mess, limit);
}
void MainWindow::updateCurFile()
{
    static int sequenceNumber = 0;
    if( m_curFile.isEmpty() )
        m_curFile = tr("map%1.vmd").arg(++sequenceNumber);
}
void MainWindow::updateWindowTitle()
{
	updateCurFile();
	QString title = m_curFile;
	if( isWindowModified() )
 		title += "*";
	title += " - viMind ver. ";
	title += VERSION_STR;
    setWindowTitle(title);
}
bool MainWindow::loadFileToThisMainWindow(const QString &fileName)
{
	QFile fp(fileName);
	if( !fp.open(QIODevice::ReadOnly) ) {
        QMessageBox::warning(this, tr("viMind"),
                             tr("Cannot read file %1:\n%2.")
                             .arg(fileName)
                             .arg(fp.errorString()));
		return false;
	}
	QByteArray ba = fp.readAll();
	fp.close();
	setContent(ba);
	m_scene->rootNode()->ensureVisible();
    setCurrentFile(fileName);
    statusBar()->showMessage(tr("File loaded"), 2000);
    //addToWindowsList(fileName);
	return true;
}
bool MainWindow::setContent(const QByteArray &ba)
{
	QTextCodec *codec = QTextCodec::codecForName("UTF-8");
	QString buffer = codec->toUnicode(ba);
	QDomDocument doc;
	if( !doc.setContent(buffer) ) {
		QMessageBox::warning(0, tr("open failed."), tr("invalid contents."));
		return false;
	}
	QDomElement root = doc.documentElement();
	//printElement(root, 0);
	//	最初のエレメントは <map version="...">
	if( root.tagName() != "map" ) {
		return false;
	}
	const qreal left = root.attribute("LEFT", "0").toDouble();
	const qreal top = root.attribute("TOP", "0").toDouble();
	const qreal width = root.attribute("WIDTH", "-1").toDouble();
	const qreal height = root.attribute("HEIGHT", "-1").toDouble();
	//const qreal rootX = root.attribute("ROOTX", "-1").toDouble();
	if( width > 0 && height > 0 ) {
                m_scene->setSceneRect(left, top, width, height);
		//m_scene->setDefaultSceneRect(QRectF(0, 0, width, height));
	}
	const int winWd = root.attribute("WINWD", "0").toInt();
	const int winHt = root.attribute("WINHT", "0").toInt();
	if( winWd > 0 && winHt > 0 ) {
		QRect g = geometry();
		g.setWidth(winWd);
		g.setHeight(winHt);
		setGeometry(g);
	}
	root = root.firstChildElement();
	if( root.tagName() != "node" ) {
		return false;
	}
	m_scene->removeAll();
	Node *rootNode = m_scene->createNode(0, root);
	//Node *rootNode = m_scene->createNode(0, Scene::LAST_CHILD, root, true, true);

	m_scene->setRootNode(rootNode);
	m_scene->addNode(rootNode, root);	//	DOM エレメントを rootNode 以下に追加
	//m_scene->layoutAll();		//	setRootNodeXPos() から呼ばれるので必要ない
	m_scene->setSelectedNode(rootNode);
	///rootNode->setTextWidthRecursive();
	//m_scene->setRootNodeXPos(rootX);
	const qreal sx = root.attribute("SX", QString("%1").arg(winWd/2)).toDouble();
	const qreal sy = root.attribute("SY", QString("%1").arg(winHt/2)).toDouble();
	rootNode->setPos(sx, sy);
#if 0
	for(;;) {
		root = root.nextSiblingElement();
		if( root.isNull() || root.tagName() != "node" )
			break;
		Node *node = m_scene->createNode(0, Scene::LAST_CHILD, root, true, true);
		//m_floatingNodes.push_back(node);
		node->setFloatingNode(true);
		addNode(node, root);
		const qreal sx = root.attribute("SX", "0").toDouble();
		const qreal sy = root.attribute("SY", "0").toDouble();
		node->setPos(sx, sy);
	}
#endif
	m_scene->layoutAll();
	return true;
}
#if 0
//	parentNode に子ノードを追加する
void MainWindow::addNode(Node *parentNode, QDomElement &element)
{
	const uint cdt = QDateTime::currentDateTime().toTime_t();
	QDomElement childEle = element.firstChildElement();
	while( !childEle.isNull() ) {
		//printElement(childEle, lvl + 1);
		const QString tagName = childEle.tagName();
		if( tagName == "node" ) {
			Node *node = m_scene->createNode(parentNode, childEle);
			//Node *node = m_scene->createNode(parentNode, Scene::LAST_CHILD, childEle, false, true);
			addNode(node, childEle);
#if 0
			const QString position = childEle.attribute("POSITION");
			if( !position.isEmpty() ) {
				node->setRightSideRecursive(position == "right");
				node->updateLinkIconPosRecursive();
			}
#endif
		}
		childEle = childEle.nextSiblingElement();
	}
}
#endif
void MainWindow::print()
{
	QPrinter printer(QPrinter::HighResolution);
	//printer.setPageSize(QPrinter::A4);
	if (QPrintDialog(&printer).exec() == QDialog::Accepted) {
		onPaintRequested(&printer);
#if 0
		QPainter painter(&printer);
		//painter.setRenderHint(QPainter::Antialiasing);
		m_scene->render(&painter);
#endif
	}
}
void MainWindow::printPreview()
{
	QPrinter printer(QPrinter::HighResolution);
	QPrintPreviewDialog pvDlg(&printer);
	connect(&pvDlg, SIGNAL(paintRequested ( QPrinter *)), this, SLOT(onPaintRequested(QPrinter*)));
	pvDlg.exec();
}
void MainWindow::onPaintRequested(QPrinter *printer)
{
#if 0
	QPainter painter(printer);
    QRectF vrect = painter.viewport();
	QRectF srect = m_scene->nodeBoundingRect();
	const qreal vRatio = vrect.width() / vrect.height();
	if( srect.width() / srect.height() > vRatio ) {		//	ノードが横長の場合
		const qreal ht = srect.width() / vRatio;
		srect.setTop(srect.top() - (ht - srect.height()) / 2);
		srect.setHeight(ht);
	} else if( srect.width() / srect.height() < vRatio ) {		//	ノードが縦長の場合
		const qreal wd = vRatio * srect.height();
		srect.setLeft(srect.left() - (wd - srect.width()) / 2);
		srect.setWidth(wd);
	}
	painter.setRenderHint(QPainter::Antialiasing);
	m_scene->render(&painter, vrect, srect);
#else
	QPainter painter(printer);
	painter.setRenderHint(QPainter::Antialiasing);
    QRectF vrect = painter.viewport();
	QRectF srect = m_scene->sceneRect();
	m_scene->render(&painter, vrect, srect);
#endif
}
