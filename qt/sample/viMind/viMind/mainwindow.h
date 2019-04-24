#ifndef MAINWINDOW_H
#define MAINWINDOW_H

///#include <QtGui/QMainWindow>
#include <QMainWindow>
#include <QtGui>
#include <QPrinter>
#include <QPrintDialog>
#include <QPrintPreviewDialog>
//#include <QtPrintSupport/QPrinter>
//#include <QtPrintSupport/QPrintDialog>
//#include <QtPrintSupport/QPrintPreviewDialog>


class Scene;
class View;
class Node;
class QAction;
class QActionGroup;
class QDomElement;

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	MainWindow(QWidget *parent = 0, Qt::WindowFlags flags = 0);
	~MainWindow();

public:
	bool	isUntitled() const { return m_curFullPath.isEmpty(); }

protected:
	void	createActions();
    void	createMenus();
    void	createToolBars();
    void	createDockWindows();
    void	setCurrentFile(const QString &fileName);
	QString curDir() const;
    bool	saveFile(const QString &fileName);
	bool	setContent(const QByteArray &);
    void	updateWindowTitle();
    void	updateCurFile();
    bool	loadFileToThisMainWindow(const QString &fileName);
    void	loadFile(const QString &fileName);
	//void	addNode(Node *, QDomElement &);

protected slots:
	void	open();
    bool	save();
    bool	saveAs();
    void	print();
    void	printPreview();
    void	onPaintRequested(QPrinter*);
	void	showMessage(const QString &, int = 0);

private:
	Scene	*m_scene;
	View	*m_view;
	//Node	*m_rootNode;
    //bool	m_isUntitled;
    QString	m_curFile;		//	ファイル名部分のみ（含拡張子）
    QString	m_curFullPath;

private:
    QAction *m_openAct;
    QAction *m_saveAct;
    QAction *m_printAct;
    QAction *m_printPreviewAct;

    QAction	*m_undoAct;
    QAction	*m_redoAct;
    QAction	*m_cutAct;
    QAction	*m_copyAct;
    //QAction	*m_pasteAfterAct;
    QAction	*m_pasteChildAct;
    QAction	*m_editNodeAct;
    QAction	*m_deleteAct;
    QAction	*m_insertChildNodeAct;
    QAction	*m_collapseExpandAct;		//	子ノードを展開・折畳
    QAction	*m_moveNodeLeftAct;
    QAction	*m_moveNodeRightAct;
    QAction	*m_moveNodeUpAct;
    QAction	*m_moveNodeDownAct;
    QActionGroup	*m_nodeStyleGroup;
    QAction	*m_formatForkAct;
    QAction	*m_formatRectAct;
    QAction	*m_formatRoundRectAct;
    QAction	*m_formatCircleRectAct;

};

#endif // MAINWINDOW_H
