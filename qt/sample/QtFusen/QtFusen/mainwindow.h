#ifndef MAINWINDOW_H
#define MAINWINDOW_H

///#include <QtGui/QMainWindow>
#include <QMainWindow>
#include <QSystemTrayIcon>

class QAction;
class QCheckBox;
class QComboBox;
class QGroupBox;
class QLabel;
class QLineEdit;
class QMenu;
class QPushButton;
class QSpinBox;
class QTextEdit;
class QPlainTextEdit;
class QSystemTrayIcon;
class QTreeWidget;
class Fusen;

class MainWindow : public QMainWindow
{
	Q_OBJECT

public:
	MainWindow(QWidget *parent = 0,  Qt::WindowFlags flags = 0);
	~MainWindow();

public slots:
	void	newFusen();
	void	font();
	void	topMost(bool);
	void	onClose(Fusen *);
	void	toOpenDialog();
	void	closedDialog();
	void	onRecieved(const QString);
	void	systrayActivated(QSystemTrayIcon::ActivationReason);
	void	onMouseEntered(Fusen *);

protected:
    void createActions();
    void createTrayIcon();
    void createMenus();
    void createCentralWidget();

    bool	isDuplicatePos(const Fusen *);
    Fusen	*createFusen();
    void addFusen(Fusen *);
    void removeFusen(Fusen *);

private:
    class QFont		m_font;
    QVector<Fusen *>	m_fusenList;

    QAction	*newFusenAct;
    QAction	*showAct;
    QAction	*quitAct;
    QAction	*fontAct;
    QAction	*m_topMostAct;

    QMenu	*m_systrayMenu;
	QSystemTrayIcon *m_systrayIcon;
	QTreeWidget	*m_treeWidget;
};

#endif // MAINWINDOW_H
