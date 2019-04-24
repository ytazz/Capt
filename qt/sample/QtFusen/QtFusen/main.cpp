#include "mainwindow.h"
///#include <QtGui>
#include <QtWidgets>
#include	"SingleApplication.h"

int main(int argc, char *argv[])
{
	SingleApplication app(argc, argv, "QtFusen");
	//app.checkSingleApp();
	if( !app.isFirstApp() ) {	//	2つめ以降のインスタンスの場合
		//
		app.sendMessage("newFusen");
		return 0;
	}
	//	ひとつめのインスタンスの場合
	if( !QSystemTrayIcon::isSystemTrayAvailable() ) {
		QMessageBox::critical(0, QObject::tr("Systray"),
							  QObject::tr("I couldn't detect any system tray "
										  "on this system."));
		return 1;
	}
	app.setQuitOnLastWindowClosed(false);		//	ウィンドウが無くなっても終了しない

	app.setApplicationName("QtFusen");
	app.setOrganizationName("N.Tsuda");

	QString locale = QLocale::system().name();
	QTranslator translator;
	bool rc = translator.load(QString("qtfusen2_") + locale);
	if( !rc ) {
		qDebug() << "can't open '" << QString("qtfusen2_") << locale << "'";
	}
	app.installTranslator(&translator);

	MainWindow w;
	QObject::connect(&app, SIGNAL(onRecieved(const QString)),
						&w, SLOT(onRecieved(const QString)));
	//w.show();
	app.setWindowIcon(QIcon(QPixmap(":QtFusen/Resources/rect2985.png")));
	return app.exec();
}
