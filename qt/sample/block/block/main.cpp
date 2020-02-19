#include "blockmain.h"
///#include <QtGui/QApplication>
#include <QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	BlockMain w;
	w.show();
	return a.exec();
}
