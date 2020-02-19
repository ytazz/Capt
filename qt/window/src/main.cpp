#include "main_window.h"
#include <QApplication>

using namespace CA;

int main(int argc, char *argv[]) {
  QApplication app(argc, argv);
  app.setWindowIcon(QIcon(":/icons/window.png"));

  MainWindow w;
  w.show();

  return app.exec();
}
