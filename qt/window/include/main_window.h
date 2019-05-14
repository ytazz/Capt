#ifndef MAIN_WINDOW_H
#define MAIN_WINDOW_H

#include "gl_widget.h"
#include "menu_widget.h"
#include "setting_widget.h"
#include <QApplication>
#include <QMainWindow>
#include <QtGui>
#include <QtWidgets>

class QAction;
class QMenu;
// class MenuWidget;
// class SettingWidget;

namespace Ui {
class MainWindow;
}

namespace CA {

class MainWindow : public QMainWindow {
  Q_OBJECT

public:
  explicit MainWindow(QWidget *parent = nullptr);
  ~MainWindow();

  void createActions();
  void createMenus();
  void createToolBars();

  void setWindowColor(QWidget *widget, int width, int height,
                      const QColor color);

  void createConnection();

private:
  Ui::MainWindow *ui;
  QAction *openAct;
  QAction *saveAsAct;

  QMenu *fileMenu;
  QMenu *editMenu;

  QWidget *cw; // central widget
  MenuWidget *widgetMenu;
  SettingWidget *widgetSetting;
  GLWidget *widgetScene;
  QWidget *widgetDetail;
  QWidget *widgetConsole;
};

} // namespace CA

#endif // MAIN_WINDOW_H
