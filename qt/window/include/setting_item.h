#ifndef SETTING_ITEM_H
#define SETTING_ITEM_H

#include "base.h"
#include "section.h"
#include <QtGui>
#include <QtWidgets>

class SettingItem : public QWidget {
  Q_OBJECT

public:
  explicit SettingItem(QWidget *parent = nullptr, item_t item_name = 0);
  ~SettingItem();

  QLabel *label_file_name, *label_file_button;
  QLabel *label_coordinate;
  QLabel *label_r_min, *label_r_max, *label_r_step, *label_r_tick;
  QLabel *label_t_min, *label_t_max, *label_t_step, *label_t_tick;

  QPushButton *button_file;

private:
  int windowWidth, windowHeight;
  QColor windowColor;

  void setWindowColor();

  void createPage(item_t item_name);
  void createGraphPage();
  void createAnalysisPage();
  void createSearchPage();
  void createHelpPage();

  void open();
  void createConnect();
};

#endif // SETTING_ITEM_H
