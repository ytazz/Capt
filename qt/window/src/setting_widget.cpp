#include "setting_widget.h"

namespace CA {

SettingWidget::SettingWidget(QWidget *parent, int width, int height)
    : QWidget(parent) {
  // number of pages
  numPage = NumberOfItem;

  // size
  windowWidth = width;
  windowHeight = height;
  setFixedSize(windowWidth, windowHeight);

  // color
  windowColor = QColor("#FFFFFF");
  setWindowColor();

  // generate stacked widget
  stackedWidget = new QStackedWidget(this);
  stackedWidget->setFixedSize(windowWidth, windowHeight);

  // generate each pages
  for (int i = 0; i < numPage; i++) {
    page[i] = new SettingItem(stackedWidget, i);
  }

  // register pages to stacked widget
  for (int i = 0; i < numPage; i++) {
    stackedWidget->addWidget(page[i]);
  }
}

SettingWidget::~SettingWidget() {}

void SettingWidget::setWindowColor() {
  QLabel *plabel = new QLabel(this);
  plabel->setFixedSize(windowWidth, windowHeight);
  QPalette palette = plabel->palette();
  palette.setColor(plabel->backgroundRole(), windowColor);
  palette.setColor(plabel->foregroundRole(), windowColor);
  plabel->setPalette(palette);
  plabel->setAutoFillBackground(true);
}

void SettingWidget::pageGraph() {
  stackedWidget->setCurrentWidget(page[GRAPH]);
  printf("page 0\n");
}

void SettingWidget::pageAnalysis() {
  stackedWidget->setCurrentWidget(page[ANALYSIS]);
  printf("page 1\n");
}

void SettingWidget::pageSearch() {
  stackedWidget->setCurrentWidget(page[SEARCH]);
  printf("page 2\n");
}

void SettingWidget::pageHelp() {
  stackedWidget->setCurrentWidget(page[HELP]);
  printf("page 3\n");
}

} // namespace CA
