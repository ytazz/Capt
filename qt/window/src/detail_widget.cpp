#include "detail_widget.h"

namespace CA {

DetailWidget::DetailWidget(QWidget *parent, int width, int height)
    : QWidget(parent) {
  // size
  this->width = width;
  this->height = height;
  this->setFixedSize(width, height);

  // color
  setWindowColor(QColor("#FFFFFF"));

  // layout
  layout = NULL;
}

DetailWidget::~DetailWidget() {}

void DetailWidget::setWindowColor(QColor color) {
  QLabel *plabel = new QLabel(this);
  plabel->setFixedSize(width, height);
  QPalette palette = plabel->palette();
  palette.setColor(plabel->backgroundRole(), color);
  palette.setColor(plabel->foregroundRole(), color);
  plabel->setPalette(palette);
  plabel->setAutoFillBackground(true);
}

void DetailWidget::setLegend(char *name, char *type, char *color) {
  Legend l;
  l.name = name;
  l.type = type;
  l.color = color;
  legend.push_back(l);
}

void DetailWidget::updateLayout() {
  QTableWidget *table = new QTableWidget(this);
  table->setFixedSize(width, height);
  table->setRowCount(22);
  table->setColumnCount(2);
  table->verticalHeader()->setVisible(false);
  table->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
  table->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
}

} // namespace CA