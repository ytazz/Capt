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

  // 不要かも
  table->clear();

  // テーブルサイズを決定
  table->setColumnCount(2);
  table->setRowCount(20);

  // 列のタイトル文字列を一度に指定
  table->setHorizontalHeaderLabels(QStringList()
                                   << tr("Title 1") << tr("Title 2"));

  // セルを埋める
  table->setItem(0, 0, new QTableWidgetItem("a"));
  table->setItem(0, 1, new QTableWidgetItem("")); // 空白の場合

  // 行の高さを指定　行ごとに指定する必要がある様子
  table->setRowHeight(0, 20);

  // 一行選択モードに設定
  table->setSelectionMode(QAbstractItemView::ContiguousSelection);
  table->setSelectionBehavior(QAbstractItemView::SelectRows);

  table->setMaximumHeight(height);
  table->setMinimumHeight(height);
  table->setMaximumWidth(width);
  table->setMinimumWidth(width);
  // this->setWidget(table);
}

} // namespace CA