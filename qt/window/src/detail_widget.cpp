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
  // layout();
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

void DetailWidget::layout() {
  QGridLayout *layout = new QGridLayout;
  layout->setSpacing(0);
  layout->setMargin(0);
  layout->setContentsMargins(0, 0, 0, 0);
  QLabel *label = new QLabel(this);
  label->setText(tr("hello"));
  label->setFixedSize(width, 20);
  layout->addWidget(label, 0, 0);
  layout->addWidget(label, 0, 1);
  // layout->addStretch();
  this->setLayout(layout);
}

// void MainWindow::paintEvent(QPaintEvent *) //描画イベント
// {
//   QPainter painter(this);
//   painter.setRenderHint(QPainter::Antialiasing, true);
//   //アンチエイリアスセット painter.setPen(QPen(Qt::black, 12,
//   Qt::DashDotLine, Qt::RoundCap)); painter.setBrush(QBrush(Qt::green,
//   Qt::SolidPattern)); painter.drawEllipse(80, 80, 400, 240); //楕円描画
// }

} // namespace CA