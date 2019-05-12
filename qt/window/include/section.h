#ifndef SECTION_H
#define SECTION_H

#include <QtGui>
#include <QtWidgets>

class Section : public QWidget {
  Q_OBJECT
private:
  QGridLayout *mainLayout;
  QToolButton *toggleButton;
  QFrame *headerLine;
  QParallelAnimationGroup *toggleAnimation;
  QScrollArea *contentArea;
  int animationDuration;

public:
  explicit Section(const QString &title = "", const int animationDuration = 100,
                   QWidget *parent = NULL);

  void setContentLayout(QLayout &contentLayout);
};

#endif // SECTION_H
