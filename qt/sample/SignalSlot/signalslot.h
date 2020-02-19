#ifndef SIGNALSLOT_H
#define SIGNALSLOT_H

#include <QWidget>

namespace Ui {
class SignalSlot;
}

class SignalSlot : public QWidget
{
    Q_OBJECT

public:
    explicit SignalSlot(QWidget *parent = nullptr);
    ~SignalSlot();

private:
    Ui::SignalSlot *ui;
};

#endif // SIGNALSLOT_H
