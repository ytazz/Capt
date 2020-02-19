#include "signalslot.h"
#include "ui_signalslot.h"

SignalSlot::SignalSlot(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::SignalSlot)
{
    ui->setupUi(this);
}

SignalSlot::~SignalSlot()
{
    delete ui;
}
