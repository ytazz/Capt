#include "originalsignalslot.h"
#include "ui_originalsignalslot.h"

OriginalSignalSlot::OriginalSignalSlot(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::OriginalSignalSlot)
{
    ui->setupUi(this);
}

OriginalSignalSlot::~OriginalSignalSlot()
{
    delete ui;
}

void OriginalSignalSlot::setText(const QString &text)
{
    emit textLengthChanged(text.length());
}

void OriginalSignalSlot::on_pushButton_clicked()
{
    ui->lineEdit->clear();
}
