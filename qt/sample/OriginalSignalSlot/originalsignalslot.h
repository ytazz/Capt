#ifndef ORIGINALSIGNALSLOT_H
#define ORIGINALSIGNALSLOT_H

#include <QWidget>

namespace Ui {
class OriginalSignalSlot;
}

class OriginalSignalSlot : public QWidget
{
    Q_OBJECT

public:
    explicit OriginalSignalSlot(QWidget *parent = nullptr);
    ~OriginalSignalSlot();

public slots:
    void setText(const QString &text);

signals:
    void textLengthChanged(int textLength);

private slots:
    void on_pushButton_clicked();

private:
    Ui::OriginalSignalSlot *ui;
};

#endif // ORIGINALSIGNALSLOT_H
