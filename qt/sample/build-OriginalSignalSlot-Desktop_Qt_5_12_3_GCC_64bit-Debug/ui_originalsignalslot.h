/********************************************************************************
** Form generated from reading UI file 'originalsignalslot.ui'
**
** Created by: Qt User Interface Compiler version 5.12.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_ORIGINALSIGNALSLOT_H
#define UI_ORIGINALSIGNALSLOT_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_OriginalSignalSlot
{
public:
    QLineEdit *lineEdit;
    QLabel *label;
    QPushButton *pushButton;

    void setupUi(QWidget *OriginalSignalSlot)
    {
        if (OriginalSignalSlot->objectName().isEmpty())
            OriginalSignalSlot->setObjectName(QString::fromUtf8("OriginalSignalSlot"));
        OriginalSignalSlot->resize(400, 300);
        lineEdit = new QLineEdit(OriginalSignalSlot);
        lineEdit->setObjectName(QString::fromUtf8("lineEdit"));
        lineEdit->setGeometry(QRect(40, 20, 311, 51));
        label = new QLabel(OriginalSignalSlot);
        label->setObjectName(QString::fromUtf8("label"));
        label->setGeometry(QRect(290, 90, 67, 17));
        pushButton = new QPushButton(OriginalSignalSlot);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));
        pushButton->setGeometry(QRect(260, 130, 89, 25));

        retranslateUi(OriginalSignalSlot);
        QObject::connect(lineEdit, SIGNAL(textChanged(QString)), OriginalSignalSlot, SLOT(setText(QString)));
        QObject::connect(OriginalSignalSlot, SIGNAL(textLengthChanged(int)), label, SLOT(setNum(int)));

        QMetaObject::connectSlotsByName(OriginalSignalSlot);
    } // setupUi

    void retranslateUi(QWidget *OriginalSignalSlot)
    {
        OriginalSignalSlot->setWindowTitle(QApplication::translate("OriginalSignalSlot", "OriginalSignalSlot", nullptr));
        label->setText(QApplication::translate("OriginalSignalSlot", "TextLabel", nullptr));
        pushButton->setText(QApplication::translate("OriginalSignalSlot", "PushButton", nullptr));
    } // retranslateUi

};

namespace Ui {
    class OriginalSignalSlot: public Ui_OriginalSignalSlot {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_ORIGINALSIGNALSLOT_H
