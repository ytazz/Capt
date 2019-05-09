/********************************************************************************
** Form generated from reading UI file 'originalsignalslot.ui'
**
** Created by: Qt User Interface Compiler version 5.5.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_ORIGINALSIGNALSLOT_H
#define UI_ORIGINALSIGNALSLOT_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
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
            OriginalSignalSlot->setObjectName(QStringLiteral("OriginalSignalSlot"));
        OriginalSignalSlot->resize(400, 300);
        lineEdit = new QLineEdit(OriginalSignalSlot);
        lineEdit->setObjectName(QStringLiteral("lineEdit"));
        lineEdit->setGeometry(QRect(40, 20, 311, 51));
        label = new QLabel(OriginalSignalSlot);
        label->setObjectName(QStringLiteral("label"));
        label->setGeometry(QRect(290, 90, 67, 17));
        pushButton = new QPushButton(OriginalSignalSlot);
        pushButton->setObjectName(QStringLiteral("pushButton"));
        pushButton->setGeometry(QRect(260, 130, 89, 25));

        retranslateUi(OriginalSignalSlot);
        QObject::connect(lineEdit, SIGNAL(textChanged(QString)), OriginalSignalSlot, SLOT(setText(QString)));
        QObject::connect(OriginalSignalSlot, SIGNAL(textLengthChanged(int)), label, SLOT(setNum(int)));

        QMetaObject::connectSlotsByName(OriginalSignalSlot);
    } // setupUi

    void retranslateUi(QWidget *OriginalSignalSlot)
    {
        OriginalSignalSlot->setWindowTitle(QApplication::translate("OriginalSignalSlot", "OriginalSignalSlot", 0));
        label->setText(QApplication::translate("OriginalSignalSlot", "TextLabel", 0));
        pushButton->setText(QApplication::translate("OriginalSignalSlot", "PushButton", 0));
    } // retranslateUi

};

namespace Ui {
    class OriginalSignalSlot: public Ui_OriginalSignalSlot {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_ORIGINALSIGNALSLOT_H
