/********************************************************************************
** Form generated from reading UI file 'signalslot.ui'
**
** Created by: Qt User Interface Compiler version 5.12.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_SIGNALSLOT_H
#define UI_SIGNALSLOT_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_SignalSlot
{
public:
    QWidget *widget;
    QHBoxLayout *horizontalLayout;
    QLabel *label;
    QLineEdit *lineEdit;
    QPushButton *pushButton;

    void setupUi(QWidget *SignalSlot)
    {
        if (SignalSlot->objectName().isEmpty())
            SignalSlot->setObjectName(QString::fromUtf8("SignalSlot"));
        SignalSlot->resize(400, 300);
        widget = new QWidget(SignalSlot);
        widget->setObjectName(QString::fromUtf8("widget"));
        widget->setGeometry(QRect(40, 20, 312, 27));
        horizontalLayout = new QHBoxLayout(widget);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        label = new QLabel(widget);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout->addWidget(label);

        lineEdit = new QLineEdit(widget);
        lineEdit->setObjectName(QString::fromUtf8("lineEdit"));

        horizontalLayout->addWidget(lineEdit);

        pushButton = new QPushButton(widget);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));

        horizontalLayout->addWidget(pushButton);


        retranslateUi(SignalSlot);
        QObject::connect(lineEdit, SIGNAL(returnPressed()), pushButton, SLOT(animateClick()));
        QObject::connect(pushButton, SIGNAL(clicked()), SignalSlot, SLOT(close()));

        QMetaObject::connectSlotsByName(SignalSlot);
    } // setupUi

    void retranslateUi(QWidget *SignalSlot)
    {
        SignalSlot->setWindowTitle(QApplication::translate("SignalSlot", "SignalSlot", nullptr));
        label->setText(QApplication::translate("SignalSlot", "TextLabel", nullptr));
        pushButton->setText(QApplication::translate("SignalSlot", "PushButton", nullptr));
    } // retranslateUi

};

namespace Ui {
    class SignalSlot: public Ui_SignalSlot {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_SIGNALSLOT_H
