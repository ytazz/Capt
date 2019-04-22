/********************************************************************************
** Form generated from reading UI file 'textfinder.ui'
**
** Created by: Qt User Interface Compiler version 5.12.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_TEXTFINDER_H
#define UI_TEXTFINDER_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QLabel>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_TextFinder
{
public:
    QWidget *widget;
    QVBoxLayout *verticalLayout;
    QHBoxLayout *horizontalLayout_2;
    QLabel *label;
    QLineEdit *lineEdit;
    QPushButton *findButton;
    QTextEdit *textEdit;

    void setupUi(QWidget *TextFinder)
    {
        if (TextFinder->objectName().isEmpty())
            TextFinder->setObjectName(QString::fromUtf8("TextFinder"));
        TextFinder->resize(400, 300);
        widget = new QWidget(TextFinder);
        widget->setObjectName(QString::fromUtf8("widget"));
        widget->setGeometry(QRect(20, 20, 361, 261));
        verticalLayout = new QVBoxLayout(widget);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QString::fromUtf8("verticalLayout"));
        verticalLayout->setContentsMargins(0, 0, 0, 0);
        horizontalLayout_2 = new QHBoxLayout();
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setObjectName(QString::fromUtf8("horizontalLayout_2"));
        label = new QLabel(widget);
        label->setObjectName(QString::fromUtf8("label"));

        horizontalLayout_2->addWidget(label);

        lineEdit = new QLineEdit(widget);
        lineEdit->setObjectName(QString::fromUtf8("lineEdit"));

        horizontalLayout_2->addWidget(lineEdit);

        findButton = new QPushButton(widget);
        findButton->setObjectName(QString::fromUtf8("findButton"));

        horizontalLayout_2->addWidget(findButton);


        verticalLayout->addLayout(horizontalLayout_2);

        textEdit = new QTextEdit(widget);
        textEdit->setObjectName(QString::fromUtf8("textEdit"));

        verticalLayout->addWidget(textEdit);


        retranslateUi(TextFinder);

        QMetaObject::connectSlotsByName(TextFinder);
    } // setupUi

    void retranslateUi(QWidget *TextFinder)
    {
        TextFinder->setWindowTitle(QApplication::translate("TextFinder", "TextFinder", nullptr));
        label->setText(QApplication::translate("TextFinder", "Keyword", nullptr));
        findButton->setText(QApplication::translate("TextFinder", "Find", nullptr));
    } // retranslateUi

};

namespace Ui {
    class TextFinder: public Ui_TextFinder {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_TEXTFINDER_H
