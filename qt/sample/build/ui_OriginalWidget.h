/********************************************************************************
** Form generated from reading UI file 'OriginalWidget.ui'
**
** Created by: Qt User Interface Compiler version 5.5.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_ORIGINALWIDGET_H
#define UI_ORIGINALWIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_OriginalWidget
{
public:
    QVBoxLayout *verticalLayout;
    QLabel *label;

    void setupUi(QWidget *OriginalWidget)
    {
        if (OriginalWidget->objectName().isEmpty())
            OriginalWidget->setObjectName(QStringLiteral("OriginalWidget"));
        OriginalWidget->resize(712, 469);
        verticalLayout = new QVBoxLayout(OriginalWidget);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        label = new QLabel(OriginalWidget);
        label->setObjectName(QStringLiteral("label"));
        QFont font;
        font.setPointSize(50);
        font.setBold(true);
        font.setWeight(75);
        label->setFont(font);
        label->setAlignment(Qt::AlignCenter);

        verticalLayout->addWidget(label);


        retranslateUi(OriginalWidget);

        QMetaObject::connectSlotsByName(OriginalWidget);
    } // setupUi

    void retranslateUi(QWidget *OriginalWidget)
    {
        OriginalWidget->setWindowTitle(QApplication::translate("OriginalWidget", "Original Widget", 0));
        label->setText(QApplication::translate("OriginalWidget", "Last Page", 0));
    } // retranslateUi

};

namespace Ui {
    class OriginalWidget: public Ui_OriginalWidget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_ORIGINALWIDGET_H
