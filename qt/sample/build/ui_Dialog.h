/********************************************************************************
** Form generated from reading UI file 'Dialog.ui'
**
** Created by: Qt User Interface Compiler version 5.5.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_DIALOG_H
#define UI_DIALOG_H

#include <QtCore/QVariant>
#include <QtWidgets/QAction>
#include <QtWidgets/QApplication>
#include <QtWidgets/QButtonGroup>
#include <QtWidgets/QDialog>
#include <QtWidgets/QFrame>
#include <QtWidgets/QHBoxLayout>
#include <QtWidgets/QHeaderView>
#include <QtWidgets/QLabel>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QSpacerItem>
#include <QtWidgets/QStackedWidget>
#include <QtWidgets/QTextEdit>
#include <QtWidgets/QVBoxLayout>
#include <QtWidgets/QWidget>
#include "page_widget/OriginalWidget.h"

QT_BEGIN_NAMESPACE

class Ui_Dialog
{
public:
    QVBoxLayout *verticalLayout;
    QFrame *frame_2;
    QHBoxLayout *horizontalLayout_6;
    QPushButton *pushButton_addPage;
    QPushButton *pushButton_insertPage;
    QTextEdit *textEdit_pageText;
    QFrame *line;
    QPushButton *pushButton_removePage;
    QSpacerItem *horizontalSpacer_3;
    QLabel *label_4;
    QLabel *label_pageIndicator;
    QFrame *frame;
    QHBoxLayout *horizontalLayout_2;
    QStackedWidget *stackedWidget;
    QWidget *page;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label;
    OriginalWidget *page_3;
    QFrame *frame_3;
    QHBoxLayout *horizontalLayout;
    QSpacerItem *horizontalSpacer;
    QPushButton *pushButton_prev;
    QPushButton *pushButton_next;
    QSpacerItem *horizontalSpacer_2;

    void setupUi(QDialog *Dialog)
    {
        if (Dialog->objectName().isEmpty())
            Dialog->setObjectName(QStringLiteral("Dialog"));
        Dialog->resize(731, 525);
        verticalLayout = new QVBoxLayout(Dialog);
        verticalLayout->setSpacing(6);
        verticalLayout->setContentsMargins(11, 11, 11, 11);
        verticalLayout->setObjectName(QStringLiteral("verticalLayout"));
        frame_2 = new QFrame(Dialog);
        frame_2->setObjectName(QStringLiteral("frame_2"));
        QSizePolicy sizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(frame_2->sizePolicy().hasHeightForWidth());
        frame_2->setSizePolicy(sizePolicy);
        frame_2->setMinimumSize(QSize(0, 50));
        frame_2->setFrameShape(QFrame::StyledPanel);
        frame_2->setFrameShadow(QFrame::Raised);
        horizontalLayout_6 = new QHBoxLayout(frame_2);
        horizontalLayout_6->setSpacing(6);
        horizontalLayout_6->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_6->setObjectName(QStringLiteral("horizontalLayout_6"));
        pushButton_addPage = new QPushButton(frame_2);
        pushButton_addPage->setObjectName(QStringLiteral("pushButton_addPage"));
        pushButton_addPage->setAutoDefault(false);

        horizontalLayout_6->addWidget(pushButton_addPage);

        pushButton_insertPage = new QPushButton(frame_2);
        pushButton_insertPage->setObjectName(QStringLiteral("pushButton_insertPage"));
        pushButton_insertPage->setAutoDefault(false);

        horizontalLayout_6->addWidget(pushButton_insertPage);

        textEdit_pageText = new QTextEdit(frame_2);
        textEdit_pageText->setObjectName(QStringLiteral("textEdit_pageText"));
        QSizePolicy sizePolicy1(QSizePolicy::Fixed, QSizePolicy::Fixed);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(textEdit_pageText->sizePolicy().hasHeightForWidth());
        textEdit_pageText->setSizePolicy(sizePolicy1);
        textEdit_pageText->setMaximumSize(QSize(200, 30));
        textEdit_pageText->setInputMethodHints(Qt::ImhNone);
        textEdit_pageText->setLineWrapMode(QTextEdit::NoWrap);
        textEdit_pageText->setAcceptRichText(false);

        horizontalLayout_6->addWidget(textEdit_pageText);

        line = new QFrame(frame_2);
        line->setObjectName(QStringLiteral("line"));
        line->setFrameShape(QFrame::VLine);
        line->setFrameShadow(QFrame::Sunken);

        horizontalLayout_6->addWidget(line);

        pushButton_removePage = new QPushButton(frame_2);
        pushButton_removePage->setObjectName(QStringLiteral("pushButton_removePage"));
        pushButton_removePage->setAutoDefault(false);

        horizontalLayout_6->addWidget(pushButton_removePage);

        horizontalSpacer_3 = new QSpacerItem(98, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout_6->addItem(horizontalSpacer_3);

        label_4 = new QLabel(frame_2);
        label_4->setObjectName(QStringLiteral("label_4"));

        horizontalLayout_6->addWidget(label_4);

        label_pageIndicator = new QLabel(frame_2);
        label_pageIndicator->setObjectName(QStringLiteral("label_pageIndicator"));
        QFont font;
        font.setBold(true);
        font.setWeight(75);
        label_pageIndicator->setFont(font);

        horizontalLayout_6->addWidget(label_pageIndicator);


        verticalLayout->addWidget(frame_2);

        frame = new QFrame(Dialog);
        frame->setObjectName(QStringLiteral("frame"));
        QSizePolicy sizePolicy2(QSizePolicy::Preferred, QSizePolicy::Expanding);
        sizePolicy2.setHorizontalStretch(0);
        sizePolicy2.setVerticalStretch(0);
        sizePolicy2.setHeightForWidth(frame->sizePolicy().hasHeightForWidth());
        frame->setSizePolicy(sizePolicy2);
        frame->setFrameShape(QFrame::StyledPanel);
        frame->setFrameShadow(QFrame::Raised);
        horizontalLayout_2 = new QHBoxLayout(frame);
        horizontalLayout_2->setSpacing(6);
        horizontalLayout_2->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_2->setObjectName(QStringLiteral("horizontalLayout_2"));
        stackedWidget = new QStackedWidget(frame);
        stackedWidget->setObjectName(QStringLiteral("stackedWidget"));
        page = new QWidget();
        page->setObjectName(QStringLiteral("page"));
        horizontalLayout_3 = new QHBoxLayout(page);
        horizontalLayout_3->setSpacing(6);
        horizontalLayout_3->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_3->setObjectName(QStringLiteral("horizontalLayout_3"));
        label = new QLabel(page);
        label->setObjectName(QStringLiteral("label"));
        QFont font1;
        font1.setPointSize(50);
        font1.setBold(true);
        font1.setWeight(75);
        label->setFont(font1);
        label->setAlignment(Qt::AlignCenter);

        horizontalLayout_3->addWidget(label);

        stackedWidget->addWidget(page);
        page_3 = new OriginalWidget();
        page_3->setObjectName(QStringLiteral("page_3"));
        stackedWidget->addWidget(page_3);

        horizontalLayout_2->addWidget(stackedWidget);


        verticalLayout->addWidget(frame);

        frame_3 = new QFrame(Dialog);
        frame_3->setObjectName(QStringLiteral("frame_3"));
        QSizePolicy sizePolicy3(QSizePolicy::Preferred, QSizePolicy::Fixed);
        sizePolicy3.setHorizontalStretch(0);
        sizePolicy3.setVerticalStretch(0);
        sizePolicy3.setHeightForWidth(frame_3->sizePolicy().hasHeightForWidth());
        frame_3->setSizePolicy(sizePolicy3);
        frame_3->setMinimumSize(QSize(0, 50));
        frame_3->setFrameShape(QFrame::StyledPanel);
        frame_3->setFrameShadow(QFrame::Raised);
        horizontalLayout = new QHBoxLayout(frame_3);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QStringLiteral("horizontalLayout"));
        horizontalSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer);

        pushButton_prev = new QPushButton(frame_3);
        pushButton_prev->setObjectName(QStringLiteral("pushButton_prev"));
        pushButton_prev->setEnabled(true);
        pushButton_prev->setAutoDefault(false);

        horizontalLayout->addWidget(pushButton_prev);

        pushButton_next = new QPushButton(frame_3);
        pushButton_next->setObjectName(QStringLiteral("pushButton_next"));
        pushButton_next->setAutoDefault(false);

        horizontalLayout->addWidget(pushButton_next);

        horizontalSpacer_2 = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);

        horizontalLayout->addItem(horizontalSpacer_2);


        verticalLayout->addWidget(frame_3);


        retranslateUi(Dialog);

        stackedWidget->setCurrentIndex(0);


        QMetaObject::connectSlotsByName(Dialog);
    } // setupUi

    void retranslateUi(QDialog *Dialog)
    {
        Dialog->setWindowTitle(QApplication::translate("Dialog", "QStackedWidget Sample", 0));
        pushButton_addPage->setText(QApplication::translate("Dialog", "add", 0));
        pushButton_insertPage->setText(QApplication::translate("Dialog", "insert", 0));
        textEdit_pageText->setPlaceholderText(QApplication::translate("Dialog", "Input text for a new page.", 0));
        pushButton_removePage->setText(QApplication::translate("Dialog", "remove", 0));
        label_4->setText(QApplication::translate("Dialog", "Current Page : ", 0));
        label_pageIndicator->setText(QApplication::translate("Dialog", "0", 0));
        label->setText(QApplication::translate("Dialog", "First Page", 0));
        pushButton_prev->setText(QApplication::translate("Dialog", "<< Prev", 0));
        pushButton_next->setText(QApplication::translate("Dialog", "Next >>", 0));
    } // retranslateUi

};

namespace Ui {
    class Dialog: public Ui_Dialog {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_DIALOG_H
