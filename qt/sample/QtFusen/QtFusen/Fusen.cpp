///#include <QtGui>
#include <QtWidgets>
//#include <QVariant>		//	for qVariantValue()
#include "Fusen.h"
#include <QDebug>


/*
11/06/08
	Qt::WindowFlags に Qt::Popup を指定すると、フレームの無いポップアップになる
	が、フォーカスを失うと？直ぐに消えてしまう
*/
Fusen::Fusen(bool topMost, QWidget *parent)
	: QWidget(parent,
				(topMost ? Qt::WindowStaysOnTopHint : Qt::Widget)		//	最前面表示
				| Qt::FramelessWindowHint)		//	フレーム無し
{
	m_mousePressed = false;
	m_mouseEntered = false;
	//setWindowIcon(QIcon(QPixmap(":QtFusen/Resources/rect2985.png")));
	//	背景を透明に
	setAttribute(Qt::WA_TranslucentBackground);
#if 0
	QPalette thisplt = this->palette();
	thisplt.setBrush(QPalette::Base, QBrush(Qt::transparent));
	this->setPalette(thisplt);
#endif

	m_firstColor = QColor("lightyellow");
	m_secondColor = QColor("greenyellow");
	m_gradType = Grad_None;
	m_editor = new QPlainTextEdit;
	m_editor->setFrameStyle(QFrame::NoFrame);	//	フレーム無し
	//	背景を透明に
	QPalette plt = m_editor->palette();
	plt.setBrush(QPalette::Base, QBrush(Qt::transparent));
	m_editor->setPalette(plt);
	//m_editor->setAutoFillBackground(false);
	//m_editor->setBackgroundVisible(false);
	////m_editor->setFont(QFont("Arial", 16));

	//##m_editor->setAutoFillBackground(false);
	//##m_editor->setAttribute(Qt::WA_OpaquePaintEvent);

	//m_editor->setTextBackgroundColor(QColor("pink"));	//テキストのある場所だけ背景色になる
#if 1
	QWidget *titleBar = new QWidget();
		QHBoxLayout *tbLayout = new QHBoxLayout();				//	タイトルバー
		tbLayout->addWidget(m_newFusen = new QToolButton());		//	＋
			m_newFusen->setText(QChar(0xff0b));
			m_newFusen->setToolTip(tr("Create a new Fusen"));
			connect(m_newFusen, SIGNAL(clicked()), this, SLOT(newFusen()));
		//tbLayout->addWidget(new QToolButton());		//	＿
		tbLayout->addStretch();
		tbLayout->addWidget(m_removeFusen = new QToolButton());		//	×
			m_removeFusen->setText(QChar(0x00d7));
			m_removeFusen->setToolTip(tr("Remove this Fusen"));
			connect(m_removeFusen, SIGNAL(clicked()), this, SLOT(close()));
		tbLayout->setContentsMargins(1, 1, 1, 1);
		titleBar->setLayout(tbLayout);
#else
#if 1
	QHBoxLayout *titleBar = new QHBoxLayout();				//	タイトルバー
		titleBar->addWidget(new QToolButton());		//	＋
		titleBar->addWidget(new QToolButton());		//	＿
		titleBar->addStretch();
		titleBar->addWidget(new QToolButton());		//	×
		titleBar->setContentsMargins(1, 1, 1, 1);
#else
		titleBar->addWidget(new QLabel(QChar(0xff0b)));		//	＋
		titleBar->addWidget(new QLabel(QChar(0xff3f)));		//	＿
		titleBar->addStretch();
		titleBar->addWidget(new QLabel(QChar(0x00d7)));		//	×
#endif
#endif
#if 1
	QHBoxLayout *statusBar = new QHBoxLayout();				//	ステータスバー
		statusBar->addSpacing(4);
		statusBar->addWidget(m_fontCB = new QFontComboBox());		//	フォントコンボボックス
		connect(m_fontCB, SIGNAL(currentFontChanged(const QFont &)),
				this, SLOT(setFont(const QFont &)));
		statusBar->addWidget(m_fontSizeSB = new QSpinBox());		//	フォントサイズ
		m_fontSizeSB->setRange(1, 100);
		connect(m_fontSizeSB, SIGNAL(valueChanged(int)),
				this, SLOT(setFontSize(int)));
#if 0
		statusBar->addWidget(m_fontSizeTB = new QToolButton());		//	フォントサイズ
		m_fontSizeTB->setToolTip(tr("Select font size"));
		m_fontSizeTB->setMenu(createFontSizeMenu());
#endif
		statusBar->addWidget(m_firstColorTB = new QToolButton());		//	カラーその１
		m_firstColorTB->setText(tr("color"));
		m_firstColorTB->setToolTip(tr("Select first color"));
		connect(m_firstColorTB, SIGNAL(clicked()), this, SLOT(selectFirstColor()));
		statusBar->addWidget(m_secondColorTB = new QToolButton());		//	カラーその２
		m_secondColorTB->setText(tr("2'd color"));
		m_secondColorTB->setToolTip(tr("Select second color"));
		connect(m_secondColorTB, SIGNAL(clicked()), this, SLOT(selectSecondColor()));
		statusBar->addWidget(m_gradientTB = new QToolButton());
		m_gradientTB->setText(tr("gradient"));
		m_gradientTB->setToolTip(tr("Select color Gradient"));
	    m_gradientTB->setPopupMode(QToolButton::MenuButtonPopup);
		m_gradientTB->setMenu(createGradientMenu());
		statusBar->addStretch();
		statusBar->addWidget(new QSizeGrip(this));
	QVBoxLayout *vLayout = new QVBoxLayout();
		vLayout->addWidget(titleBar);
		//vLayout->addLayout(titleBar);
		vLayout->addWidget(m_editor);
		vLayout->addLayout(statusBar);
#else
	QVBoxLayout *vLayout = new QVBoxLayout();
		vLayout->addWidget(m_editor);
#endif
	setLayout(vLayout);
	//	left, top, right, bottom
	vLayout->setContentsMargins(2, 4,
								2 + SHADOW_WIDTH, 4 + SHADOW_WIDTH);
	//setContentsMargins(0, 0, 0, 0);

	//m_editor->installEventFilter(this);
	m_editor->viewport()->installEventFilter(this);
	//hideStatusBarWidgets();

#if 0
	//	ドロップシャドウを指定してみる
	QGraphicsDropShadowEffect* shadowEffect = new QGraphicsDropShadowEffect;
	shadowEffect->setBlurRadius( 12 );
	shadowEffect->setOffset( 6, 6 );
	shadowEffect->setColor( QColor( 20, 20, 20 ) );
	setGraphicsEffect(shadowEffect);
#endif
	setMouseTracking(true);
}

Fusen::~Fusen()
{
	//emit onClose(this);
}

QMenu *Fusen::createFontSizeMenu()
{
	QMenu *menu = new QMenu();
	return menu;
}
void Fusen::setFont(const QFont &font)
{
	m_font = font;
	m_fontCB->setCurrentFont(font);
	m_fontSizeSB->setValue(font.pointSize());
	m_editor->setFont(font);
}
void Fusen::setFontSize(int sz)
{
	m_fontSize = sz;
	setFont(QFont(m_font.family(), sz));
}
QMenu *Fusen::createGradientMenu()
{
    QList<int> gradients;
    gradients << Grad_None
    			<< Grad_Top
    			<< Grad_TopRight
    			<< Grad_Right
    			<< Grad_BottomRight
    			<< Grad_Bottom
    			<< Grad_BottomLeft
    			<< Grad_Left
    			<< Grad_TopLeft;
    QStringList names;
    names << tr("no gradation")
    	<< tr("top")
    	<< tr("top-right")
    	<< tr("right")
    	<< tr("bottom-right")
    	<< tr("bottom")
    	<< tr("bottom-left")
    	<< tr("left")
    	<< tr("top-left");
    m_gradientMenu = new QMenu(this);
    for (int i = 0; i < gradients.count(); ++i) {
        QAction *action = new QAction(names.at(i), this);
        m_gradMenuActions.push_back(action);
        action->setData(gradients.at(i));
        action->setIcon(createGradientIcon(gradients.at(i)));
        connect(action, SIGNAL(triggered()), this, SLOT(gradientChanged()));
        m_gradientMenu->addAction(action);
        if ( i == 0 ) {
        	//	ドロップダウンではなく、その左のアイコンをクリックした場合のアクションを設定
            m_gradientMenu->setDefaultAction(action);
        }
    }
    return m_gradientMenu;
}
QBrush createGradientBrush(int grad, QColor color1, QColor color2, const QRectF &r)
{
	if( grad == Fusen::Grad_None )
		return QBrush(color1);
    QLinearGradient lg;
	lg.setColorAt(0, color1);
	lg.setColorAt(1, color2);
    switch( grad ) {
	case Fusen::Grad_Top:
		lg.setStart(r.topLeft());
		lg.setFinalStop(r.bottomLeft());
    	break;
	case Fusen::Grad_Bottom:
		lg.setStart(r.bottomLeft());
		lg.setFinalStop(r.topLeft());
    	break;
	case Fusen::Grad_Left:
		lg.setStart(r.topLeft());
		lg.setFinalStop(r.topRight());
    	break;
	case Fusen::Grad_Right:
		lg.setStart(r.topRight());
		lg.setFinalStop(r.topLeft());
    	break;
	case Fusen::Grad_TopLeft:
		lg.setStart(r.topLeft());
		lg.setFinalStop(r.bottomRight());
    	break;
	case Fusen::Grad_TopRight:
		lg.setStart(r.topRight());
		lg.setFinalStop(r.bottomLeft());
    	break;
	case Fusen::Grad_BottomLeft:
		lg.setStart(r.bottomLeft());
		lg.setFinalStop(r.topRight());
    	break;
	case Fusen::Grad_BottomRight:
		lg.setStart(r.bottomRight());
		lg.setFinalStop(r.topLeft());
    	break;
    }
	return QBrush(lg);
}
QIcon Fusen::createGradientIcon(int grad)
{
	QRect r(0, 0, 20, 20);
    QPixmap pixmap(r.width(), r.height());
    QPainter painter(&pixmap);
    painter.setPen(Qt::NoPen);
    QBrush br = createGradientBrush(grad, Qt::lightGray, Qt::white, r);
    painter.fillRect(r, br);
    return QIcon(pixmap);
}
void Fusen::gradientChanged()
{
    gradientAction = qobject_cast<QAction *>(sender());
	///setGradType(qVariantValue<int>(gradientAction->data()));
	setGradType(gradientAction->data().toInt());
#if 0
    gradientAction = qobject_cast<QAction *>(sender());
    m_gradientMenu->setDefaultAction(gradientAction);
    m_gradType = qVariantValue<int>(gradientAction->data());
	update();
#endif
}
void Fusen::setGradType(int grad)
{
	m_gradType = grad;
	if( grad >= 0 && grad < m_gradMenuActions.size() )
	    m_gradientMenu->setDefaultAction(m_gradMenuActions[grad]);
	update();
}

void Fusen::setTopMost(bool b)
{
	QPoint p = pos();
	QSize sz = size();
	if( b ) {
		setWindowFlags( Qt::WindowStaysOnTopHint | Qt::FramelessWindowHint );
	} else {
		setWindowFlags( Qt::FramelessWindowHint );
	}
	move(p);
	resize(sz);
	show();
}

bool Fusen::eventFilter(QObject *obj, QEvent *event)
{
	if( obj == m_editor->viewport() ) {
		//qDebug() << "Fusen::eventFilter() type = " << event->type();
#if 0
		if( event->type() == QEvent::Paint ) {
			qDebug() << "QEvent::Paint";
			QPainter painter(m_editor->viewport());
			QRect r = m_editor->viewport()->rect();
			QLinearGradient linearGrad(QPointF(0, 0), QPointF(0, r.height())); 
			linearGrad.setColorAt(0, QColor("lightyellow"));
			linearGrad.setColorAt(1, QColor("greenyellow"));
			painter.setPen(Qt::transparent);
			painter.setBrush(QBrush(linearGrad));
			//painter.setBrush(QColor("lightyellow"));
			painter.drawRect(r);
			//return true;
		}
#endif
	}
	return QObject::eventFilter(obj, event);
}

void Fusen::paintEvent(QPaintEvent * event)
{
	QPainter painter(this);
	QRect r = rect();
#if 1
#if 0
	painter.setPen(Qt::transparent);
	painter.setBrush(createGradientBrush(m_gradType, m_firstColor, m_secondColor, r));
	painter.drawRect(r);
#else
	painter.setPen(Qt::transparent);
	painter.setBrush(QColor(128, 128, 128, 0x80));
	//painter.setBrush(QBrush(QColor(128, 128, 128, 0xc0), Qt::Dense4Pattern));
	r.setTop( r.top() + SHADOW_WIDTH );
	r.setLeft( r.left() + SHADOW_WIDTH );
	painter.drawRect(r);
	r = rect();
	r.setRight( r.right() - SHADOW_WIDTH );
	r.setBottom( r.bottom() - SHADOW_WIDTH );
	//painter.setPen(Qt::transparent);
	painter.setBrush(createGradientBrush(m_gradType, m_firstColor, m_secondColor, r));
	painter.drawRect(r);
#endif
#else
#if 1
	QLinearGradient linearGrad(QPointF(0, 0), QPointF(0, r.height())); 
	linearGrad.setColorAt(0, m_firstColor /*QColor("lightyellow")*/);
	linearGrad.setColorAt(1, m_secondColor /*QColor("greenyellow")*/);
	painter.setPen(Qt::transparent);
	painter.setBrush(QBrush(linearGrad));
	painter.drawRect(r);
#else
	painter.setBrush(QColor("pink"));
	painter.drawRect(rect());
#endif
#endif
}
void Fusen::showStatusBarWidgets()
{
	m_fontCB->show();
	m_fontSizeSB->show();
	m_firstColorTB->show();
	m_secondColorTB->show();
	m_gradientTB->show();
}
void Fusen::hideStatusBarWidgets()
{
#if 1
	m_fontCB->hide();
	m_fontSizeSB->hide();
	m_firstColorTB->hide();
	m_secondColorTB->hide();
	m_gradientTB->hide();
#endif
}
void Fusen::enterEvent ( QEvent * event )
{
	m_mouseEntered = true;
	showStatusBarWidgets();
	emit mouseEntered(this);
}
void Fusen::leaveEvent ( QEvent * event )
{
	m_mouseEntered = false;
	QPoint p = QCursor::pos();
	QRect r = geometry();
	if( !r.contains(p) )
		hideStatusBarWidgets();
}
void Fusen::mousePressEvent(QMouseEvent * event)
{
	m_startWinPos = pos();
	m_startMousePos = QCursor::pos();
	m_mousePressed = true;
	grabMouse();
}
void Fusen::mouseReleaseEvent(QMouseEvent * event)
{
	m_mousePressed = false;
	releaseMouse();
}
void Fusen::mouseMoveEvent(QMouseEvent * event)
{
	if( m_mousePressed )
		move(m_startWinPos + QCursor::pos() - m_startMousePos);
#if 0
	if( m_mouseEntered ) {
		QPoint p = QCursor::pos();
		QRect r = geometry();
		if( !r.contains(p) ) {
			hideStatusBarWidgets();
			m_mouseEntered = false;
		}
	}
#endif
}
void Fusen::closeEvent(QCloseEvent * event)
{
	event->accept();
	emit onClose(this);
}

QString Fusen::text() const
{
	return m_editor->toPlainText();
}
void Fusen::setText(const QString &text)
{
	m_editor->setPlainText(text);
}

void Fusen::selectColor(QColor &color)
{
	emit toOpenDialog();
	QColor c = QColorDialog::getColor(color);
	emit closedDialog();
	if( c.isValid() )
		color = c;
#if 0
	const bool tm = (windowFlags() & Qt::WindowStaysOnTopHint) != 0;
	if( tm )
		setTopMost(false);
	QColor c = QColorDialog::getColor(color);
	//setTopMost(tm);
	if( tm )
		setTopMost(tm);
	if( !c.isValid() ) return;
	color = c;
#endif
}
void Fusen::selectFirstColor()
{
	selectColor(m_firstColor);
}
void Fusen::selectSecondColor()
{
	selectColor(m_secondColor);
}
