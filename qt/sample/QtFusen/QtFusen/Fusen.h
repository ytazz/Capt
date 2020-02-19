#ifndef FUSEN_H
#define FUSEN_H

#include <QWidget>

class QFontComboBox;
class QToolButton;
class QSpinBox;
class QMenu;
class QFont;
class QAction;

class Fusen : public QWidget
{
	Q_OBJECT

public:
	enum {
		Grad_None = 0,
		Grad_Top,
		Grad_TopRight,
		Grad_Right,
		Grad_BottomRight,
		Grad_Bottom,
		Grad_BottomLeft,
		Grad_Left,
		Grad_TopLeft,
	};
	enum {
		SHADOW_WIDTH = 4,
	};

public:
	Fusen(bool topMost, QWidget *parent = 0);
	~Fusen();

public:
	QString	text() const;
	int		fontSize() const { return m_fontSize; }
	QFont	font() const { return m_font; }
	QColor	firstColor() const { return m_firstColor; }
	QColor	secondColor() const { return m_secondColor; }
	int		gradType() const { return m_gradType; }

public:
	void	setText(const QString &);
	//void	setFontSize(int sz) { m_fontSize = sz; }
	void	setFirstColor(const QColor &color) { m_firstColor = color; }
	void	setSecondColor(const QColor &color) { m_secondColor = color; }
	void	setGradType(int);
	void	setTopMost(bool);
	void	hideStatusBarWidgets();
	void	showStatusBarWidgets();

public slots:
	void	setFont(const QFont &);
	void	setFontSize(int);

protected:
	void	paintEvent(QPaintEvent * event);
	void	enterEvent ( QEvent * event );
	void	leaveEvent ( QEvent * event );
	void	mousePressEvent(QMouseEvent * event);
	void	mouseReleaseEvent(QMouseEvent * event);
	void	mouseMoveEvent(QMouseEvent * event);
	void	closeEvent(QCloseEvent * event);
	bool	eventFilter(QObject *obj, QEvent *event);
	void	selectColor(QColor &);
	QMenu	*createFontSizeMenu();
    QMenu	*createGradientMenu();
	QIcon	createGradientIcon(int grad);

protected slots:
	void	newFusen()	{ emit newFusenClicked(); }
	//void	remove();
	void	selectFirstColor();			//	1st color 選択
	void	selectSecondColor();		//	2'd color 選択
	void	gradientChanged();			//	グラデーションタイプ変更

signals:
	void	newFusenClicked();
	void	toOpenDialog();
	void	closedDialog();
	void	onClose(Fusen *);
	void	mouseEntered(Fusen *);

private:
	bool			m_mousePressed;
	bool			m_mouseEntered;
	QToolButton		*m_newFusen;
	QToolButton		*m_removeFusen;
	QFontComboBox	*m_fontCB;
	QSpinBox		*m_fontSizeSB;
	//QToolButton		*m_fontSizeTB;
	int				m_fontSize;
	QFont			m_font;

	int				m_gradType;				//	グラデーションタイプ
	QColor			m_firstColor;			//	1st color 
	QColor			m_secondColor;			//	2'd color
	QToolButton		*m_firstColorTB;		//	1st color ツールボタン
	QToolButton		*m_secondColorTB;		//	2'd color ツールボタン
	QToolButton		*m_gradientTB;			//	グラデーション ツールボタン
	QAction			*gradientAction;		//	デフォルトアクション
	QMenu			*m_gradientMenu;		//	グラデーション メニュー
	QVector<QAction *>	m_gradMenuActions;	//	各グラデーションメニュー

	class QPlainTextEdit	*m_editor;
	QPoint	m_startWinPos;
	QPoint	m_startMousePos;
};

#endif // FUSEN_H
