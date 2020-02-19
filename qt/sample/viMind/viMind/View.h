#ifndef VIEW_H
#define VIEW_H

#include <QGraphicsView>

class View : public QGraphicsView
{
	Q_OBJECT

public:
	View(QWidget *parent = 0);
	~View();

protected:
	void	mouseDoubleClickEvent ( QMouseEvent * mouseEvent );
	void	wheelEvent ( QWheelEvent * event );
	void	viewScale(qreal s);		//	zoomIn/OutÅiåªî{ó¶Çsî{Åj

signals:
	void	showMessage(const QString &, int = 0);

private:
	qreal	m_rotateAngle;
};

#endif // VIEW_H
