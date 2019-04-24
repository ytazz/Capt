#include <QtGui>
#include "View.h"

View::View(QWidget *parent)
	: QGraphicsView(parent)
{
	m_rotateAngle = 0;
}

View::~View()
{
}
void View::wheelEvent ( QWheelEvent * event )
{
	const bool ctrl = (event->modifiers() & Qt::ControlModifier) != 0;
	const bool shift = (event->modifiers() & Qt::ShiftModifier) != 0;
	const bool alt = (event->modifiers() & Qt::AltModifier) != 0;
	if( !alt && ctrl && !shift ) {
		//qDebug() << event->delta();
		const qreal s = event->delta() > 0 ? 1.05 : 1/1.05;
		viewScale(s);
		return;
	}
	if( !alt && !ctrl && shift ) {		//	Zé≤âÒÇËÇÃâÒì]
		m_rotateAngle = event->delta() > 0 ? 5 : -5;
		rotate(m_rotateAngle);
		//showMessage(tr("rotate %1 degrees").arg(m_rotateAngle));
		return;
	}
#if 0
	if( !alt && ctrl && shift ) {		//	Yé≤âÒÇËÇÃâÒì]
		m_rotateAngle = event->delta() > 0 ? 5 : -5;
		QTransform tf = transform();
		tf.rotate(m_rotateAngle, Qt::YAxis);
		setTransform(tf);
		return;
	}
	if( alt && shift && !ctrl ) {		//	Xé≤âÒÇËÇÃâÒì]
		m_rotateAngle = event->delta() > 0 ? 5 : -5;
		QTransform tf = transform();
		tf.rotate(m_rotateAngle, Qt::XAxis);
		setTransform(tf);
		return;
	}
#endif
	QGraphicsView::wheelEvent( event );
	//event->ignore();
}
//	zoomIn/OutÅiåªî{ó¶Çsî{Åj
void View::viewScale(qreal s)
{
	scale(s, s);
	//qDebug() << transform();
	QTransform tf = transform();
	const qreal m11 = tf.m11();
	showMessage(tr("view scale:%1%").arg(m11*100), 5*1000);
}
void View::mouseDoubleClickEvent ( QMouseEvent * mouseEvent )
{
	qDebug() << "View::mouseDoubleClickEvent()";
	QGraphicsView::mouseDoubleClickEvent(mouseEvent);
}
