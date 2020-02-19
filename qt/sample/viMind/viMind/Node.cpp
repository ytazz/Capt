#include <QtGui>
#include <QtWidgets>
#include "Node.h"
#include "Scene.h"
#include <algorithm>

Node::Node(const QString &text, Node *parent)
	: QGraphicsTextItem(text, parent)
	, m_branch(0)
	, m_expanded(true)
{
	if( !parent )	//	ルートノードの場合
		setFlags( QGraphicsItem::ItemIsSelectable 		//	選択可能
					| QGraphicsItem::ItemIsMovable );	//	移動可能
	else {
		setFlags( QGraphicsItem::ItemIsSelectable );	//	選択可能
		parent->m_children.push_back(this);				//	親ノードの子ノードリストに追加
		//Node *pNode = dynamic_cast<Node *>(parent);
		//pNode->m_children.push_back(this);				//	親ノードが指定されている場合
	}
	setCursor(Qt::IBeamCursor);			//	アイビームカーソル
}

Node::~Node()
{

}
bool Node::isRootNode() const
{
	return parentNode() == 0;
}
Node *Node::parentNode() const
{
	return dynamic_cast<Node *>(parentItem());
}
Node *Node::firstChildNode() const
{
	if( m_children.isEmpty() ) return 0;
	return m_children[0];
}
int Node::childIndexOf(Node *node) const
{
	return m_children.indexOf(node);
}
//	子ノード全体の高さを返す
qreal Node::childrenHeight() const
{
	if( m_children.isEmpty() ) return 0;
	qreal h = -CC_SPACE;
	foreach(const Node *node, m_children) {
		if( node->isVisible() ) {
			h += node->treeHeight();	//	子ツリー幅
			h += CC_SPACE;		//	子ノード間隔
		}
	}
	return h;
}
//	子ノードを含めた全体の高さを返す
qreal Node::treeHeight() const
{
	//	ツリー高さは子ノードの合計と自分の高さの最大値
	return qMax(childrenHeight(), boundingRect().height());
}
//	ノードが重ならないように子ノード位置を決め、ツリーの高さを返す
qreal Node::layoutChildren()
{
	const QRectF br = boundingRect();
	if( m_children.isEmpty() )		//	子ノードが無い場合
		return br.height();
	const qreal x = br.right() + PC_SPACE;	//	子ノードのX座標
	const qreal cht = childrenHeight();	//	子ノード全体の高さ
	qreal y = br.center().y() - cht / 2;	//	最初の子ノード位置
	foreach(Node *node, m_children) {
		if( node->isVisible() ) {
			const qreal th = node->layoutChildren();			//	子ノードをレイアウト
			node->setPos(x, y + th / 2 - node->boundingRect().center().y());
			node->setBranch();		//	枝をセットアップ
			y += th + CC_SPACE;
		}
	}
	return qMax(cht, br.height());
}
#if 0
//	ノードが重ならないように子ノード位置を決める
void Node::layoutChildren()
{
	if( m_children.isEmpty() ) return;
	const qreal x = boundingRect().width() + PC_SPACE;	//	子ノード x 座標
	const qreal cht = childrenHeight();	//	子ノード全体の高さ
	qreal y = boundingRect().center().y() - cht / 2;	//	最初の子ノード位置
	foreach(Node *node, m_children) {
		node->layoutChildren();			//	子ノードをレイアウト
		const qreal th = node->treeHeight();
		node->setPos(x, y + th / 2 - node->boundingRect().height() / 2);
		y += th + CC_SPACE;
	}
}
#endif

void Node::setBranch()
{
	Node *pNode = parentNode();
	if( pNode == 0 ) return;
	if( m_branch == 0 )
		m_branch = new QGraphicsPathItem(this);
	const QRectF pbr = pNode->boundingRect();
	QPointF pc(pbr.right() - pos().x(), pbr.center().y() - pos().y());	//	親ノード接続点
	QPointF cc(0, boundingRect().center().y());		//	子ノード接続点
	QPainterPath path;
	path.moveTo(pc);			//	親ノード接続点位置
	const QPointF c1((pc.x() + cc.x()) / 2, pc.y());		//	制御点 1
	const QPointF c2(pc.x(), cc.y());					//	制御点 2
	path.cubicTo(c1, c2, cc);
	m_branch->setPath(path);
#if 0
	m_branch->setLine(pbr.right() - pos().x(),			//	親ノード右端
						pbr.center().y() - pos().y(),
						0,
						boundingRect().center().y());
#endif
}
Node *Node::prevNode() const
{
	Node *p = parentNode();
	if( !p ) return 0;
	QList<Node *>::iterator itr = std::find(p->m_children.begin(), p->m_children.end(), this);
	if( itr == p->m_children.end() || itr == p->m_children.begin() )
		return 0;
	return *--itr;
}
Node *Node::nextNode() const
{
	Node *p = parentNode();
	if( !p ) return 0;
	QList<Node *>::iterator itr = std::find(p->m_children.begin(), p->m_children.end(), this);
	if( itr == p->m_children.end() || ++itr == p->m_children.end() )
		return 0;
	return *itr;
}
//	指定ノードを子ノードリストから削除
void Node::removeChildNode(Node *node)
{
	QList<Node *>::iterator itr = std::find(m_children.begin(), m_children.end(), node);
	if( itr != m_children.end() )
		m_children.erase(itr);
}
void Node::makeChildrenVisible(bool b)
{
	foreach(Node *node, m_children) {
		node->setVisible(b);
		//node->makeChildrenVisible(b);
	}
}
void Node::insert(int ix, Node *node)
{
	m_children.insert(ix, node);
	node->setParentItem(this);
}
Node *Node::addNode(Node *node)
{
	node->setParentItem(this);
	m_children.push_back(QPointer<Node>(node));
	return node;
}
bool Node::moveUp()
{
	Node *pNode = parentNode();
	if( !pNode ) return false;
	QList<Node *>::iterator itr = std::find(pNode->m_children.begin(), pNode->m_children.end(), this);
	if( itr == pNode->m_children.begin() || itr == pNode->m_children.end() )
		return false;
	const int ix = itr - pNode->m_children.begin();
	pNode->m_children.swap(ix - 1, ix);
#if 0
	itr = pNode->m_children.erase(itr);		//	erase() は削除した次の要素へのイテレータを返す
	--itr;
	pNode->m_children.insert(itr, this);
#endif
#if 0
	for(QList<QPointer<Node> >::iterator itr = pNode->m_children.begin(),
											iend = pNode->m_children.end();
		itr != iend;
		++itr)
	{
		if( *itr == this ) {
			itr = pNode->m_children.erase(itr);
			if( itr == pNode->m_children.begin() )
				itr = pNode->m_children.end();
			else
				--itr;
			pNode->m_children.insert(itr, this);
			this->setModifiedDT();
			break;
		}
	}
#endif
	return true;
}
bool Node::moveDown()
{
	Node *pNode = parentNode();
	if( !pNode ) return false;
	QList<Node *>::iterator itr = std::find(pNode->m_children.begin(), pNode->m_children.end(), this);
	if( itr == pNode->m_children.end() || itr + 1 == pNode->m_children.end() )
		return false;
	const int ix = itr - pNode->m_children.begin();
	pNode->m_children.swap(ix, ix + 1);
#if 0
	itr = pNode->m_children.erase(itr);		//	erase() は削除した次の要素へのイテレータを返す
	++itr;
	pNode->m_children.insert(itr, this);
#endif
#if 0
	for(QList<QPointer<Node> >::iterator itr = pNode->m_children.begin(),
											iend = pNode->m_children.end();
		itr != iend;
		++itr)
	{
		if( *itr == this ) {
			itr = pNode->m_children.erase(itr);
			if( itr == pNode->m_children.end() )
				itr = pNode->m_children.begin();
			else
				++itr;
			pNode->m_children.insert(itr, this);
			this->setModifiedDT();
			break;
		}
	}
#endif
	return true;
}
bool Node::levelUp()
{
	Node *pNode = parentNode();
	if( !pNode ) return false;		//	ルートノードの場合
	Node *ppNode = pNode->parentNode();
	if( !ppNode ) return false;		//	ルートノードの子供の場合
	QList<Node *>::iterator itr = std::find(pNode->m_children.begin(),
														pNode->m_children.end(),
														this);
	pNode->m_children.erase(itr);
	itr = std::find(ppNode->m_children.begin(),
					ppNode->m_children.end(),
					pNode);
	ppNode->m_children.insert(++itr, this);
	//m_parentNode = ppNode;
	setParentItem(ppNode);
	///this->setModifiedDT();
	return true;
}
bool Node::levelDown()
{
	Node *pNode = parentNode();
	if( !pNode ) return false;
	Node *dstNode = prevNode();
	if( !dstNode ) {		//	兄がいない場合
		dstNode = nextNode();	//	兄が無い場合は弟の下に移動
		if( !dstNode ) return false;	//	弟もいない場合
	}
	if( !dstNode->expanded() )
		dstNode->setExpanded(true);
	QList<Node *>::iterator itr = std::find(pNode->m_children.begin(),
														pNode->m_children.end(),
														this);
	pNode->m_children.erase(itr);
	dstNode->addNode(this);
	///this->setModifiedDT();
	return true;
}
bool Node::moveLeft()
{
	Node *pNode = parentNode();
	if( !pNode ) return false;		//	ルートノードの場合
#if 0
	Node *ppNode = pNode->parentNode();
	if( !ppNode && isRightSide() ) {	//	ルートノードの右側子供の場合
		setRightSideRecursive(LEFT_SIDE);
		moveToLastChild();
		updateLinkIconPosRecursive();
	} else if( !isRightSide() )
		levelDown();
	else
#endif
		levelUp();
	return this;
}
bool Node::moveRight()
{
	Node *pNode = parentNode();
	if( !pNode ) return false;		//	ルートノードの場合
#if 0
	Node *ppNode = pNode->parentNode();
	if( !ppNode && !isRightSide() ) {	//	ルートノードの左側子供の場合
		setRightSideRecursive(RIGHT_SIDE);
		moveToLastChild();
		updateLinkIconPosRecursive();
	} else if( !isRightSide() )
		levelUp();
	else
#endif
	return levelDown();
	//return this;
}
void Node::paint(QPainter * painter, const QStyleOptionGraphicsItem * option, QWidget * widget)
{
	if( isRootNode() ) {
		painter->setPen(Qt::black);
		painter->setBrush(Qt::transparent);
		painter->drawEllipse(boundingRect());
	}
	QGraphicsTextItem::paint(painter, option, widget);
}
QRectF Node::boundingRect() const
{
	QRectF br = QGraphicsTextItem::boundingRect();
	if( !isRootNode() ) {
		return br;
	}
	const qreal sqrt2 = sqrt(2.0);
	const qreal wd2 = br.width() * sqrt2;
	const qreal ht2 = br.height() * sqrt2;
	return QRectF((br.width() - wd2) / 2, (br.height() - ht2) / 2, wd2, ht2);		//	外接楕円
}
QString Node::toXmlText() const
{
	QString buffer = "<node TEXT=\"";
	//buffer += Qt::escape(toPlainText());
	buffer += toPlainText().toHtmlEscaped();
	buffer += "\" ";
	if( isRootNode() ) {
		const QPointF sp = scenePos();
		buffer += QString("SX=\"%1\" SY=\"%2\" ").arg(sp.x()).arg(sp.y());
	}
#if 0
	else if( !parent() ) {		//	ルート直下の子供の場合
		if( isRightSide() )
			buffer += "POSITION=\"right\" ";
		else
			buffer += "POSITION=\"left\" ";
	}
	buffer += QString("CREATED=\"%1\" ").arg(m_createdDT);
	buffer += QString("MODIFIED=\"%1\" ").arg(m_modifiedDT);
	switch( nodeStyle() ) {
	case FORK_STYLE: buffer += "STYLE=\"fork\" "; break;
	case RECT_STYLE: buffer += "STYLE=\"rect\" "; break;
	case ROUND_RECT_STYLE: buffer += "STYLE=\"roundRect\" "; break;
	case CIRCLE_RECT_STYLE: buffer += "STYLE=\"circleRect\" "; break;
	}
	buffer += QString("FONTNAME=\"%1\" FONTSIZE=\"%2\" ").
				arg(font().family()).arg(font().pointSize());
	if( !m_linkedFileName.isEmpty() ) {
		buffer += "LINK=\"" + m_linkedFileName + "\" ";
		if( !m_pixmapWidth.isEmpty() )
			buffer += "PXMWIDTH=\"" + m_pixmapWidth + "\" ";
	}
	if( isAlignColumn() )
		buffer += "ALIGNCOLUMN=\"true\" ";
	//const Qt::Alignment align = document()->defaultTextOption().alignment();
	if( m_textAlign == Qt::AlignHCenter )
		buffer += "ALIGN=\"center\" ";
	else if( m_textAlign == Qt::AlignRight )
		buffer += "ALIGN=\"right\" ";
	if( isNodeDownward() )
		buffer += "DOWNWARD=\"true\" ";
	if( expanded() )
		buffer += "EXPANDED=\"true\" ";
	else
		buffer += "EXPANDED=\"false\" ";
	buffer += "FILLCOLOR=\"" + m_fillColor.name() + "\" ";
#endif
	if( m_children.isEmpty() )
		buffer += "/>\n";
	else {
		buffer += ">\n";
		foreach(Node *ptr, m_children) {
			buffer += ptr->toXmlText();
		}
		buffer += "</node>\n";
	}
	return buffer;
}
#if 0
void Node::mousePressEvent ( QGraphicsSceneMouseEvent * event )
{
	QGraphicsTextItem::mousePressEvent(event);
}
#endif
void Node::mouseMoveEvent ( QGraphicsSceneMouseEvent * event )
{
	Scene *pScene = scene();
	if( !isRootNode() && pScene->mode() == Scene::COMMAND &&
				QLineF(event->scenePos(), event->buttonDownScenePos(Qt::LeftButton))
				.length() >= QApplication::startDragDistance() )
	{
		pScene->setDragSourceNode(this);
		///pScene->setDropTarget(0);
		///pScene->setDroppedToSameScene(false);
		QDrag *drag = new QDrag(event->widget());
		QMimeData *mime = new QMimeData;
		QString buffer = "<map>\n";
		buffer += toXmlText();
		buffer += "</map>\n";
		mime->setHtml(buffer);
		drag->setMimeData(mime);
		if( Qt::MoveAction == drag->exec() ) {
			//emit doDelete(this);
			pScene->doDelete(this);
#if 0
			if( !pScene->droppedToSameScene() ) {
				qDebug() << "Qt::MoveAction";
				emit doDelete(this);
			}
#endif
		}
		pScene->setDragSourceNode(0);
		///pScene->setDropTarget(0);
		return;
	}
	QGraphicsTextItem::mouseMoveEvent(event);
}
Scene *Node::scene()
{
	return dynamic_cast<Scene *>(QGraphicsItem::scene());
}
//	node が this の子孫ノードかどうかをチェック
bool Node::isDescendantNode(const Node *node) const
{
	foreach(Node *ptr, m_children) {
		if( ptr == node || ptr->isDescendantNode(node) ) {
			return true;
		}
	}
	return false;
}
void Node::mouseDoubleClickEvent ( QGraphicsSceneMouseEvent * mouseEvent )
{
	qDebug() << "Node::mouseDoubleClickEvent()";
	QGraphicsTextItem::mouseDoubleClickEvent(mouseEvent);
}
