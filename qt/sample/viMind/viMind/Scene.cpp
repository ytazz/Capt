#include <QtGui>
#include <QtWidgets>
#include <QtXml>
#include "Scene.h"
#include "Node.h"
#include "undoMgr.h"
#include "View.h"
#include <QDebug>

Scene::Scene(QObject *parent)
	: QGraphicsScene(parent)
	, m_rootNode(0)
	, m_mode(COMMAND)
	, m_editNode(0)
	, m_dragSourceNode(0)
	, m_dropTarget(0)
{
	m_undoStack = new QUndoStack;
	addItem(m_rootNode = addNode("Root"));
	//addItem(m_rootNode = new Node("Root"));
	setSelectedNode(m_rootNode);
#if 0
	new Node("Node1", m_rootNode);
	new Node("Node222", m_rootNode);
	new Node("Node3", m_rootNode);
	m_rootNode->layoutChildren();
#endif
#if 0
	Node *chNode1 = new Node("Node1", m_rootNode);
		new Node("node1-1", chNode1);
		new Node("node1-2", chNode1);
	Node *chNode2 = new Node("Node222", m_rootNode);
		new Node("node2-1", chNode2);
		new Node("node2-2", chNode2);
		new Node("Node3", m_rootNode);
	m_rootNode->layoutChildren();
#endif

#if 0
	Node *chNode = new Node(m_rootNode);
	chNode->setPlainText("Node");
	chNode->setPos(100, -10);
	Node *chNode2 = new Node(m_rootNode);
	chNode2->setPlainText("Node2");
	chNode2->setPos(100, 10);
#endif
}

Scene::~Scene()
{

}
void Scene::onContentsChanged()
{
	layoutAll();
	///if( m_selectedNode != 0 )
	///	m_selectedNode->ensureVisible();
   	///emit contentsChanged();
}
Node *Scene::selectedNode() const
{
	QList<QGraphicsItem *> lst = selectedItems();
	if( lst.isEmpty() ) return 0;
	return dynamic_cast<Node *>(lst[0]);
}
void Scene::setSelectedNode(Node *node)
{
	clearSelection();
	if( node != 0 )
		node->setSelected(true);
}
void Scene::layoutAll()
{
	m_rootNode->layoutChildren();
}
void Scene::removeAll()
{
	//m_rootNode->removeFromScene();
	removeItem(m_rootNode);
	///m_rootNode->freeNode();
	m_rootNode = 0;
	//m_selectedNode = 0;
}
Node *Scene::addNode(const QString &text, Node *parentNode)
{
	Node *node = new Node(text, parentNode);
	connect(node->document(), SIGNAL(contentsChanged()),
			this, SLOT(onContentsChanged()));
	connect(node, SIGNAL(doDelete(Node *)), this, SLOT(doDelete(Node *)));
	return node;
}
void Scene::addChildNode()
{
	Node *curNode = selectedNode();
	if( curNode == 0 ) return;
	Node *node = addNode("Node", curNode);

#if 1
	//	作成したノードをいったん削除しない版
	m_undoStack->push(new UndoAddNodeCommand(this, node));
	//layoutAll();
#else
	//	QUndoStack::push() で redo() がコールされてしまうので、作成したノードをいったん削除
	Node *pNode = node->parentNode();
	const int ix = pNode->childIndexOf(node);
	pNode->removeChildNode(node);	//	親ノードの子ノードリストから削除
	removeItem(node);		//	シーンから node を削除
	m_undoStack->push(new UndoAddNodeCommand(this, node, pNode, ix));
	//layoutAll();
#endif
}
void Scene::keyPressEvent ( QKeyEvent * keyEvent )
{
	int key = keyEvent->key();
	if( key == Qt::Key_Shift || key == Qt::Key_Control )
		return;
	const bool ctrl = (keyEvent->modifiers() & Qt::ControlModifier) != 0;
	const bool shift = (keyEvent->modifiers() & Qt::ShiftModifier) != 0;
	switch( m_mode ) {
	case COMMAND:
		if( !ctrl && !shift ) {
			switch( key ) {
			case Qt::Key_Up:
				doCurUp();
				return;
			case Qt::Key_Down:
				doCurDown();
				return;
			case Qt::Key_Left:
				doCurLeft();
				return;
			case Qt::Key_Right:
				doCurRight();
				return;
			}
		}
		break;
	case INSERT:
		if( key == Qt::Key_Escape ) {
			setMode(COMMAND);
#if 0
			if( m_editNode->toPlainText() != m_orgText ) {
				const QString &text = m_editNode->toPlainText();
				m_editNode->setPlainText(m_orgText);
				m_undoStack->push(new UndoEditNodeTextCommand(this, m_editNode, text));
			}
#endif
			return;
		}
		QGraphicsScene::keyPressEvent( keyEvent );
		break;
	}
}
void Scene::doCurUp()
{
	Node *node = selectedNode();
	if( !node || node->isRootNode() ) return;	//	非選択状態、ルートノードでは移動不可
	Node *nNode = node->prevNode();
	if( nNode != 0 )
		setSelectedNode(nNode);
}
void Scene::doCurDown()
{
	Node *node = selectedNode();
	if( !node || node->isRootNode() ) return;	//	非選択状態、ルートノードでは移動不可
	Node *nNode = node->nextNode();
	if( nNode != 0 )
		setSelectedNode(nNode);
}
void Scene::doCurLeft()
{
	Node *node = selectedNode();
	if( !node || node->isRootNode() ) return;	//	非選択状態、ルートノードでは移動不可
	Node *nNode = node->parentNode();
	if( nNode != 0 )
		setSelectedNode(nNode);
}
void Scene::doCurRight()
{
	Node *node = selectedNode();
	if( !node ) return;	//	非選択状態では移動不可
	Node *nNode = node->firstChildNode();
	if( nNode != 0 )
		setSelectedNode(nNode);
}
//	選択ノード以下を削除
void Scene::removeSelectedNode()
{
	Node *node = selectedNode();
	if( !node || node->isRootNode() ) return;
	//removeNode(node);
	m_undoStack->push(new UndoRemoveNodeCommand(this, node));
}
void Scene::removeNode(Node *node)
{
	if( !node || node->isRootNode() ) return;
	Node *selNode = node->nextNode();		//	削除後に選択されるノード
	if( !selNode ) {
		if( !(selNode = node->prevNode()) )
			selNode = node->parentNode();
	}
	Node *pNode = node->parentNode();
	if( pNode )
		pNode->removeChildNode(node);	//	親ノードの子ノードリストから削除
	removeItem(node);		//	シーンから node を削除
	//layoutAll();
	setSelectedNode(selNode);
}
void Scene::doCollapseExpand()
{
	doCollapseExpand(selectedNode());
}
void Scene::doCollapseExpand(Node *node)
{
	if( !node || node->isRootNode() || !node->hasChildren() )
		return;
	node->setExpanded( !node->expanded() );
	layoutAll();
}
void Scene::setMode(Mode m)
{
	if( m == m_mode ) return;
	m_mode = m;
	Node *node = selectedNode();
	if( node != 0 ) {
		switch( m_mode ) {
		case COMMAND:
			node->setTextInteractionFlags(Qt::NoTextInteraction);
			if( node == m_editNode && m_editNode->toPlainText() != m_orgText ) {
				const QString &text = m_editNode->toPlainText();
				m_editNode->setPlainText(m_orgText);
				m_undoStack->push(new UndoEditNodeTextCommand(this, m_editNode, text));
			}
			break;
		case INSERT:
			m_editNode = node;
			m_orgText = node->toPlainText();
			node->setTextInteractionFlags(Qt::TextEditorInteraction | Qt::TextEditable);
			node->setFocus();
			//node->setTextCursor(QTextCursor(node->document()));	//	テキストカーソル有効化
			break;
		}
	}
}
void Scene::moveNodeUp()
{
	Node *node = selectedNode();
	if( !node ) return;
	Node *srcPN = node->parentNode();
	const int srcIx = srcPN->childIndexOf(node);
	//const bool isRightSide = node->isRightSide();
	if( !node->moveUp() ) return;
	Node *dstPN = node->parentNode();
	m_undoStack->push(new UndoMoveNodeCommand(this, node, srcPN, srcIx, dstPN));
	//layoutAll();
}
void Scene::moveNodeDown()
{
	Node *node = selectedNode();
	if( !node ) return;
	Node *srcPN = node->parentNode();
	const int srcIx = srcPN->childIndexOf(node);
	if( !node->moveDown() ) return;
	Node *dstPN = node->parentNode();
	m_undoStack->push(new UndoMoveNodeCommand(this, node, srcPN, srcIx, dstPN));
	//layoutAll();
}
void Scene::moveNodeLeft()
{
	Node *node = selectedNode();
	if( !node ) return;
	Node *srcPN = node->parentNode();
	const int srcIx = srcPN->childIndexOf(node);
	if( !node->moveLeft() ) return;
	Node *dstPN = node->parentNode();
	m_undoStack->push(new UndoMoveNodeCommand(this, node, srcPN, srcIx, dstPN));
	//layoutAll();
}
void Scene::moveNodeRight()
{
	Node *node = selectedNode();
	if( !node ) return;
	Node *srcPN = node->parentNode();
	const int srcIx = srcPN->childIndexOf(node);
	if( !node->moveRight() ) return;
	Node *dstPN = node->parentNode();
	m_undoStack->push(new UndoMoveNodeCommand(this, node, srcPN, srcIx, dstPN));
	//layoutAll();
}
void Scene::undo()
{
	if( !m_undoStack->canUndo() ) return;
	//m_undoMgr->count();
	//m_undoMgr->index();
	const bool macro = m_undoStack->command(m_undoStack->index() - 1)->childCount() > 1;
	m_undoStack->undo();
	if( macro )		//	複数コマンドを処理した場合は、まとめて再レイアウトを行う
		layoutAll();
}
void Scene::redo()
{
	if( !m_undoStack->canRedo() ) return;
	//m_undoMgr->count();
	//m_undoMgr->index();
	const bool macro = m_undoStack->command(m_undoStack->index())->childCount() > 1;
	m_undoStack->redo();
	if( macro )		//	複数コマンドを処理した場合は、まとめて再レイアウトを行う
		layoutAll();
}
QString Scene::toXmlText(const QRect &wRect) const
{
	QString buffer = "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
						"<map version=\"0.0\" ";
	const QRectF sr = sceneRect();
	buffer += QString("LEFT=\"%1\" TOP=\"%2\" WIDTH=\"%3\" HEIGHT=\"%4\" ")
				.arg(sr.left())
				.arg(sr.top())
				.arg(sr.width())
				.arg(sr.height());
	buffer += QString("WINWD=\"%1\" WINHT=\"%2\" ")		//	メインウィンドウサイズ
				.arg(wRect.width())
				.arg(wRect.height());
	buffer += ">\n";
	if( m_rootNode != 0 )
		buffer += m_rootNode->toXmlText();
#if 0
	foreach(Node *node, m_floatingNodes) {
		if( node )
			buffer += node->toXmlText();
	}
#endif
	buffer += "</map>\n";
	return buffer;
}
Node *Scene::createNode(Node *ptr,					//	基準ノード
						//CreatePosition addPos,		//	追加位置
						const QDomElement &element)
						//bool isRightSide,
						//bool loading)				//	ファイルロード中
{
	//const QString tagName = element.tagName();
	const QString title = element.attribute("TEXT");
	Node *node = addNode(title, ptr);
	addItem(node);
	//Node *node = createNode(ptr, addPos, title, loading);
#if 0
	QGraphicsItem *pNode = node->parentItem();
	if( addPos != LAST_CHILD )
		ptr = node;		//	直前・直後の場合は基準ノードを更新
	node->setRightSide(isRightSide);
	const bool expanded = element.attribute("EXPANDED") != "false";
	node->setExpanded(expanded);
	node->setNodeDownward(element.attribute("DOWNWARD") == "true");
	uchar s = Node::RECT_STYLE;
	const QString style = element.attribute("STYLE");
	if( style == "fork" ) s = Node::FORK_STYLE;
	else if( style == "roundRect" ) s = Node::ROUND_RECT_STYLE;
	else if( style == "circleRect" ) s = Node::CIRCLE_RECT_STYLE;
	node->setNodeStyle(s);
	const QString fontName = element.attribute("FONTNAME", globalSettings()->m_fontName);
	const int fontSize = element.attribute("FONTSIZE", QString("%1").arg(globalSettings()->m_fontSize)).toInt();
	node->setFont(QFont(fontName, fontSize));
	const QString link = element.attribute("LINK");
	if( !link.isEmpty() ) {
		node->setLink(link);
		const QString pixmapWidth = element.attribute("PXMWIDTH");
		if( !pixmapWidth.isEmpty() )
			node->setPixmapWidth(pixmapWidth);
	}
	const QString align = element.attribute("ALIGN");
	if( align == "center" ) {
		node->setTextAlign(Qt::AlignHCenter);
		//node->document()->setDefaultTextOption(QTextOption(Qt::AlignHCenter));
	} else if( align == "right" ) {
		node->setTextAlign(Qt::AlignRight);
		//node->document()->setDefaultTextOption(QTextOption(Qt::AlignRight));
	}
	const bool alignColumn = element.attribute("ALIGNCOLUMN", "false") == "true";
	node->setAlignColumn(alignColumn);
	const QString colorName = element.attribute("FILLCOLOR");
	if( !colorName.isEmpty() ) {
		node->setFillColor(QColor(colorName));
	}
	const uint cdt = QDateTime::currentDateTime().toTime_t();
	QString dt = element.attribute("CREATED");
	node->setCreatedDT(dt.isEmpty() ? cdt : dt.toUInt());
	dt = element.attribute("MODIFIED");
	node->setModifiedDT(dt.isEmpty() ? cdt : dt.toUInt());
#endif
	return node;
}
void Scene::cut()
{
	Node *node = selectedNode();
	if( !node || node == rootNode() )	//	ルートノードは削除不可
		return;
	copy();
	removeSelectedNode();
}
void Scene::copy()
{
	Node *node = selectedNode();
	if( !node ) return;
	///bool emp = true;
	QMimeData *mime = new QMimeData();
#if 0
	QString text = node->toOutlineText();
	if( !text.isEmpty() ) {
		mime->setText(text);
		emp = false;
	}
#endif
	QString text = node->toXmlText();
	if( !text.isEmpty() ) {
		text = "<map>\n" + text + "</map>\n";
		mime->setHtml(text);
		///emp = false;
		//QClipboard *cb = QApplication::clipboard();
		QClipboard *cb = QGuiApplication::clipboard();
		cb->setMimeData(mime);
	}
#if 0
	if( !emp ) {
		QClipboard *cb = QApplication::clipboard();
		cb->setMimeData(mime);
	}
#endif
}
Node *Scene::doPaste(const QString &buffer, Node *node)
{
	Node *nNode = 0;
	if( buffer.startsWith("<map>") ) {
		QDomDocument doc;
		if( doc.setContent(buffer) ) {		//	XML → DOM 変換成功
			QDomElement element = doc.documentElement();
			//qDebug() << element;
			//if( node != 0 && node->isRootNode() || crpos == LAST_CHILD ) {
				if( node != 0 && !node->expanded() )
					node->setExpanded(true);
				nNode = addNode(node, element);
				m_undoStack->push(new UndoAddNodeCommand(this, nNode));
				//layoutAll();
			//} else {
			//	nNode = addNode(node, crpos, element, toRightSide /*!node || node->isRightSide()*/);
			//}
#if 0
			if( nNode != 0 ) {
				if( nNode->parentNode() != 0 ) {
					//int index = nNode->parentNode()->childIndexOf(nNode);
					//removeNode(nNode);
					m_undoStack->push(new UndoAddNodeCommand(this, nNode));
				}
				else {
					removeFromFloatingNodes(nNode);
					if( bPos )
						nNode->setPos(pos);
					else
						nNode->setPos(m_nodeBoundingRectRaw.x() + m_nodeBoundingRectRaw.width()/2,
										m_nodeBoundingRectRaw.bottom() + globalSettings()->m_ccSpace);
					nNode->setFont(QFont(globalSettings()->m_fontName, globalSettings()->m_fontSize));
					m_undoMgr->push(new UndoAddNodeCommand(this, nNode, -1));
				}
			}
#endif
		}
	}
	return nNode;
}
void Scene::pasteChild()
{
	Node *node = selectedNode();
	if( !node ) return;
	///QClipboard *cb = QApplication::clipboard();
	QClipboard *cb = QGuiApplication::clipboard();
	const QMimeData *mime = cb->mimeData();
	if( !mime ) return;
	if( mime->hasHtml() ) {
		QString buffer = mime->html();
		if( doPaste(buffer, node) != 0 ) return;
	}
#if 0
	if( mime->hasText() ) {
		QString buffer = mime->text();
		if( crpos == LAST_CHILD )
			addChildNode(buffer);
		else
			openNextNode(buffer);
		return;
	}
#endif
}
//	parentNode に子ノードを追加する
Node *Scene::addNode(Node *parentNode, QDomElement &element)
{
	Node *node = 0;
        //const uint cdt = QDateTime::currentDateTime().toTime_t();
	QDomElement childEle = element.firstChildElement();
	while( !childEle.isNull() ) {
		//printElement(childEle, lvl + 1);
		const QString tagName = childEle.tagName();
		if( tagName == "node" ) {
			node = createNode(parentNode, childEle);
			//Node *node = m_scene->createNode(parentNode, Scene::LAST_CHILD, childEle, false, true);
			addNode(node, childEle);
#if 0
			const QString position = childEle.attribute("POSITION");
			if( !position.isEmpty() ) {
				node->setRightSideRecursive(position == "right");
				node->updateLinkIconPosRecursive();
			}
#endif
		}
		childEle = childEle.nextSiblingElement();
	}
	return node;
}
void Scene::dragEnterEvent ( QGraphicsSceneDragDropEvent * event )
{
	QGraphicsItem *item = itemAt(event->scenePos(), QTransform());
	if( item != 0 && event->mimeData()->hasHtml() ) {
		event->setAccepted(true);
	}
}
void Scene::dragMoveEvent ( QGraphicsSceneDragDropEvent * event )
{
	//View *view = dynamic_cast<View *>(event->source());
	m_dropTarget = dynamic_cast<Node *>(itemAt(event->scenePos(), QTransform()));
	event->setAccepted(m_dropTarget != 0 && event->mimeData()->hasHtml() &&
							m_dragSourceNode != m_dropTarget && 
							(!m_dragSourceNode || !m_dragSourceNode->isDescendantNode(m_dropTarget)));
}
void Scene::dropEvent ( QGraphicsSceneDragDropEvent * event )
{
	qDebug() << "Scene::dropEvent()";
	doPaste(event->mimeData()->html(), m_dropTarget);
}
void Scene::doDelete(Node *node)
{
	m_undoStack->push(new UndoRemoveNodeCommand(this, node));
}
void Scene::mouseDoubleClickEvent ( QGraphicsSceneMouseEvent * mouseEvent )
{
	qDebug() << "Scene::mouseDoubleClickEvent()";
	QGraphicsScene::mouseDoubleClickEvent(mouseEvent);
}
