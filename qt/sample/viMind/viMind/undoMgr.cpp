#include "undoMgr.h"
#include "Node.h"
#include "Scene.h"

UndoRemoveNodeCommand::UndoRemoveNodeCommand(
							Scene *scene,
							Node *node)
	: QUndoCommand(0)
	, m_scene(scene)
	, m_node(node)
	//, m_index(node->parentNode() ? node->parentNode()->childIndexOf(node) : -1)
{
	m_parentNode = node->parentNode();
	m_index = m_parentNode ? m_parentNode->childIndexOf(node) : -1;
}
void UndoRemoveNodeCommand::undo()
{
#if 0
	if( m_node->parentNode() ) {
		Node *pNode = m_node->parentNode();
		pNode->insert(m_index, m_node);
		if( !pNode->isVisible() ) {			//	親ノードが折り畳まれていた場合
			while( !pNode->isVisible() )
				pNode = pNode->parentNode();
			m_scene->doCollapseExpand(pNode);
		}
	} else
		m_scene->addToFloatingNodes(m_node);
#endif
	//if( m_parentNode )
		m_parentNode->insert(m_index, m_node);
	//m_scene->addItem(m_node);
	m_scene->setSelectedNode(m_node);
	m_scene->layoutAll();
	m_node->ensureVisible();
}
void UndoRemoveNodeCommand::redo()
{
	m_scene->removeNode(m_node);
	m_scene->layoutAll();
}
//----------------------------------------------------------------------
#if 1
UndoAddNodeCommand::UndoAddNodeCommand(
						Scene *scene,
						Node *node)
	: QUndoCommand(0)
	, m_scene(scene)
	, m_node(node)
	, m_parentNode(0)
	, m_index(-1)
{
}
void UndoAddNodeCommand::undo()
{
	m_scene->removeNode(m_node);
	m_scene->layoutAll();
}
void UndoAddNodeCommand::redo()
{
	if( !m_parentNode ) {		//	push() 時にコールされた場合
		m_parentNode = m_node->parentNode();
		m_index = m_parentNode->childIndexOf(m_node);
	} else {
		m_parentNode->insert(m_index, m_node);
		//m_scene->addItem(m_node);
		m_scene->setSelectedNode(m_node);
	}
	m_scene->layoutAll();
	m_node->ensureVisible();
}
#else
UndoAddNodeCommand::UndoAddNodeCommand(
						Scene *scene,
						Node *node,
						Node *parentNode,
						int index)
	: QUndoCommand(0)
	, m_scene(scene)
	, m_node(node)
	, m_parentNode(parentNode)
	, m_index(index)
{
}
void UndoAddNodeCommand::undo()
{
	m_scene->removeNode(m_node);
	m_scene->layoutAll();
}
void UndoAddNodeCommand::redo()
{
	if( m_parentNode )
		m_parentNode->insert(m_index, m_node);
	m_scene->addItem(m_node);
	m_scene->setSelectedNode(m_node);
	m_scene->layoutAll();
}
#endif
//----------------------------------------------------------------------
UndoMoveNodeCommand::UndoMoveNodeCommand(Scene *scene, Node *node,
											Node *sp, int si,
											Node *dp)
	: QUndoCommand(0)
	, m_scene(scene)
	, m_node(node)
	, m_srcPN(sp)
	, m_srcIx(si)
	, m_dstPN(dp)
	, m_dstIx(-1)
{
}
void UndoMoveNodeCommand::undo()
{
	m_dstPN->removeChildNode(m_node);
	m_srcPN->insert(m_srcIx, m_node);
	//m_node->setRightSideRecursive(m_srcIsRS);
	//m_node->updateLinkIconPosRecursive();
	m_scene->setSelectedNode(m_node);
	m_scene->layoutAll();
	m_node->ensureVisible();
}
void UndoMoveNodeCommand::redo()
{
	if( m_dstIx < 0 ) {		//	push() 時の redo() は処理を行わないように
		m_dstIx = m_dstPN->childIndexOf(m_node);
		//m_dstIsRS = m_node->isRightSide();
		m_scene->layoutAll();
	} else {
		m_srcPN->removeChildNode(m_node);
		m_dstPN->insert(m_dstIx, m_node);
		m_scene->setSelectedNode(m_node);
		m_scene->layoutAll();
		m_node->ensureVisible();
	}
}
//----------------------------------------------------------------------
void UndoEditNodeTextCommand::doUndoRedo()
{
	const QString text = m_node->toPlainText();
	//if( text == m_text )
	//	return;		//	push() 時の redo() の場合
	m_node->setPlainText(m_text);
	m_text = text;
	m_scene->layoutAll();
	m_scene->setSelectedNode(m_node);
	m_node->ensureVisible();
}
