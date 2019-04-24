#pragma once

#include <QUndoStack>

class Scene;
class Node;

//----------------------------------------------------------------------
//	ノード削除
class UndoRemoveNodeCommand : public QUndoCommand
{
public:
	UndoRemoveNodeCommand(Scene *scene, Node *node);
	~UndoRemoveNodeCommand() {};
	
public:
	void	undo();
	void	redo();

private:
	Scene	*m_scene;
	Node	*m_node;			//	削除されたノード
	Node	*m_parentNode;		//	削除されたノードの親ノード
	int		m_index;			//	何番目の兄弟か (0..*)
};
//----------------------------------------------------------------------
//	ノード追加
class UndoAddNodeCommand : public QUndoCommand
{
public:
#if 1
	UndoAddNodeCommand(Scene *scene, Node *node);
#else
	UndoAddNodeCommand(Scene *scene, Node *node, Node *parentNode, int index);
#endif
	~UndoAddNodeCommand() {};
	
public:
	void	undo();
	void	redo();

private:
	Scene	*m_scene;
	Node	*m_node;			//	追加されたノード
	Node	*m_parentNode;		//	追加されたノードの親ノード
	int		m_index;			//	何番目の兄弟か (0..*)
};
//----------------------------------------------------------------------
//	ノード移動
class UndoMoveNodeCommand : public QUndoCommand
{
public:
	UndoMoveNodeCommand(Scene *scene, Node *node,
						Node *sp, int si,
						Node *dp);
	~UndoMoveNodeCommand() {};
	
public:
	void	undo();
	void	redo();

private:
	Scene	*m_scene;
	Node	*m_node;			//	移動するノード
	Node	*m_srcPN;			//	移動元親ノード
	int		m_srcIx;			//	移動元子番号（0..*）
	Node	*m_dstPN;			//	移動先親ノード
	int		m_dstIx;			//	移動先子番号（0..*）
};
//----------------------------------------------------------------------
//	ノードテキスト修正
class UndoEditNodeTextCommand : public QUndoCommand
{
public:
	UndoEditNodeTextCommand(Scene *scene, Node *node,
						const QString &text)
		: m_scene(scene)
		, m_node(node)
		, m_text(text)
		{}
	~UndoEditNodeTextCommand() {};
	
public:
	void	undo() { doUndoRedo(); }
	void	redo() { doUndoRedo(); }

protected:
	void	doUndoRedo();

private:
	Scene	*m_scene;
	Node	*m_node;			//	移動するノード
	QString	m_text;
};
