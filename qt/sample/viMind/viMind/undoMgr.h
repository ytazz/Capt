#pragma once

#include <QUndoStack>

class Scene;
class Node;

//----------------------------------------------------------------------
//	�m�[�h�폜
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
	Node	*m_node;			//	�폜���ꂽ�m�[�h
	Node	*m_parentNode;		//	�폜���ꂽ�m�[�h�̐e�m�[�h
	int		m_index;			//	���Ԗڂ̌Z�킩 (0..*)
};
//----------------------------------------------------------------------
//	�m�[�h�ǉ�
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
	Node	*m_node;			//	�ǉ����ꂽ�m�[�h
	Node	*m_parentNode;		//	�ǉ����ꂽ�m�[�h�̐e�m�[�h
	int		m_index;			//	���Ԗڂ̌Z�킩 (0..*)
};
//----------------------------------------------------------------------
//	�m�[�h�ړ�
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
	Node	*m_node;			//	�ړ�����m�[�h
	Node	*m_srcPN;			//	�ړ����e�m�[�h
	int		m_srcIx;			//	�ړ����q�ԍ��i0..*�j
	Node	*m_dstPN;			//	�ړ���e�m�[�h
	int		m_dstIx;			//	�ړ���q�ԍ��i0..*�j
};
//----------------------------------------------------------------------
//	�m�[�h�e�L�X�g�C��
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
	Node	*m_node;			//	�ړ�����m�[�h
	QString	m_text;
};
