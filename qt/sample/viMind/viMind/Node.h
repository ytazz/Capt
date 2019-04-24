#ifndef NODE_H
#define NODE_H

#include <QGraphicsTextItem>

class Scene;

class Node : public QGraphicsTextItem
{
	Q_OBJECT

public:
	enum {
		PC_SPACE = 16,	//	�e�q�m�[�h�Ԋu
		CC_SPACE = 4,	//	�q�m�[�h�Ԋu
	};
	enum {
		FORK_STYLE = 0,
		RECT_STYLE,
		ROUND_RECT_STYLE,
		CIRCLE_RECT_STYLE,
	};

public:
	Node(const QString &text = QString(), Node *parent = 0);
	~Node();

public:
	bool	expanded() const { return m_expanded; }
	bool	isRootNode() const;
	bool	hasChildren() const { return !m_children.isEmpty(); }
	Node	*parentNode() const;
	Node	*firstChildNode() const;	//	�ŏ��̎q���m�[�h��Ԃ��B�q�ǂ��������ꍇ�� 0 ��Ԃ�
	Node	*prevNode() const;		//	���O�m�[�h��Ԃ��B���Z�̏ꍇ�� 0 ��Ԃ�
	Node	*nextNode() const;		//	���̃m�[�h��Ԃ��B�����q�̏ꍇ�� 0 ��Ԃ�
	qreal	childrenHeight() const;		//	�q�m�[�h�S�̂̍�����Ԃ�
	qreal	treeHeight() const;		//	�q�m�[�h���܂߂��S�̂̍�����Ԃ�
	int		childIndexOf(Node *) const;
	QRectF	boundingRect() const;
	uchar	nodeStyle() const { return m_nodeStyle; }
	QString	toXmlText() const;
	bool	isDescendantNode(const Node *) const;	//	�q���m�[�h���ǂ������`�F�b�N

public:
	void	setExpanded(bool b) { makeChildrenVisible(m_expanded = b); }
	void	makeChildrenVisible(bool b);	//	�q�m�[�h�̕\����\����Ԑݒ�
	qreal	layoutChildren();		//	�q�m�[�h�ʒu�����߁A�c���[�̍�����Ԃ�
	void	removeChildNode(Node *);	//	�w��m�[�h���q�m�[�h���X�g����폜
	Node	*addNode(Node *);
	void	insert(int, Node *);	//	�w��Ԗڂ̎q���Ƃ��ăm�[�h�ǉ�
	bool	moveUp();
	bool	moveDown();
	bool	moveLeft();
	bool	moveRight();
	bool	levelUp();		//	�m�[�h����̊K�w�Ɉړ�
	bool	levelDown();	//	�m�[�h�����̊K�w�Ɉړ�
	Scene	*scene();

protected:
    void    setBranch();            //  �}���Z�b�g�A�b�v

protected:
	void	paint(QPainter *, const QStyleOptionGraphicsItem *, QWidget *);
	void	mouseDoubleClickEvent ( QGraphicsSceneMouseEvent * mouseEvent );
	//void	mousePressEvent ( QGraphicsSceneMouseEvent * event );
	void	mouseMoveEvent ( QGraphicsSceneMouseEvent * event );

signals:
	void	doDelete(Node *);

private:
	bool	m_expanded;			//	�W�J���
	uchar	m_nodeStyle;		//	�m�[�h�X�^�C�� �t�H�[�N/��`/...
	QList<Node *>	m_children;		//	�q�m�[�h�ւ̃|�C���^�̃��X�g
	class QGraphicsPathItem	*m_branch;		//	�}
	//class QGraphicsLineItem	*m_branch;		//	�}
};

#endif // NODE_H
