#ifndef NODE_H
#define NODE_H

#include <QGraphicsTextItem>

class Scene;

class Node : public QGraphicsTextItem
{
	Q_OBJECT

public:
	enum {
		PC_SPACE = 16,	//	親子ノード間隔
		CC_SPACE = 4,	//	子ノード間隔
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
	Node	*firstChildNode() const;	//	最初の子供ノードを返す。子どもが無い場合は 0 を返す
	Node	*prevNode() const;		//	直前ノードを返す。長兄の場合は 0 を返す
	Node	*nextNode() const;		//	次のノードを返す。末っ子の場合は 0 を返す
	qreal	childrenHeight() const;		//	子ノード全体の高さを返す
	qreal	treeHeight() const;		//	子ノードを含めた全体の高さを返す
	int		childIndexOf(Node *) const;
	QRectF	boundingRect() const;
	uchar	nodeStyle() const { return m_nodeStyle; }
	QString	toXmlText() const;
	bool	isDescendantNode(const Node *) const;	//	子孫ノードかどうかをチェック

public:
	void	setExpanded(bool b) { makeChildrenVisible(m_expanded = b); }
	void	makeChildrenVisible(bool b);	//	子ノードの表示非表示状態設定
	qreal	layoutChildren();		//	子ノード位置を決め、ツリーの高さを返す
	void	removeChildNode(Node *);	//	指定ノードを子ノードリストから削除
	Node	*addNode(Node *);
	void	insert(int, Node *);	//	指定番目の子供としてノード追加
	bool	moveUp();
	bool	moveDown();
	bool	moveLeft();
	bool	moveRight();
	bool	levelUp();		//	ノードを上の階層に移動
	bool	levelDown();	//	ノードを下の階層に移動
	Scene	*scene();

protected:
    void    setBranch();            //  枝をセットアップ

protected:
	void	paint(QPainter *, const QStyleOptionGraphicsItem *, QWidget *);
	void	mouseDoubleClickEvent ( QGraphicsSceneMouseEvent * mouseEvent );
	//void	mousePressEvent ( QGraphicsSceneMouseEvent * event );
	void	mouseMoveEvent ( QGraphicsSceneMouseEvent * event );

signals:
	void	doDelete(Node *);

private:
	bool	m_expanded;			//	展開状態
	uchar	m_nodeStyle;		//	ノードスタイル フォーク/矩形/...
	QList<Node *>	m_children;		//	子ノードへのポインタのリスト
	class QGraphicsPathItem	*m_branch;		//	枝
	//class QGraphicsLineItem	*m_branch;		//	枝
};

#endif // NODE_H
