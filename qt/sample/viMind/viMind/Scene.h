#ifndef SCENE_H
#define SCENE_H

#include <QGraphicsScene>

class Node;
class QDomElement;

class Scene : public QGraphicsScene
{
	Q_OBJECT

public:
	enum Mode {
		COMMAND = 0,	//	コマンドモード
		INSERT,			//	ノードテキスト編集モード
	};

public:
	Scene(QObject *parent = 0);
	~Scene();

public:
	Node	*selectedNode() const;
	Mode	mode() const { return m_mode; }
	Node	*rootNode() const { return m_rootNode; }
	//qreal	rootNodeXPos() const { return rootNode()->scenePos().x(); }
	QString	toXmlText(const QRect &wRect) const;

public:
	void	setMode(Mode m);
	//void	setMode(Mode m, Cursor = HOME);
	void	setRootNode(Node *root) { m_rootNode = root; }
	void	setDragSourceNode(Node *node) { m_dragSourceNode = node; }
	void	removeAll();
	void	layoutAll();
	void	setSelectedNode(Node *);
	void	doCollapseExpand(Node *);		//	ノード展開・折畳
	void	removeNode(Node *);
    Node	*createNode(Node *, const QDomElement &);
	Node	*addNode(Node *, QDomElement &);

public slots:
	void	undo();
	void	redo();
	void	editNode()	{ setMode(INSERT); }
	void	doCurUp();
	void	doCurDown();
	void	doCurLeft();
	void	doCurRight();
	void	addChildNode();
	void	removeSelectedNode();
	void	doCollapseExpand();		//	ノード展開・折畳
	void	onContentsChanged();
	void	moveNodeUp();			//	ノード上移動
	void	moveNodeDown();			//	ノード下移動
	void	moveNodeLeft();			//	ノード左移動
	void	moveNodeRight();		//	ノード右移動
    void	cut();
    void	copy();
    //void	paste();
    void	pasteChild();
	void	doDelete(Node *);

protected:
	Node	*addNode(const QString &, Node * = 0);
	Node	*doPaste(const QString &, Node *);

protected:
	void	keyPressEvent ( QKeyEvent * keyEvent );
	void	mouseDoubleClickEvent ( QGraphicsSceneMouseEvent * mouseEvent );

	void	dragEnterEvent ( QGraphicsSceneDragDropEvent * event );
	void	dragMoveEvent ( QGraphicsSceneDragDropEvent * event );
	void	dropEvent ( QGraphicsSceneDragDropEvent * event );

signals:
	void	contentsChanged();

private:
	Mode	m_mode;
	Node	*m_rootNode;
	Node	*m_editNode;	//	編集中ノード
	QString	m_orgText;		//	編集前テキスト
	class QUndoStack	*m_undoStack;
	Node	*m_dragSourceNode;
	Node	*m_dropTarget;
};

#endif // SCENE_H
