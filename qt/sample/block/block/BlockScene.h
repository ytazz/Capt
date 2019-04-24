#ifndef BLOCKSCENE_H
#define BLOCKSCENE_H

#include <QWidget>

class QSound;


class BlockScene : public QWidget
{
	Q_OBJECT

public:
	enum {
		SCENE_WIDTH = 320,
		SCENE_HEIGHT = 480,

		WALL_WIDTH = 10,		//	壁の幅
		PAD_WIDTH = 4,			//	パッド厚さ
		PAD_LENGTH = 64,		//	パッド長さ
		BALL_RADIUS = 4,		//	ボール半径
		INIT_BALL_DX = 2,		//	初期ボール速度
		INIT_BALL_DY = -4,
		MAX_BALL_DX = 8,		//	最大ボール速度
		MIN_BALL_DX = -8,

		BLOCK_WIDTH = 30,		//	ブロック幅
		BLOCK_HEIGHT = 10,		//	ブロック高さ
		N_ROW_BLOCK = 6,		//	ブロック段数
		N_COLUMN_BLOCK = (SCENE_WIDTH - WALL_WIDTH * 2) / BLOCK_WIDTH,
								//	１行のブロック個数
		N_ROW_SPACE = 3,		//	画面上部空白段数
		BLOCK_Y = WALL_WIDTH + N_ROW_SPACE * BLOCK_HEIGHT,
								//	ブロック描画位置
	};
public:
	BlockScene(QWidget *parent = 0);
	~BlockScene();

public:
	QSize	sizeHint () const { return QSize(SCENE_WIDTH, SCENE_HEIGHT); }

public slots:
	void	restart(bool resetBlock = true);
	void	onTimer();
	void	doPaint();
	void	doUpdate();
	void	onPlay() { m_ballCatched = false; }

protected:
	void	paintEvent(QPaintEvent * event);
    void	mousePressEvent ( QMouseEvent * event );
    void	mouseMoveEvent ( QMouseEvent * event );

    void	iteratePad();		//	パッド位置更新
	void	iterateBall();		//	ボール位置更新・ブロック消去判定
	void	refrectBlock();		//	ブロックによる反射
	void	createBlock();

signals:
	void	ballReleased();		//	ゲーム開始
	void	hitBlock();			//	ブロック消去
	void	ballLoss();

private:
	bool	m_ballCatched;		//	ボールがパッドにくっついてる状態
	int		m_mouse_x;

	int		m_ballX;		//	ボール中心座標
	int		m_ballY;
	int		m_ballDx;		//	ボール速度/20msec
	int		m_ballDy;
	int		m_padX;			//	パッド左端座標
	int		m_prevPadX;		//	直前パッド左端座標

	int		m_blockLeft;	//	残りブロック数
	bool	m_block[N_ROW_BLOCK][N_COLUMN_BLOCK];		//	true for ブロック有り

	QSound	*m_sound;
};

#endif // BLOCKSCENE_H
