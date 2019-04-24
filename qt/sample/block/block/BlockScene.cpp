///#include <QtGui>
#include <QtWidgets>
#include "BlockScene.h"
#include <QDebug>

BlockScene::BlockScene(QWidget *parent)
	: QWidget(parent)
{
	m_mouse_x = 0;

	restart();

	//m_sound = new QSound(":MainWindow/Resources/cls2a.mp3");
	//setMouseTracking(true);
}

BlockScene::~BlockScene()
{

}
void BlockScene::restart(bool resetBlock)
{
	m_ballCatched = true;
	if( resetBlock )
		createBlock();

	m_ballX = SCENE_WIDTH / 2;
	m_ballY = SCENE_HEIGHT - PAD_WIDTH - BALL_RADIUS;
	m_ballDx = INIT_BALL_DX;
	m_ballDy = INIT_BALL_DY;
	m_prevPadX = m_padX = SCENE_WIDTH / 2 - PAD_LENGTH / 2;
}
void BlockScene::createBlock()
{
	m_blockLeft = N_ROW_BLOCK * N_COLUMN_BLOCK;
	for(int i = 0; i < N_ROW_BLOCK; ++i)
		for(int k = 0; k < N_COLUMN_BLOCK; ++k)
			m_block[i][k] = true;
}
void BlockScene::mousePressEvent ( QMouseEvent * event )
{
	if( m_ballCatched ) {
		m_ballCatched = false;
		emit ballReleased();
	}
}
void BlockScene::mouseMoveEvent ( QMouseEvent * event )
{
	m_mouse_x = event->x();
}

void BlockScene::onTimer()
{
	iteratePad();
	if( m_ballCatched ) {
		m_ballX = m_padX + PAD_LENGTH / 2;
		m_ballY = SCENE_HEIGHT - PAD_WIDTH - BALL_RADIUS;
	} else {
		iterateBall();
	}
	update();
}

void BlockScene::iteratePad()
{
	//	パッド位置更新
	m_padX = mapFromGlobal(QCursor::pos()).x() - PAD_LENGTH / 2;
	//m_padX = m_mouse_x - PAD_LENGTH / 2;
	if( m_padX < WALL_WIDTH )
		m_padX = WALL_WIDTH;
	else if( m_padX + PAD_LENGTH > SCENE_WIDTH - WALL_WIDTH )
		m_padX = SCENE_WIDTH - WALL_WIDTH - PAD_LENGTH;
}

//	ブロックによる反射
//		前提条件：ボール垂直方向速度はブロック高さを超えない
void BlockScene::refrectBlock()
{
	const int r = (m_ballY - BLOCK_Y) / BLOCK_HEIGHT;
	if( r < 0 || r >= N_ROW_BLOCK )
		return;
	//	反射位置Y座標
	const int Y = BLOCK_Y + r * BLOCK_HEIGHT + (m_ballDy < 0 ? BLOCK_HEIGHT : 0);
	//	反射位置X座標
	const int X = m_ballX + (Y - m_ballY) * m_ballDx / m_ballDy;
	int c = (X - WALL_WIDTH) / BLOCK_WIDTH;
	if( c < 0 || c >= N_COLUMN_BLOCK )
		return;
	if( m_block[r][c] ) {
		//	存在しているブロックの上辺 or 下辺にぶつかった場合
		m_ballY = Y + Y - m_ballY;
		m_ballDy = -m_ballDy;
	} else {
		//c = (m_ballX - WALL_WIDTH) / BLOCK_WIDTH;
		//if( c < 0 || c >= N_COLUMN_BLOCK || !m_block[r][c] )
		//	return;
		const int blockX = c * BLOCK_WIDTH + WALL_WIDTH;	//	ブロック左辺座標
		if( c + 1 < N_COLUMN_BLOCK && m_ballDx > 0 &&
			m_ballX >= blockX + BLOCK_WIDTH &&
			m_block[r][c + 1] )
		{
			//	ひとつ右のブロックの左辺での反射
			m_ballX = (blockX + BLOCK_WIDTH) * 2 - m_ballX;
			m_ballDx = - m_ballDx;
			++c;
		} else if( c - 1 >= 0 && m_ballDx < 0 &&
			m_ballX <= blockX  &&
			m_block[r][c - 1] )
		{
			//	ひとつ左のブロックの右辺での反射
			m_ballX = blockX * 2 - m_ballX;
			m_ballDx = - m_ballDx;
			--c;
		} else
			return;
	}
	m_block[r][c] = false;
	QApplication::beep();
	emit hitBlock();
	if( !--m_blockLeft )
		restart();
#if 0
	if( r >= 0 && r < N_ROW_BLOCK ) {
		int c = (m_ballX - WALL_WIDTH) / BLOCK_WIDTH;
		if( c >= 0 && c < N_COLUMN_BLOCK && m_block[r][c] ) {
			//	存在しているブロックにぶつかった場合
			//	反射したブロック端 Y 座標：
			const int Y = BLOCK_Y + r * BLOCK_HEIGHT + (m_ballDy < 0 ? BLOCK_HEIGHT : 0);
			const int X = m_ballX + (Y - m_ballY) * m_ballDx / m_ballDy;
			const int blockX = c * BLOCK_WIDTH + WALL_WIDTH;
			if( X >= blockX && X <= blockX + BLOCK_WIDTH ) {
				//	ブロック下辺 or 上辺での反射
				m_ballY = Y + Y - m_ballY;
				m_ballDy = -m_ballDy;
			} else if( m_ballDx > 0 ) {
				if( c > 0 && m_block[r][c-1] ) {
					//	左にブロックが存在した場合
					--c;
					//	ブロック下辺 or 上辺での反射
					m_ballY = Y + Y - m_ballY;
					m_ballDy = -m_ballDy;
				} else {
					//	ブロック左辺での反射
					m_ballX = blockX * 2 - m_ballX;
					m_ballDx = - m_ballDx;
				}
			} else {
				if( c + 1 < N_ROW_BLOCK && m_block[r][c+1] ) {
					//	右にブロックが存在した場合
					++c;
					//	ブロック下辺 or 上辺での反射
					m_ballY = Y + Y - m_ballY;
					m_ballDy = -m_ballDy;
				} else {
					//	ブロック右辺での反射
					m_ballX = (blockX + BLOCK_WIDTH) * 2 - m_ballX;
					m_ballDx = - m_ballDx;
				}
			}
			m_block[r][c] = false;
			QApplication::beep();
			emit hitBlock();
			if( !--m_blockLeft ) {
				restart();
				//createBlock();
			}
		}
	}
#endif
}

void BlockScene::iterateBall()
{
	//	ボール座標更新処理：
	m_ballX += m_ballDx;
	m_ballY += m_ballDy;

	refrectBlock();		//	ブロックによる反射・ブロック消去

	//	左右の壁による反射
	if( m_ballX < WALL_WIDTH + BALL_RADIUS ) {
		m_ballX = (WALL_WIDTH + BALL_RADIUS) * 2 - m_ballX;
		m_ballDx = -m_ballDx;
		QApplication::beep();
	} else if( m_ballX > SCENE_WIDTH - WALL_WIDTH - BALL_RADIUS) {
		m_ballX = (SCENE_WIDTH - WALL_WIDTH - BALL_RADIUS) * 2 - m_ballX;
		m_ballDx = -m_ballDx;
		QApplication::beep();
		//m_sound->play();
	}
	//	上の壁による反射
	if( m_ballY < WALL_WIDTH + BALL_RADIUS ) {
		m_ballY = (WALL_WIDTH + BALL_RADIUS) * 2 - m_ballY;
		m_ballDy = -m_ballDy;
		QApplication::beep();
	}
	//	パッドによる反射
	if( m_ballY > SCENE_HEIGHT - PAD_WIDTH ) {
		if( m_ballX >= m_padX && m_ballX <= m_padX + PAD_LENGTH ) {
			if( m_ballX < m_padX + PAD_LENGTH / 3 ) {
				//	パッドの左 1/3 に当たった場合
				if( m_ballDx > 0 ) m_ballDx = -m_ballDx;	//	反射
				m_ballDx += qrand() % 7 - 3;		//	＋3 ～ -3
				if( m_ballDx < MIN_BALL_DX ) m_ballDx = MIN_BALL_DX;
				else if( m_ballDx > MAX_BALL_DX ) m_ballDx = MAX_BALL_DX;
#if 0
				if( m_padX < m_prevPadX ) {		//	パッドが左方向に移動中
					if( m_ballDx > 0 )
						m_ballDx = -m_ballDx;	//	反射
					else if( m_ballDx > MIN_BALL_DX )
						--m_ballDx;
				} else if( m_padX > m_prevPadX ) {		//	パッドが右方向に移動中
					if( m_ballDx > 0 ) --m_ballDx;
				}
#endif
#if 0
				if( m_ballDx > 0 ) m_ballDx = -m_ballDx;	//	反射
				else if( !m_ballDx ) m_ballDx = -1;
				else if( m_ballDx > MIN_BALL_DX ) --m_ballDx;
#endif
			} else if( m_ballX >= m_padX + PAD_LENGTH * 2 / 3 ) {
				//	パッドの右 1/3 に当たった場合
				if( m_ballDx < 0 ) m_ballDx = -m_ballDx;	//	反射
				m_ballDx += qrand() % 3 - 1;
				if( m_ballDx < MIN_BALL_DX ) m_ballDx = MIN_BALL_DX;
				else if( m_ballDx > MAX_BALL_DX ) m_ballDx = MAX_BALL_DX;
#if 0
				if( m_padX < m_prevPadX ) {		//	パッドが左方向に移動中
					if( m_ballDx < 0 ) ++m_ballDx;
				} else if( m_padX > m_prevPadX ) {		//	パッドが右方向に移動中
					if( m_ballDx < 0 )
						m_ballDx = -m_ballDx;	//	反射
					else if( m_ballDx < MAX_BALL_DX )
						++m_ballDx;
				}
#endif
#if 0
				if( m_ballDx < 0 ) m_ballDx = -m_ballDx;	//	反射
				else if( !m_ballDx ) m_ballDx = +1;
				else if( m_ballDx < MAX_BALL_DX ) ++m_ballDx;
#endif
			}
			m_ballY += (m_ballDy = -m_ballDy) * 2;
			QApplication::beep();
		} else {
			emit ballLoss();
		}
	}
	
	m_prevPadX = m_padX;
}

void BlockScene::doUpdate()
{
	update();
}
void BlockScene::paintEvent(QPaintEvent * event)
{
	doPaint();
}

void BlockScene::doPaint()
{
	//qDebug() << "BlockScene::doPaint()" << m_ballX << ", " << m_ballY;
	//QTimer::singleShot(20, this, SLOT(doUpdate()));
	QPainter painter(this);
	QRect r = rect();

	//	壁部分描画
	painter.setPen(Qt::black);
	painter.setBrush(Qt::black);
	painter.drawRect(0, 0, WALL_WIDTH, r.height());
	painter.drawRect(r.width() - WALL_WIDTH, 0, WALL_WIDTH, r.height());
	painter.drawRect(0, 0, r.width(), WALL_WIDTH);

	//	ブロック部分描画
	painter.setPen(Qt::black);
	painter.setBrush(Qt::yellow);
	int y = BLOCK_Y;
	for(int i = 0; i < N_ROW_BLOCK; ++i) {
		int x = WALL_WIDTH;
		for(int k = 0; k < N_COLUMN_BLOCK; ++k) {
			if( m_block[i][k] )		//	ブロックが有る場合
				painter.drawRect(x, y, BLOCK_WIDTH, BLOCK_HEIGHT);
			x += BLOCK_WIDTH;
		}
		y += BLOCK_HEIGHT;
	}

	//	ボール描画
	painter.setPen(Qt::black);
	painter.setBrush(Qt::green);
	painter.drawEllipse(m_ballX - BALL_RADIUS, m_ballY - BALL_RADIUS,
							BALL_RADIUS * 2, BALL_RADIUS * 2);
	//painter.drawRect(m_ballX, m_ballY, BALL_SIZE, BALL_SIZE);

	//	パッド描画
	painter.setBrush(Qt::blue);
	painter.drawRect(m_padX, SCENE_HEIGHT - PAD_WIDTH, PAD_LENGTH, PAD_WIDTH);

	//update();
}
