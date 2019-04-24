#ifndef BLOCKMAIN_H
#define BLOCKMAIN_H

///#include <QtGui/QWidget>
#include <QtWidgets>

class QLCDNumber;
class QLabel;
class QPushButton;
class QTimer;
class BlockScene;

class BlockMain : public QWidget
{
	Q_OBJECT

public:
	///BlockMain(QWidget *parent = 0, Qt::WFlags flags = 0);
	BlockMain(QWidget *parent = 0, Qt::WindowFlags flags = 0);
	~BlockMain();

public slots:
	void	ballLoss();		//	ボールをパッドで打ち損なった場合
	void	hitBlock();		//	ブロック消去した場合
	void	restart();		//	再ゲーム
	void	ballReleased();			//	ボールリリース
	void	onPlayPause(bool);		//	【Play/Pause】ボタン

private:
	int		m_score;			//	スコア
	int		m_ballLeft;			//	残りボール数
	QLCDNumber	*m_scoreLCD;		//	スコア
	QLCDNumber	*m_ballLeftLCD;		//	残りボール数
	//QLabel	*m_scoreLabel;		//	スコア
	//QLabel	*m_ballLeftLabel;	//	残りボール数
	QPushButton	*m_playPauseButton;		//	Play/Pause ボタン

	QTimer	*m_timer;
	BlockScene	*m_blockScene;
};

#endif // BLOCKMAIN_H
