#include <QtGui>
#include "blockmain.h"
#include "BlockScene.h"

///BlockMain::BlockMain(QWidget *parent, Qt::WFlags flags)
BlockMain::BlockMain(QWidget *parent, Qt::WindowFlags flags)
	: QWidget(parent, flags)
{
	qsrand(QTime(0, 0).msecsTo(QTime::currentTime()));
	//m_ballLeft = 5;

	QVBoxLayout *vLayout = new QVBoxLayout();
		vLayout->addWidget(new QLabel(tr("SCORE:")));
		//vLayout->addWidget(m_scoreLabel = new QLabel(tr("000000")));
		vLayout->addWidget(m_scoreLCD = new QLCDNumber());
		vLayout->addStretch();
		vLayout->addWidget(m_playPauseButton = new QPushButton(tr("Play")));
		m_playPauseButton->setCheckable(true);
		m_playPauseButton->setChecked(false);	//	チェック状態＝プレイ中
		connect(m_playPauseButton, SIGNAL(toggled(bool)), this, SLOT(onPlayPause(bool)));
		vLayout->addWidget(new QLabel(tr("Left:")));
		vLayout->addWidget(m_ballLeftLCD = new QLCDNumber());
		//vLayout->addWidget(m_ballLeftLabel = new QLabel(tr("0")));
		//m_ballLeftLabel->setText(QString("%1").arg(m_ballLeft));
	QHBoxLayout *hLayout = new QHBoxLayout();
		hLayout->addLayout(vLayout);
		hLayout->addWidget(m_blockScene = new BlockScene());
		connect(m_blockScene, SIGNAL(ballLoss()), this, SLOT(ballLoss()));
		connect(m_blockScene, SIGNAL(hitBlock()), this, SLOT(hitBlock()));
		connect(m_blockScene, SIGNAL(ballReleased()), this, SLOT(ballReleased()));
	hLayout->setSizeConstraint(QLayout::SetFixedSize);
	setLayout(hLayout);
	//setSizePolicy(QSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed));

#if 1
	m_timer = new QTimer();
	connect(m_timer, SIGNAL(timeout()), m_blockScene, SLOT(onTimer()));
	//m_timer->start(20);		//	50/sec
#endif

	//grabMouse ();
	restart();
}

BlockMain::~BlockMain()
{
	//releaseMouse ();
}

void BlockMain::restart()
{
	m_timer->start(20);		//	50/sec
	m_score = 0;
	m_scoreLCD->display(m_score);
	//m_scoreLabel->setText(QString().sprintf("%06d", m_score));
	m_ballLeft = 2;
	m_ballLeftLCD->display(m_ballLeft);
	//m_ballLeftLabel->setText(QString("%1").arg(m_ballLeft));
	update();
	m_blockScene->restart();
}

void BlockMain::ballReleased()
{
	m_playPauseButton->setChecked(true);
	//onPlayPause(true);
}

//	【Paly/Pause】ボタンが押された場合の処理
//
//		【Pause】が押されるとプレイ中のボールを止める、パッド移動も休止
//					→ タイマーを止め、フレーム更新中断
//		【Play】は、ゲーム開始・ゲーム中断終了、の２つの機能がある
//					
void BlockMain::onPlayPause(bool play)
{
	m_playPauseButton->setText(play ? tr("Pause") : tr("Play"));
	if( play ) {
		m_blockScene->onPlay();
		m_timer->start(20);		//	50/sec
	} else {
		m_timer->stop();
	}
}

void BlockMain::hitBlock()
{
	m_score += 10;
	m_scoreLCD->display(m_score);
	//m_scoreLabel->setText(QString().sprintf("%06d", m_score));
}

void BlockMain::ballLoss()
{
	m_playPauseButton->setChecked(false);
	--m_ballLeft;
	m_ballLeftLCD->display(m_ballLeft);
	update();
	if( !m_ballLeft ) {
		//	ゲームオーバー
		m_timer->stop();
		if( QMessageBox::Yes ==
				QMessageBox::question(this, "block", "GAME OVER\n\ngame again ?",
										QMessageBox::Yes | QMessageBox::No) )
		{
			restart();
		} else
			qApp->quit();
	} else {
		//m_ballLeftLabel->setText(QString("%1").arg(m_ballLeft));
		m_blockScene->restart(/* reset block = */false);
	}
	m_timer->start(20);		//	for パッド移動
}
