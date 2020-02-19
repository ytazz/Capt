#ifndef SINGLEAPPLICATION_H
#define SINGLEAPPLICATION_H

#include <QApplication>
#include	<QTimer>
#include	<QLocalServer>
#include	<QLocalSocket>

class SingleApplication : public QApplication
{
	Q_OBJECT

public:
	SingleApplication(int &argc, char *argv[], const QString uniqKeyString);

public:
	bool	isFirstApp();		//	最初のインスタンスか？
	//	最初に起動されたインスタンスへメッセージ送信
	void	sendMessage(const QString &text);

public slots:
	void	onAccepted();	//	２度目以降に起動されたインスタンスからのメッセージ受信

signals:
	void	onRecieved(const QString);	//	２度目以降に起動されたインスタンスからのメッセージ受信

private:
	bool	m_connected;
	const QString	m_uniqKeyString;
	QLocalSocket	m_localSocket;
	QLocalServer	m_localServer;
};

#endif // SINGLEAPPLICATION_H
