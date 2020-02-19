#include "SingleApplication.h"
#include <QtGui>

SingleApplication::SingleApplication(int &argc, char *argv[], const QString uniqKeyString)
	: m_connected(false), m_uniqKeyString(uniqKeyString), QApplication(argc, argv)
{
	connect(&m_localServer, SIGNAL(newConnection()), this, SLOT(onAccepted()));
}

bool SingleApplication::isFirstApp()
{
	m_localSocket.connectToServer(m_uniqKeyString);
	if( m_localSocket.waitForConnected(200) ) {
		return false;
	} else {
		m_localServer.listen(m_uniqKeyString);
		return true;
	}
}
void SingleApplication::onAccepted()
{
	QLocalSocket *sock = m_localServer.nextPendingConnection();
#if		0
	if( !sock->waitForReadyRead() ) {
		return;
	}
#endif
	QByteArray ba = sock->readAll();
	QTextCodec *codec = QTextCodec::codecForName("UTF-8");
	QString buff = codec->toUnicode(ba);
	emit onRecieved(buff);
}
//	最初に起動されたインスタンスへメッセージ送信
void SingleApplication::sendMessage(const QString &text)
{
	m_localSocket.write(text.toUtf8());
	m_localSocket.flush();
	if( !m_localSocket.waitForBytesWritten(1000) ) {
		return;
	}
}
