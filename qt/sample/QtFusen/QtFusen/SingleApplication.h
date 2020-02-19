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
	bool	isFirstApp();		//	�ŏ��̃C���X�^���X���H
	//	�ŏ��ɋN�����ꂽ�C���X�^���X�փ��b�Z�[�W���M
	void	sendMessage(const QString &text);

public slots:
	void	onAccepted();	//	�Q�x�ڈȍ~�ɋN�����ꂽ�C���X�^���X����̃��b�Z�[�W��M

signals:
	void	onRecieved(const QString);	//	�Q�x�ڈȍ~�ɋN�����ꂽ�C���X�^���X����̃��b�Z�[�W��M

private:
	bool	m_connected;
	const QString	m_uniqKeyString;
	QLocalSocket	m_localSocket;
	QLocalServer	m_localServer;
};

#endif // SINGLEAPPLICATION_H
