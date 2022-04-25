#include "plot.h"
#include <iostream>

#include <sbudp.h>

using namespace std;
using namespace Capt;

class MyCallback : public UDPReceiveCallback{
public:
	virtual void OnUDPReceive(const byte* buf, size_t len){

	}
};

UDPReceiver  udpReceiver;
//UDPSender    udpSender;
//MyCallback   udpCallback;

int    portRx;
int    portTx;
string address;

struct RxData{

};

struct TxData{

};

int main(int argc, char const *argv[]) {
	Scenebuilder::XML xmlCapt;
	xmlCapt.Load("conf/capt.xml");

	Scenebuilder::XML xmlCom;
	xmlCom.Load("conf/com.xml");
	xmlCom.Get(portRx , ".port_rx");
	xmlCom.Get(portTx , ".port_tx");
	xmlCom.Get(address, ".address");

	Capturability cap;
	
	cap.Read(xmlCapt.GetRootNode());
	
	cap.Load("data/");
	printf("load done\n");

	//udpReceiver.SetCallback(&udpCallback);
	//udpReceiver.Connect    (portRx);
	//
	//udpSender.Connect(address.c_str(), portTx, false);

	while(true){

	}

	return 0;
}
