#include "plot.h"
#include <iostream>

#include <sbudp.h>

using namespace std;
using namespace Capt;

CheckRequest   req;
CheckResult    res;
Capturability  cap;

UDPReceiver    udpReceiver;
UDPSender      udpSender;

class MyCallback : public UDPReceiveCallback{
public:
	virtual void OnUDPReceive(const byte* buf, size_t len){
		if(len == sizeof(CheckRequest)){
			printf("received request\n");
			copy(buf, buf + len, (byte*)&req);

			cap.Check(req, res);

			udpSender.Send((const byte*)&res, sizeof(CheckResult));
			printf("sent result\n");
		}
	}
};

MyCallback    udpCallback;

int    portRx;
int    portTx;
string address;

int main(int argc, char const *argv[]) {
	Scenebuilder::XML xmlCapt;
	xmlCapt.Load("conf/capt.xml");

	Scenebuilder::XML xmlCom;
	xmlCom.Load("conf/com.xml");
	xmlCom.Get(portRx , ".port_rx");
	xmlCom.Get(portTx , ".port_tx");
	xmlCom.Get(address, ".address");
	
	cap.Read(xmlCapt.GetRootNode());
	
	cap.Load("data/");
	printf("load done\n");

	udpReceiver.SetCallback(&udpCallback);
	udpReceiver.Connect    (portRx);
	
	udpSender.Connect(address.c_str(), portTx, false);

	while(true){

	}

	return 0;
}
