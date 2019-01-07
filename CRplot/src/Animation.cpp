#include <CRplot.h>

#include <DataLoader.h>
#include <DataStruct.h>
#include <BalanceMonitor.h>

#include <algorithm>
#include <ctime>
#include <iostream>
#include <string>
#include <vector>

#ifdef LINUX
#include <unistd.h>
#endif
#ifdef WINDOWS
#include <windows.h>
#endif

float PI = 3.1415;

using namespace std;
using namespace nkk;

int main() {
    DataLoader loader("data.csv");
    vector<Data> dataList = loader.getData();

    BalanceMonitor mod;

    State test_state;
    test_state.icp.r  = 0.056;
    test_state.icp.th = 0.0;
    test_state.swf.r  = 0.158;
    test_state.swf.th = (90.0) * PI / 180.0;

    PolarCoord test_swf;
    test_swf.r  = 0.152;
    test_swf.th = 2.406;

    mod.setCurrent(test_state, test_swf);
    mod.findCaptureRegion(&dataList);

    CRplot pp;
    int    count = 0;

    while (count < 19) {
        test_state.icp.th += ((360.0) * PI / 180.0) / 20;
        mod.setCurrent(test_state, test_swf);
        mod.findCaptureRegion(&dataList);
        pp.plot(test_state, mod.captureRegion);
#ifdef LINUX
        usleep(500 * 1000);  // usleep takes sleep time in us
#endif
#ifdef WINDOWS
        Sleep(500);
#endif
        count++;
    }

    return 0;
}
