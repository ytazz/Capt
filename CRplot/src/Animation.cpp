#include "../include/CRplot.h"
#include <stdlib.h>
#include <vector>

float PI = 3.1415;

using namespace std;
using namespace nkk;


int main() {
    BalanceMonitor mod;

    string home_dir  = getenv("HOME");
    string data_dir  = home_dir + "/ca_data/humanoids_d/data.csv";
    string table_dir = home_dir + "/ca_data/humanoids_d/gridsTable.csv";

    mod.loadData(data_dir, table_dir);

    CAstate test_state;
    test_state.icp.r  = 0.07;
    test_state.icp.th = 0.5;
    test_state.swf.r  = 0.152;
    test_state.swf.th = 0.349;

    PolarCoord test_swf;
    test_swf.r  = 0.152;
    test_swf.th = 0.349;

    CRplot               pp;
    int                  count = 0;
    std::vector<CAinput> cr;

    // while (count < 19) {
    //     test_state.icp.th += ((360.0) * PI / 180.0) / 20;
    //     if (test_state.icp.th > 2*PI)
    //     {
    //         test_state.icp.th = test_state.icp.th - 2*PI;
    //     }
    //     mod.set_polar(test_state, test_swf);
    //     mod.findCaptureRegion();
    //     cr = mod.getCurrentCR();
    //     std::cout << cr.size() << ',' << count << "\n";
    //     pp.plot(test_state, cr);
    //     usleep(500 * 1000);  // usleep takes sleep time in us
    //
    //     count++;
    // }

    while (count < 19) {
        test_state.swf.th += (2.792 - 0.349) / 20;
        if (test_state.swf.th > 2*PI)
        {
            test_state.swf.th = test_state.swf.th - 2*PI;
        }
        mod.set_polar(test_state, test_swf);
        mod.findCaptureRegion();
        cr = mod.getCurrentCR();
        std::cout << cr.size() << ',' << count << "\n";
        pp.plot(test_state, cr);
        usleep(500 * 1000);  // usleep takes sleep time in us

        count++;
    }

    return 0;
}
