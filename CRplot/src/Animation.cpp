#include "../include/CRplot.h"
#include <stdlib.h>
#include <vector>

float PI = 3.1415;

using namespace std;
using namespace nkk;


int main() {
    string home_dir  = getenv("HOME");
    string data_dir  = home_dir + "/ca_data/nao/data.csv";
    string table_dir = home_dir + "/ca_data/nao/gridsTable.csv";

    BalanceMonitor testMonitor;

    // std::vector<float> suFt_p_icp;
    // suFt_p_icp.push_back(0.0436);
    // suFt_p_icp.push_back(0.02);

    // std::vector<float> suFt_p_swFt;
    // suFt_p_swFt.push_back(0.0);
    // suFt_p_swFt.push_back(0.10);

    // std::vector<float> suFt_p_refLP;
    // suFt_p_refLP.push_back(0.0);
    // suFt_p_refLP.push_back(0.12);

    CAstate current_state;
    current_state.icp.r = 0.05;
    current_state.icp.th = 0.5;
    current_state.swf.r = 0.11;
    current_state.swf.th = 1.96;

    PolarCoord refLP;
    refLP.r = 0.12;
    refLP.th = 1.56;

    testMonitor.loadData(data_dir, table_dir);
    // testMonitor.set_xy(suFt_p_icp, suFt_p_swFt, suFt_p_refLP);
    testMonitor.set_polar(current_state, refLP);

    std::vector<float>   modifiedLandingPosition;
    std::vector<CAinput> cr;
    CRplot               pp;

    int count = 0;


    while (count < 20) {
        if (testMonitor.current_state.icp.th > 2 * PI) {
            testMonitor.current_state.icp.th =
                testMonitor.current_state.icp.th - 2 * PI;
        }
        testMonitor.current_state.icp.th += PI/20;
        testMonitor.findCaptureRegion();
        cr                      = testMonitor.getCurrentCR();
        modifiedLandingPosition = testMonitor.getLP_xy();
        std::cout << modifiedLandingPosition[0] << ", "
                  << modifiedLandingPosition[1] << "\n";
        std::cout << "safe :" << testMonitor.safe() << "\n";
        std::cout << cr.size() << ',' << count << "\n";

        pp.plot(testMonitor.current_state, cr);

        usleep(500 * 1000);  // usleep takes sleep time in us
        count++;
    }

    return 0;
}
