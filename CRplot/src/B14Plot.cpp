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

    std::vector<float> suFt_p_icp;
    suFt_p_icp.push_back(0.02);
    suFt_p_icp.push_back(0.04);

    std::vector<float> suFt_p_swFt;
    suFt_p_swFt.push_back(-0.02146);
    suFt_p_swFt.push_back(0.10);

    std::vector<float> suFt_p_refLP;
    suFt_p_refLP.push_back(0.02476);
    suFt_p_refLP.push_back(0.11);

    testMonitor.loadData(data_dir, table_dir);
    testMonitor.set_xy(suFt_p_icp, suFt_p_swFt, suFt_p_refLP);

    std::vector<float>   modifiedLandingPosition;
    std::vector<CAinput> cr;
    CRplot               pp;



    testMonitor.findCaptureRegion();
    cr                      = testMonitor.getCurrentCR();
    modifiedLandingPosition = testMonitor.getLP_xy();

    std::cout << modifiedLandingPosition[0] << ", "
              << modifiedLandingPosition[1] << "\n";

    std::cout << "safe :" << testMonitor.safe() << "\n";
    std::cout << cr.size() << "\n";

    pp.plot(testMonitor.closestGridfrom(testMonitor.current_state), cr);

    return 0;
}
