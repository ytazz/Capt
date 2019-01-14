/*
    author G. KIM
*/

#include "../include/CRplot.h"

using namespace std;
using namespace nkk;

CRplot::CRplot() {
    string output      = "file";
    string supportfoot = "right";

    p("set encoding utf8");
    p("set size ratio -1");
    p("set polar");
    p("set grid polar 0");

    if (output == "file") {
        p("set terminal gif animate optimize delay 50 size 600,900");
        p("set output 'plot.gif'");
    }

    p("set yrange [-0.22:0.22]");
    if (supportfoot == "right") {
        p("set xrange [0.22:-0.1]");
        p("set link y2");
        p("set y2tics");
        p("set ytics scale 0");
        p("set ytics format \"\"");
        p("set y2label \"X [m]\" font \",15\" rotate by -90");
        p("set y2label offset -2, 0");
    } else {
        p("set xrange [0.0:0.22]");
        p("set ylabel \"X [m]\" font \",15\" rotate by 90");
        p("set ylabel offset 2, 0");
    }
    p("set xlabel \"Y [m]\" font \",15\"");

    p("set xtics nomirror");
    p("set ytics nomirror");
    p("set key bmargin center");
    p("set key spacing 1");

    p("set theta clockwise top");
    p("set rtics scale 0");
    p("set rtics format \"\"");
    p("unset raxis");

    p("unset key");
}

CRplot::~CRplot() {}

void CRplot::plot(CAstate              current_state,
                  std::vector<CAinput> captureRegion) {
    fprintf(p.gp, "plot");

    if (!captureRegion.empty()) {
        sort(captureRegion.begin(), captureRegion.end(),
             [](const CAinput& a, const CAinput& b) {
                 return a.n > b.n;
             });  // descending

        int step_num = captureRegion[0].n;

        while (step_num != 0) {
            fprintf(
                p.gp,
                "'-' t '%d' with points pointsize 1.5 pointtype 7 lc \"%s\",",
                step_num, p.int_color[step_num].c_str());
            step_num--;
        }

        fprintf(p.gp, "'-' t 'Capture Point' with points pointsize 3.0 "
                      "pointtype 7 lc \"orange\",");
        fprintf(p.gp, "'-' t 'Current Swing Foot' with points pointsize 3.0 "
                      "pointtype 7 lc \"green\"\n");

        size_t ind = 0;
        step_num   = captureRegion[0].n;

        while (step_num != 0) {
            while (captureRegion[ind].n == step_num) {
                fprintf(p.gp, "%f %f\n", captureRegion[ind].swf.th,
                        captureRegion[ind].swf.r);
                ind++;
            }
            fprintf(p.gp, "e\n");
            step_num--;
        }
    } else {
        fprintf(p.gp, "'-' t 'Capture Point' with points pointsize 3.0 "
                      "pointtype 7 lc \"orange\",");
        fprintf(p.gp, "'-' t 'Current Swing Foot' with points pointsize 3.0 "
                      "pointtype 7 lc \"green\"\n");
    }

    fprintf(p.gp, "%f %f\n", current_state.icp.th, current_state.icp.r);
    fprintf(p.gp, "e\n");
    fprintf(p.gp, "%f %f\n", current_state.swf.th, current_state.swf.r);
    fprintf(p.gp, "e\n");
    fflush(p.gp);
}
