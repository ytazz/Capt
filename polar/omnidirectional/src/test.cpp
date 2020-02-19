#include <iostream>
#include <math.h>
#include <string>
#include <vector>

using namespace std;

#define sq(x) ((x) * (x))

//////////////////////////////// parameter ////////////////////////////////////
#define HEIGHTOFCOM 0.27           //[m]
#define FOOTVEL 0.44               //[m/s]
#define FOOTSIZE 0.04              //[m]
#define MINIMUM_STEPPING_TIME 0.1  //[s]
#define OMEGA sqrt(9.81 / HEIGHTOFCOM)
#define PI 3.141
////////////////////////////////////////////////////////////////////////////////
struct PolarCoord {
    float r, th;
};
struct State {
    PolarCoord icp;
    PolarCoord swf;
};

float radToDe(float rad) {
    return rad*180/3.141592;
}


float deToRad(float degree) {
    return degree*3.141592/180;
}

float distanceTwoPolar(PolarCoord a, PolarCoord b) {
    return sqrtf(sq(a.r) + sq(b.r) - 2 * a.r * b.r * cos(a.th - b.th));
}


State stepping(State p0, PolarCoord u) {
    PolarCoord hatCP;
    State      p1;
    float      tau;

    tau = distanceTwoPolar(p0.swf, u) / FOOTVEL +
          MINIMUM_STEPPING_TIME;  //所要時間

    //所要時間後の状態
    hatCP.r  = (p0.icp.r - FOOTSIZE) * expf(OMEGA * tau) + FOOTSIZE;
    //여기서 footstep만큼 빼기 때문에 r이 더 커지지 않음(0곱하기 상수는 0)
    hatCP.th = p0.icp.th;

    std::cout << hatCP.r << "\n";
    std::cout << radToDe(hatCP.th) << "\n";

    // reset
    p1.swf.r    = u.r;
    p1.swf.th   = PI - u.th;
    p1.icp.r    = distanceTwoPolar(u, hatCP);
    float alpha = hatCP.r * cos(hatCP.th) - u.r * cos(u.th);
    float beta  = -hatCP.r * sin(hatCP.th) + u.r * sin(u.th);
    if (beta > 0) {
        p1.icp.th = acosf(alpha / p1.icp.r);
    } else {
        p1.icp.th = 2 * PI - acosf(alpha / p1.icp.r);
    }

    return p1;
}

float distanceToLineSegment(PolarCoord step, PolarCoord cp) {
    PolarCoord phi;

    float xi = fminf(fmaxf(cp.r * cos(step.th - cp.th) / step.r, 0), 1);

    phi.r  = xi * step.r;
    phi.th = step.th;

    std::cout << phi.r << ", " << phi.th <<"\n";
    std::cout << distanceTwoPolar(phi, cp) << "\n";
    return distanceTwoPolar(phi, cp);
}



bool isZeroStepCapt(State p) {
    float threshold = FOOTSIZE;

    if (distanceToLineSegment(p.swf, p.icp) < threshold) {
        return true;
    } else {
        return false;
    }
}

void printState(State abc) {
    std::cout << abc.icp.r << ',' << abc.icp.th*180.0/PI << ',' << abc.swf.r << ','
    << abc.swf.th*180.0/PI << "\n";
}



int main(void) {
    State testp1;
    testp1.icp.r  = 0.04;
    testp1.icp.th = 0.0;
    testp1.swf.r  = 0.088;
    testp1.swf.th = deToRad(90);

    PolarCoord u;
    u.r  = 0.095;
    u.th = deToRad(90);


    State result1;
    result1 = stepping(testp1, u);

    printState(result1);

    bool result2;
    result2 = isZeroStepCapt(result1);

    std::cout << result2 << "\n";



    return 0;
}










/////
