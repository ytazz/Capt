#include <vector>
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <string>

using namespace std;

#define sq(x) ((x) * (x))

//////////////////////////////// parameter ////////////////////////////////////
#define HEIGHTOFCOM 0.225            //[m]
#define FOOTVEL 0.6                 //[m/s]
#define FOOTSIZE 0.045                //[m]
#define MINIMUM_STEPPING_TIME 0.1  //[s]
#define OMEGA sqrt(9.81/HEIGHTOFCOM)
#define PI 3.141
////////////////////////////////////////////////////////////////////////////////
struct PolarCoord {
    float r, th;z
};
struct State {
    PolarCoord icp;
    PolarCoord swf;
};

float distanceTwoPolar (PolarCoord a, PolarCoord b){
    return sqrtf(sq(a.r) + sq(b.r) - 2*a.r*b.r*cos(a.th-b.th));
}


State stepping (State p0, PolarCoord u){
    PolarCoord hatCP;
    State p1;
    float tau;

    tau =  distanceTwoPolar(p0.swf, u) / FOOTVEL + MINIMUM_STEPPING_TIME; //所要時間

    //所要時間後の状態
    hatCP.r = (p0.icp.r - FOOTSIZE) * expf(OMEGA * tau) + FOOTSIZE;
    hatCP.th = p0.icp.th;

    //reset
    p1.swf.r = u.r;
    p1.swf.th = PI - u.th;
    p1.icp.r = distanceTwoPolar(u, hatCP);
    float alpha = hatCP.r * cos(hatCP.th) - u.r * cos(u.th);
    float beta =  -hatCP.r * sin(hatCP.th) + u.r * sin(u.th);
    if (beta > 0) {
        p1.icp.th = acosf(alpha/p1.icp.r);
    }else{
        p1.icp.th = 2*PI - acosf(alpha/p1.icp.r);
    }
    if (p1.icp.th > 6.28){
      p1.icp.th = 0.0;
    }


    return p1;
}

bool isZeroStepCapt(State p){
    float threshold = FOOTSIZE;

    if (p.icp.r <= threshold) {
        return true;
    }else if (p.icp.r < p.swf.r &&
              p.swf.th - atanf(FOOTSIZE/p.swf.r) < p.icp.th &&
              p.icp.th < p.swf.th + atanf(FOOTSIZE/p.swf.r)){
        return true;
    }else{
      return false;
    }
}


int main(void) {
    State testp1;
    testp1.icp.r = 0.05;
    testp1.icp.th = 0.0;
    testp1.swf.r = 0.09;
    testp1.swf.th = 0.0;

    PolarCoord u;
    u.r = 0.05;
    u.th = 0.0;

    bool result1;
    result1 = isZeroStepCapt(testp1);

    State result2;
    result2 = stepping(testp1, u);

    printf("%lf %lf\n",testp1.swf.th, 180*(atanf(FOOTSIZE/testp1.swf.r))/PI);

    printf("%lf, %lf, %lf, %lf\n",result2.icp.r, result2.icp.th,
                                  result2.swf.r, result2.swf.th);
    printf("%d\n", result1);

    return 0;
}





















/////
