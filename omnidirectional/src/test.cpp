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
    float r, th;
};
struct State {
    PolarCoord icp;
    PolarCoord sfPos;
};

float distanceTwoPolar (PolarCoord a, PolarCoord b){
    return sqrtf(sq(a.r) + sq(b.r) - 2*a.r*b.r*cos(a.th-b.th));
}


State stepping (State p0, PolarCoord u){
    PolarCoord hatCP;
    State p1;
    float tau;

    tau =  distanceTwoPolar(p0.sfPos, u) / FOOTVEL + MINIMUM_STEPPING_TIME; //所要時間

    //所要時間後の状態
    hatCP.r = (p0.icp.r - FOOTSIZE) * expf(OMEGA * tau) + FOOTSIZE;
    hatCP.th = p0.icp.th;

    //reset
    p1.sfPos.r = u.r;
    p1.sfPos.th = PI - u.th;
    p1.icp.r = distanceTwoPolar(u, hatCP);
    float alpha = hatCP.r * cos(hatCP.th) - u.r * cos(u.th);
    float beta =  -hatCP.r * sin(hatCP.th) + u.r * sin(u.th);
    if (beta > 0) {
        p1.icp.th = acosf(alpha/p1.icp.r);
    }else{
        p1.icp.th = 2*PI - acosf(alpha/p1.icp.r);
    }

    return p1;
}



int main(void) {
    State testp1;
    testp1.icp.r = 0.05;
    testp1.icp.th = PI*3/2.0;
    testp1.sfPos.r = 0.09;
    testp1.sfPos.th = PI/2.0;

    PolarCoord u;
    u.r = 0.05;
    u.th = PI/2.0;

    State result;
    result = stepping(testp1, u);

    printf("%lf, %lf, %lf, %lf\n",result.icp.r, result.icp.th,
                                  result.sfPos.r, result.sfPos.th);

    return 0;
}





















/////
