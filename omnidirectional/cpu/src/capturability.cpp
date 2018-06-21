/**
   \author GWANWOO KIM
 */
#include "capturability.h"

void linspace(float result[], float min, float max) {
    if (max > min) {
        float h = (max - min)/(numGrid - 1);
        for (size_t i = 0; i < numGrid; i++) {
            result[i] = min + i*h;
        }
    }else{
        printf("%lf should be bigger than %lf \n", max, min);
    }
}

void makeStatesSpace(States* result,
                     float cpR[], float cpTh[], float stepR[], float stepTh[] ) {
    long int ind = 0;
    for (size_t i = 0; i < numGrid; i++) {
        for (size_t j = 0; j < numGrid; j++) {
            for (size_t k = 0; k < numGrid; k++) {
                for (size_t l = 0; l < numGrid; l++) {
                    result[ind].cp.r = cpR[i];
                    result[ind].cp.th = cpTh[j];
                    result[ind].step.r = stepR[k];
                    result[ind].step.th = stepTh[l];
                    ind++;
                }
            }
        }
    }
}

void makeInputSpace(PolarCoord* result, float stepR[], float stepTh[]) {
    long int ind = 0;
    for (size_t i = 0; i < numGrid; i++) {
        for (size_t j = 0; j < numGrid; j++) {
            result[ind].r = stepR[i];
            result[ind].th = stepTh[j];
            ind++;
        }
    }
}

void writeFile(States* data){
    FILE *fp;
    fp = fopen("statesSpace.csv", "w");

    for (long int i = 0; i < N; i++) {
        fprintf(fp, "%lf, %lf, %lf, %lf\n",
                data[i].cp.r, data[i].cp.th, data[i].step.r, data[i].step.th);
    }
    fclose(fp);
}

States oneStepAfter (States p0, PolarCoord u){
    PolarCoord hatCP;
    States p1;
    float tau;

    tau =  distanceTwoPolar(p0.step, u) / FOOTVEL; //所要時間

    //所要時間後の状態
    hatCP.r = p0.cp.r * expf(OMEGA * tau);
    hatCP.th = p0.cp.th;

    //reset
    p1.step.r = u.r;
    p1.step.th = PI - u.th;
    p1.cp.r = distanceTwoPolar(u, hatCP);
    float alpha = hatCP.r * cos(hatCP.th) - u.r * cos(u.th);
    float beta = hatCP.r * sin(hatCP.th) - u.r * sin(u.th);
    if (beta < 0) {
        p1.cp.th = acosf(alpha/p1.cp.r);
    }else{
        p1.cp.th = 2*PI - acosf(alpha/p1.cp.r);
    }

    return p1;
}

float distanceTwoPolar (PolarCoord a, PolarCoord b){
    return sqrtf(sq(a.r) + sq(b.r) - 2*a.r*b.r*cos(a.th-b.th));
}

bool isZeroStepCapt(States p){
    float threshold;

    float b1 = p.step.th - PI/2;
    float b2 = p.step.th - atanf(FOOTSIZE / p.step.r);
    float b3 = p.step.th + atanf(FOOTSIZE / p.step.r);
    float b4 = p.step.th + PI/2;

    if (b1 < p.cp.th && p.cp.th <= b2) {
        threshold = FOOTSIZE / cos(p.cp.th - (p.step.th - PI/2));

    }else if (b2 < p.cp.th && p.cp.th <= b3) {
        threshold = p.step.r * cos(p.cp.th - p.step.th) +
                    sqrtf(sq(p.step.r)*(sq(cos(p.cp.th - p.step.th)) - 1) + sq(FOOTSIZE));

    }else if (b3 < p.cp.th && p.cp.th <= b4) {
        threshold = FOOTSIZE / cos(p.cp.th - (p.step.th + PI/2));

    }else{
        threshold = FOOTSIZE;
    }

    if (p.cp.r < threshold) {
        return true;
    }else{
        return false;
    }
}

// void testZero(void){
//     FILE *fp;
//     fp = fopen("test.csv", "w");
//
//     float cpR[numGrid], cpTh[numGrid];
//     linspace(cpR, CP_MIN.r, CP_MAX.r);
//     linspace(cpTh, CP_MIN.th, CP_MAX.th);
//
//     float swFt_R = 0.3;
//     float swFt_Th = 160.0/180.0 * PI;
//
//     std::vector<float> result;
//
//     for (size_t i = 0; i < numGrid; i++) {
//         for (size_t j = 0; j < numGrid; j++) {
//             States testPoint;
//             testPoint.cp.r = cpR[i];
//             testPoint.cp.th = cpTh[j];
//             testPoint.step.r = swFt_R;
//             testPoint.step.th = swFt_Th;
//             if (isZeroStepCapt(testPoint)) {
//                 result.push_back(cpR[i]);
//                 result.push_back(cpTh[j]);
//             }
//         }
//     }
//
//     for (size_t i = 0; i < result.size()/2; i++) {
//         fprintf(fp, "%lf, %lf\n", result[i*2 + 0], result[i*2 + 1]);
//     }
//
//     fclose(fp);
// }




























//
