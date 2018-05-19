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
                     float stepX[], float stepY[], float cpX[], float cpY[] ) {
    long int row = 0;
    for (size_t i = 0; i < numGrid; i++) {
        for (size_t j = 0; j < numGrid; j++) {
            for (size_t k = 0; k < numGrid; k++) {
                for (size_t l = 0; l < numGrid; l++) {
                    result[row].step.x = stepX[l];
                    result[row].step.y = stepY[k];
                    result[row].cp.x = cpX[j];
                    result[row].cp.y = cpY[i];
                    row++;
                }
            }
        }
    }
}

void writeFile(States* data){
    FILE *fp;
    fp = fopen("csv/statesSpace.csv", "w");

    for (long int i = 0; i < N; i++) {
        fprintf(fp, "%lf, %lf, %lf, %lf\n",
                data[i].step.x, data[i].step.y, data[i].cp.x, data[i].cp.y);
    }
    fclose(fp);
}

__device__ TwoDim rotation_inv(TwoDim in, float theta) {
    TwoDim result;
    result.x = in.x*cos(theta) + in.y*sin(theta);
    result.y = in.x*(-sin(theta)) + in.y*cos(theta);
    return result;
}


__device__ float calcTheta (TwoDim p) {
    if (p.x == 0 && p.y == 0) {
        return 0.0f;
    }else{
        if (p.x > 0.0) {
            return -PI + acosf(-p.y/sqrtf(p.x*p.x + p.y*p.y));
        }else{
            return acosf(p.y/sqrtf(p.x*p.x + p.y*p.y));
        }
    }
}

__global__ void transf(States *hat_set, States *set ){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float theta;

    while (tid < N) {
        theta = calcTheta(set[tid].cp);
        hat_set[tid].cp = rotation_inv(set[tid].cp, theta);
        hat_set[tid].step = rotation_inv(set[tid].step, theta);
        tid += blockDim.x * gridDim.x;
    }
}

__device__ int zeroStepCapt(States p){
    if (p.step.y < p.cp.y - FOOTSIZE ) {
        return false;
    }else{
        float r;
        r = fabsf(p.step.x * p.cp.y) / sqrtf(p.step.x*p.step.x + p.step.y*p.step.y);
        if (r < FOOTSIZE) {
            return true;
        }else{
            return false;
        }
    }
}



// void nearbyZero(float *a) {
//     float threshold = 0.0001;
//     if (abs(*a) < threshold) {
//         *a = 0.0;
//     }
// }
