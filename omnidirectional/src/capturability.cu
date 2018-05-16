/**
   \author GWANWOO KIM
 */
#include "param.h"
#include "common/nvidia.h"
#include <cuda.h>
#include <vector>
#include <math.h>

using namespace std;

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

void cartesianProduct_4(float result[],
                        float arr1[], float arr2[], float arr3[], float arr4[] ) {
    long int row = 0;
    for (size_t i = 0; i < numGrid; i++) {
        for (size_t j = 0; j < numGrid; j++) {
            for (size_t k = 0; k < numGrid; k++) {
                for (size_t l = 0; l < numGrid; l++) {
                    result[row*4 + 0] = arr1[l];
                    result[row*4 + 1] = arr2[k];
                    result[row*4 + 2] = arr3[j];
                    result[row*4 + 3] = arr4[i];
                    row++;
                }
            }
        }
    }
}

void rotation_inv(float *resultX, float *resultY, float x, float y, float theta) {
    *resultX = x*cos(theta) + y*sin(theta);
    *resultY = x*(-sin(theta)) + y*cos(theta);
}
