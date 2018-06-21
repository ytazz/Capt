/**
   \author GWANWOO KIM
 */
#include "capturability.h"

void linspace(float result[], float min, float max, int n) {
    if (max > min) {
        float h = (max - min)/(n - 1);
        for (size_t i = 0; i < n; i++) {
            result[i] = min + i*h;
        }
    }else{
        printf("%lf should be bigger than %lf \n", max, min);
    }
}

void makeStatesSpace(States* result,
                     float cpR[], float cpTh[], float stepR[], float stepTh[] ) {
    long int ind = 0;

    for (size_t i = 0; i < N_CP_R; i++) {
        for (size_t j = 0; j < N_CP_TH; j++) {
            for (size_t k = 0; k < N_FOOT_R; k++) {
                for (size_t l = 0; l < N_FOOT_TH; l++) {
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
    for (size_t i = 0; i < N_FOOT_R; i++) {
        for (size_t j = 0; j < N_FOOT_TH; j++) {
            result[ind].r = stepR[i];
            result[ind].th = stepTh[j];
            ind++;
        }
    }
}

void writeFile(States* data, int length, std::string str){
    FILE *fp;
    fp = fopen(str.c_str(), "w");

    for (long int i = 0; i < length; i++) {
        fprintf(fp, "%lf, %lf, %lf, %lf\n",
                data[i].cp.r, data[i].cp.th, data[i].step.r, data[i].step.th);
    }
    fclose(fp);
}

int getLength(States *temp, int prevLength){
    int length = 0;
    for (size_t i = 0; i < prevLength; i++) {
        if (temp[i].step.r != FAILED) {
            length++;
        }
    }
    return length;
}

void getSortedArray(States *array, States *temp, int prevLength){
    int ind = 0;
    for (size_t i = 0; i < prevLength; i++) {
        if (temp[i].step.r != FAILED) {
            array[ind] = temp[i];
            ind++;
        }
    }
}

// __global__ void step_0(States *result_set, States *statesSpace ){
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//
//     while (tid < N_ENTIRE) {
//         for (size_t i = 0; i < N_INPUT; i++) {
//             if (isZeroStepCapt(statesSpace[tid])) {
//                 result_set[tid] = statesSpace[tid];
//             }else{
//                 result_set[tid].step.r = FAILED;
//                 result_set[tid].step.th = FAILED;
//                 result_set[tid].cp.r = FAILED;
//                 result_set[tid].cp.th = FAILED;
//             }
//         }
//         tid += blockDim.x * gridDim.x;
//     }
// }

__global__ void step_0(States *statesSpace){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N_ENTIRE) {
        if (isZeroStepCapt(statesSpace[tid])) {
            statesSpace[tid].c = 0;
        }
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void step_1(States *result_set, States *statesSpace, PolarCoord *input ){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    States oneStepAfterState;

    while (tid < N_ENTIRE) {
        for (size_t i = 0; i < N_INPUT; i++) {
            oneStepAfterState = stepping(statesSpace[tid], input[i]);

            if (isZeroStepCapt(oneStepAfterState) || isZeroStepCapt(statesSpace[tid])) {
                result_set[tid] = statesSpace[tid];
            }else{
                result_set[tid].step.r = FAILED;
                result_set[tid].step.th = FAILED;
                result_set[tid].cp.r = FAILED;
                result_set[tid].cp.th = FAILED;
            }
        }
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void step_N(States *result_set, States *statesSpace, PolarCoord *input,
                       States *prevSet, int prevSetLength,
                       float *cpR, float *cpTh, float *stepR, float *stepTh){

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    States oneStepAfterState;

    while (tid < N_ENTIRE) {
        for (size_t i = 0; i < N_INPUT; i++) {
            oneStepAfterState = stepping(statesSpace[tid], input[i]);

            if (isInPrevSet(prevSet, prevSetLength, oneStepAfterState, cpR, cpTh, stepR, stepTh)) {
                result_set[tid] = statesSpace[tid];
            }else{
                result_set[tid].step.r = FAILED;
                result_set[tid].step.th = FAILED;
                result_set[tid].cp.r = FAILED;
                result_set[tid].cp.th = FAILED;
            }
        }
        tid += blockDim.x * gridDim.x;
    }
}

__device__ States stepping (States p0, PolarCoord u){
    PolarCoord hatCP;
    States p1;
    float tau;

    tau =  distanceTwoPolar(p0.step, u) / FOOTVEL + MINIMUM_STEPPING_TIME; //所要時間

    //所要時間後の状態
    hatCP.r = (p0.cp.r - FOOTSIZE) * expf(OMEGA * tau) + FOOTSIZE;
    hatCP.th = p0.cp.th;

    //reset
    p1.step.r = u.r;
    p1.step.th = PI - u.th;
    p1.cp.r = distanceTwoPolar(u, hatCP);
    float alpha = hatCP.r * cos(hatCP.th) - u.r * cos(u.th);
    float beta =  -hatCP.r * sin(hatCP.th) + u.r * sin(u.th);
    if (beta > 0) {
        p1.cp.th = acosf(alpha/p1.cp.r);
    }else{
        p1.cp.th = 2*PI - acosf(alpha/p1.cp.r);
    }

    return p1;
}

__device__ float distanceTwoPolar (PolarCoord a, PolarCoord b){
    return sqrtf(sq(a.r) + sq(b.r) - 2*a.r*b.r*cos(a.th-b.th));
}

__device__ bool isInPrevSet(States *prevSet, int lengthOfArray, States p,
                            float cpR[], float cpTh[], float stepR[], float stepTh[]){

    if (p.cp.r > CP_MAX_R) {
        return false;
    }else{
        States bound[16]; //2^4 = 16
        int flag[16];

        setBound(bound, p, cpR, cpTh, stepR, stepTh);

        for (size_t i = 0; i < lengthOfArray; i++) {
            for (size_t j = 0; j < 16; j++) {
                if (prevSet[i].cp.r == bound[j].cp.r &&
                    prevSet[i].cp.th == bound[j].cp.th &&
                    prevSet[i].step.r == bound[j].step.r &&
                    prevSet[i].step.th == bound[j].step.th) {
                    flag[j] = 1;
                }
            }
        }

        if (flag[0] == 1 && flag[1] == 1 && flag[2] == 1 && flag[3] == 1 &&
            flag[4] == 1 && flag[5] == 1 && flag[6] == 1&& flag[7] == 1 &&
            flag[8] == 1 && flag[9] == 1 && flag[10] == 1&& flag[11] == 1 &&
            flag[12] == 1 && flag[13] == 1 && flag[14] == 1&& flag[15] == 1) {
            return true;
        }else{
            return false;
        }
    }
}

__device__ bool isZeroStepCapt(States p){
    float threshold = FOOTSIZE;

    if (p.cp.r <= threshold) {
        return true;
    }else{
        return false;
    }
}

__device__ void setBound(States *bound, States p,
                         float cpR[], float cpTh[], float stepR[], float stepTh[]){

    int ind_cpR, ind_cpTh, ind_stepR, ind_stepTh;

    int i= 0;
    while (1) {
        if (cpR[i] > p.cp.r) {
            ind_cpR = i;
            break;
        }
        i++;
    }

    i = 0;
    while (1) {
        if (cpTh[i] > p.cp.th) {
            ind_cpTh = i;
            break;
        }
        i++;
    }

    i = 0;
    while (1) {
        if (stepR[i] > p.step.r) {
            ind_stepR = i;
            break;
        }
        i++;
    }

    i = 0;
    while (1) {
        if (stepTh[i] > p.step.th) {
            ind_stepTh = i;
            break;
        }
        i++;
    }

    int ind = 0;

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            for (size_t k = 0; k < 2; k++) {
                for (size_t l = 0; l < 2; l++) {
                    bound[ind].cp.r = cpR[ind_cpR-1 + i];
                    bound[ind].cp.th = cpTh[ind_cpTh-1 + j];
                    bound[ind].step.r = stepR[ind_stepR-1 + k];
                    bound[ind].step.th = stepTh[ind_stepTh-1 + l];
                    ind++;
                }
            }
        }
    }
}


/******************************************************************************
   多分破棄
   __device__ bool isZeroStepCapt(States p){
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
 *******************************************************************************/

/******************************************************************************
   保留

   __device__ float foot(float th){

   float b1 = PI/4;
   float b2 = PI*3/4;
   float b3 = PI*5/4;
   float b4 = PI*7/4;

   if (0 <= th && th < b1) {
   }else if (b1 <= th && th < b2 ){

   }else if(b2 <= th && th < b3){

   }else if(b3 <= th && th < b4){

   }else if(b4 <= th && th < 2*PI)

   }

   void testZero(void){
    FILE *fp;
    fp = fopen("test.csv", "w");

    float cpR[numGrid], cpTh[numGrid];
    linspace(cpR, CP_MIN.r, CP_MAX.r);
    linspace(cpTh, CP_MIN.th, CP_MAX.th);

    float swFt_R = 0.3;
    float swFt_Th = 160.0/180.0 * PI;

    std::vector<float> result;

    for (size_t i = 0; i < numGrid; i++) {
        for (size_t j = 0; j < numGrid; j++) {
            States testPoint;
            testPoint.cp.r = cpR[i];
            testPoint.cp.th = cpTh[j];
            testPoint.step.r = swFt_R;
            testPoint.step.th = swFt_Th;
            if (isZeroStepCapt(testPoint)) {
                result.push_back(cpR[i]);
                result.push_back(cpTh[j]);
            }
        }
    }

    for (size_t i = 0; i < result.size()/2; i++) {
        fprintf(fp, "%lf, %lf\n", result[i*2 + 0], result[i*2 + 1]);
    }

    fclose(fp);
   }

 *******************************************************************************/
