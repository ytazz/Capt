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

void makeGridsTable(float cpR[], float cpTh[], float stepR[], float stepTh[]){
    FILE *fp;
    fp = fopen("gridsTable.csv", "w");

    for (int i = 0; i < N_CP_R; i++) {
        fprintf(fp, "%lf, %lf, %lf, %lf \n",
                cpR[i], cpTh[i], stepR[i], stepTh[i]);
    }
    fclose(fp);

}

void initializing(Data *dataSet,
                  float cpR[], float cpTh[], float stepR[], float stepTh[]){
    long int a = 0;
    int b = 0;


    for (size_t i = 0; i < N_CP_R; i++) {
        for (size_t j = 0; j < N_CP_TH; j++) {
            for (size_t k = 0; k < N_FOOT_R; k++) {
                for (size_t l = 0; l < N_FOOT_TH; l++) {
                    dataSet[a].state.icp.r = cpR[i];
                    dataSet[a].state.icp.th = cpTh[j];
                    dataSet[a].state.swf.r = stepR[k];
                    dataSet[a].state.swf.th = stepTh[l];
                    dataSet[a].n = FAILED;
                    b = 0;
                    for (size_t n = 0; n < N_FOOT_R; n++) {
                        for (size_t m = 0; m < N_FOOT_TH; m++) {
                            dataSet[a].input[b].step.r = stepR[n];
                            dataSet[a].input[b].step.th = stepTh[m];
                            dataSet[a].input[b].c_r = FAILED;
                            b++;
                        }
                    }
                    a++;
                }
            }
        }
    }
}

void writeData(Data* data, std::string str){
    FILE *fp;
    fp = fopen(str.c_str(), "w");

    for (long int i = 0; i < N_STATE*N_INPUT; i++) {
        int a = i/N_INPUT;
        int b = i%N_INPUT;
        if (data[a].input[b].c_r != FAILED) {
            fprintf(fp, "%lf, %lf, %lf, %lf, %d, %lf, %lf, %d \n",
                    data[a].state.icp.r, data[a].state.icp.th,
                    data[a].state.swf.r, data[a].state.swf.th, data[a].n,
                    data[a].input[b].step.r, data[a].input[b].step.th,
                    data[a].input[b].c_r);
        }
    }
    fclose(fp);
}

__global__ void step_1(Data *dataSet){

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    State oneStepAfterState;

    while (tid < (N_STATE*N_INPUT)) {
        int a = tid/N_INPUT;
        int b = tid%N_INPUT;

        oneStepAfterState = stepping(dataSet[a].state, dataSet[a].input[b].step);

        if (isZeroStepCapt(oneStepAfterState)) {
            dataSet[a].input[b].c_r = 1;
            dataSet[a].n = 1;

        }
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void step_N(Data *dataSet, int n_step,
                       float *cpR, float *cpTh, float *stepR, float *stepTh){

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    State oneStepAfterState;

    while (tid < (N_STATE*N_INPUT)) {
        int a = tid/N_INPUT;
        int b = tid%N_INPUT;

        if (dataSet[a].input[b].c_r == FAILED) {
            oneStepAfterState = stepping(dataSet[a].state, dataSet[a].input[b].step);

            if (isInPrevSet(dataSet, oneStepAfterState,
                            n_step, cpR, cpTh, stepR, stepTh)) {
                dataSet[a].input[b].c_r = n_step;
                if (dataSet[a].n == FAILED) {
                    dataSet[a].n = n_step;
                }
            }
        }
        tid += blockDim.x * gridDim.x;
    }
}

__device__ State stepping (State p0, PolarCoord u){
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

    return p1;
}

__device__ float distanceTwoPolar (PolarCoord a, PolarCoord b){
    return sqrtf(sq(a.r) + sq(b.r) - 2*a.r*b.r*cos(a.th-b.th));
}

__device__ bool isInPrevSet(Data *dataSet, State p, int n_step,
                            float cpR[], float cpTh[], float stepR[], float stepTh[]){
    if (p.icp.r >= CP_MAX_R) {
        return false;
    }else{
        State bound[16]; //2^4 = 16
        int flag[16];

        setBound(bound, p, cpR, cpTh, stepR, stepTh);

        for (size_t i = 0; i < 16; i++) {
            for (size_t j = 0; j < N_STATE; j++) {
                if (dataSet[j].state.icp.r == bound[i].icp.r &&
                    dataSet[j].state.icp.th == bound[i].icp.th &&
                    dataSet[j].state.swf.r == bound[i].swf.r &&
                    dataSet[j].state.swf.th == bound[i].swf.th) {

                    flag[i] = dataSet[j].n;
                    if (dataSet[j].n == FAILED) {
                        return false;
                    }
                    break;
                }
            }
        }
        if (flag[0] < n_step && flag[1] < n_step && flag[2] < n_step && flag[3] < n_step &&
            flag[4] < n_step && flag[5] < n_step && flag[6] < n_step && flag[7] < n_step &&
            flag[8] < n_step && flag[9] < n_step && flag[10] < n_step && flag[11] < n_step &&
            flag[12] < n_step && flag[13] < n_step && flag[14] < n_step && flag[15] < n_step) {
            return true;
        }else{
            return false;
        }
    }
}

__device__ bool isZeroStepCapt(State p){
    float threshold = FOOTSIZE;

    if (p.icp.r <= threshold) {
        return true;
    }else{
        return false;
    }
}

__device__ void setBound(State *bound, State p,
                         float cpR[], float cpTh[], float stepR[], float stepTh[]){

    int ind_cpR, ind_cpTh, ind_stepR, ind_stepTh;

    for (size_t i = 0; i < N_CP_R; i++) {
        if (cpR[i] >= p.icp.r) {
            ind_cpR = i;
            break;
        }
    }
    if (ind_cpR == 0) {
        ind_cpR = 1;
    }

    for (size_t i = 0; i < N_CP_TH; i++) {
        if (cpTh[i] >= p.icp.th) {
            ind_cpTh = i;
            break;
        }
    }
    if (ind_cpTh == 0) {
        ind_cpTh = 1;
    }

    for (size_t i = 0; i < N_FOOT_R; i++) {
        if (stepR[i] >= p.swf.r) {
            ind_stepR = i;
            break;
        }
    }
    if (ind_stepR == 0) {
        ind_stepR = 1;
    }

    for (size_t i = 0; i < N_FOOT_TH; i++) {
        if (stepTh[i] >= p.swf.th) {
            ind_stepTh = i;
            break;
        }
    }
    if (ind_stepTh == 0) {
        ind_stepTh = 1;
    }

    int ind = 0;

    for (size_t i = 0; i < 2; i++) {
        for (size_t j = 0; j < 2; j++) {
            for (size_t k = 0; k < 2; k++) {
                for (size_t l = 0; l < 2; l++) {
                    bound[ind].icp.r = cpR[ind_cpR-1 + i];
                    bound[ind].icp.th = cpTh[ind_cpTh-1 + j];
                    bound[ind].swf.r = stepR[ind_stepR-1 + k];
                    bound[ind].swf.th = stepTh[ind_stepTh-1 + l];
                    ind++;
                }
            }
        }
    }
}
