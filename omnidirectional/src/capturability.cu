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
                    result[ind].c = FAILED;
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

void writeFile(std::vector<States> data, std::string str){
    FILE *fp;
    fp = fopen(str.c_str(), "w");

    for (long int i = 0; i < data.size(); i++) {
        fprintf(fp, "%lf, %lf, %lf, %lf, %d \n",
                data[i].cp.r, data[i].cp.th, data[i].step.r, data[i].step.th, data[i].c);
    }
    fclose(fp);
}

__global__ void step_0(States *statesSpace){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    while (tid < N_ENTIRE) {
        if (isZeroStepCapt(statesSpace[tid])) {
            statesSpace[tid].c = 0;
        }
        tid += blockDim.x * gridDim.x;
    }
}

__global__ void step_N(States *statesSpace, PolarCoord input, int n_step,
                       float *cpR, float *cpTh, float *stepR, float *stepTh){

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    States oneStepAfterState;

    while (tid < N_ENTIRE) {
        if (statesSpace[tid].c == FAILED) {
            oneStepAfterState = stepping(statesSpace[tid], input);
            if (isInPrevSet(statesSpace, oneStepAfterState,
                            n_step, cpR, cpTh, stepR, stepTh)) {
                statesSpace[tid].c = n_step;
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

__device__ bool isInPrevSet(States *statesSpace, States p, int n_step,
                            float cpR[], float cpTh[], float stepR[], float stepTh[]){
    if (p.cp.r >= CP_MAX_R) {
        return false;
    }else{
        States bound[16]; //2^4 = 16
        int flag[16];

        setBound(bound, p, cpR, cpTh, stepR, stepTh);

        for (size_t i = 0; i < 16; i++) {
            for (size_t j = 0; j < N_ENTIRE; j++) {
                if (statesSpace[j].cp.r == bound[i].cp.r &&
                    statesSpace[j].cp.th == bound[i].cp.th &&
                    statesSpace[j].step.r == bound[i].step.r &&
                    statesSpace[j].step.th == bound[i].step.th) {

                    flag[i] = statesSpace[j].c;
                    if (statesSpace[j].c == FAILED) {
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

    for (size_t i = 0; i < N_CP_R; i++) {
        if (cpR[i] >= p.cp.r) {
            ind_cpR = i;
            break;
        }
    }
    if (ind_cpR == 0) {
        ind_cpR = 1;
    }

    for (size_t i = 0; i < N_CP_TH; i++) {
        if (cpTh[i] >= p.cp.th) {
            ind_cpTh = i;
            break;
        }
    }
    if (ind_cpTh == 0) {
        ind_cpTh = 1;
    }

    for (size_t i = 0; i < N_FOOT_R; i++) {
        if (stepR[i] >= p.step.r) {
            ind_stepR = i;
            break;
        }
    }
    if (ind_stepR == 0) {
        ind_stepR = 1;
    }

    for (size_t i = 0; i < N_FOOT_TH; i++) {
        if (stepTh[i] >= p.step.th) {
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


   void testStepping (void){
       States test;

       test.cp.r = 0.3;
       test.cp.th = PI/2;
       test.step.r = 0.3;
       // test.step.th = PI/6.0;
       test.step.th = PI*5.0/6.0;

       // PolarCoord testinput = {0.3, PI*5.0/6.0};
       PolarCoord testinput = {0.3, 0.0};
       States oneStepAfterState;

       oneStepAfterState = stepping(test, testinput);
       std::cout << oneStepAfterState.cp.r <<',' << oneStepAfterState.cp.th
                 <<',' << oneStepAfterState.step.r <<',' << oneStepAfterState.step.th <<'\n';
   }

   void testBound(void) {
       States test;

       test.cp.r = CP_MIN_R;
       test.cp.th = CP_MIN_TH;
       test.step.r = FOOT_MIN_R;
       test.step.th = FOOT_MIN_TH;

       float cpR[N_CP_R], cpTh[N_CP_TH], stepR[N_FOOT_R], stepTh[N_FOOT_TH];
       linspace(cpR, CP_MIN_R, CP_MAX_R, N_CP_R);
       linspace(cpTh, CP_MIN_TH, CP_MAX_TH, N_CP_TH);
       linspace(stepR, FOOT_MIN_R, FOOT_MAX_R, N_FOOT_R);
       linspace(stepTh, FOOT_MIN_TH, FOOT_MAX_TH, N_FOOT_TH);


       States bound[16];

       setBound(bound, test, cpR, cpTh, stepR, stepTh);

       for (size_t i = 0; i < 16; i++) {
           std::cout << bound[i].cp.r <<',' << bound[i].cp.th
                     <<',' << bound[i].step.r <<',' << bound[i].step.th <<'\n';
       }

   }

   void prevset_test(void){
       States test;

       test.cp.r = 0.06;
       test.cp.th = 4.2;
       test.step.r = FOOT_MIN_R;
       test.step.th = FOOT_MIN_TH;

       float cpR[N_CP_R], cpTh[N_CP_TH], stepR[N_FOOT_R], stepTh[N_FOOT_TH];
       linspace(cpR, CP_MIN_R, CP_MAX_R, N_CP_R);
       linspace(cpTh, CP_MIN_TH, CP_MAX_TH, N_CP_TH);
       linspace(stepR, FOOT_MIN_R, FOOT_MAX_R, N_FOOT_R);
       linspace(stepTh, FOOT_MIN_TH, FOOT_MAX_TH, N_FOOT_TH);

       States *statesSpace = new States[N_ENTIRE];
       makeStatesSpace(statesSpace, cpR, cpTh, stepR, stepTh);
       States *dev_StatesSpace;
       HANDLE_ERROR(cudaMalloc((void **)&dev_StatesSpace, N_ENTIRE*sizeof(States)));
       HANDLE_ERROR(cudaMemcpy(dev_StatesSpace, statesSpace, N_ENTIRE*sizeof(States),
                               cudaMemcpyHostToDevice));


       // 0-step Capturable Basin
       step_0<<<BPG,TPB>>>(dev_StatesSpace);
       HANDLE_ERROR(cudaMemcpy(statesSpace, dev_StatesSpace, N_ENTIRE*sizeof(States),
                               cudaMemcpyDeviceToHost));

       int result;
       result = isInPrevSet(statesSpace, test, 0, cpR, cpTh, stepR, stepTh);
       printf("%d\n", result);


       cudaFree( dev_StatesSpace );
       delete [] statesSpace;
   }


 *******************************************************************************/
