#include "capturability.h"

int main(void)
{
    float cpR[N_CP_R], cpTh[N_CP_TH], stepR[N_FOOT_R], stepTh[N_FOOT_TH];
    linspace(cpR, CP_MIN_R, CP_MAX_R, N_CP_R);
    linspace(cpTh, CP_MIN_TH, CP_MAX_TH, N_CP_TH);
    linspace(stepR, FOOT_MIN_R, FOOT_MAX_R, N_FOOT_R);
    linspace(stepTh, FOOT_MIN_TH, FOOT_MAX_TH, N_FOOT_TH);

    float *dev_cpR, *dev_cpTh, *dev_stepR, *dev_stepTh;
    HANDLE_ERROR(cudaMalloc((void **)&dev_cpR, N_CP_R*sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_cpTh, N_CP_TH*sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_stepR, N_FOOT_R*sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)&dev_stepTh, N_FOOT_TH*sizeof(float)));
    HANDLE_ERROR(cudaMemcpy(dev_cpR, cpR, N_CP_R*sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_cpTh, cpTh, N_CP_TH*sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_stepR, stepR, N_FOOT_R*sizeof(float), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_stepTh, stepTh, N_FOOT_TH*sizeof(float), cudaMemcpyHostToDevice));

    States *statesSpace = new States[N_ENTIRE];
    makeStatesSpace(statesSpace, cpR, cpTh, stepR, stepTh);
    States *dev_StatesSpace;
    HANDLE_ERROR(cudaMalloc((void **)&dev_StatesSpace, N_ENTIRE*sizeof(States)));
    HANDLE_ERROR(cudaMemcpy(dev_StatesSpace, statesSpace, N_ENTIRE*sizeof(States),
                            cudaMemcpyHostToDevice));

    PolarCoord *inputSpace = new PolarCoord[N_INPUT];
    makeInputSpace(inputSpace, stepR, stepTh);
    PolarCoord *dev_InputSpace;
    HANDLE_ERROR(cudaMalloc((void **)&dev_InputSpace, N_INPUT*sizeof(PolarCoord)));
    HANDLE_ERROR(cudaMemcpy(dev_InputSpace, inputSpace, N_INPUT*sizeof(PolarCoord),
                            cudaMemcpyHostToDevice));

    // 0-step Capturable Basin
    step_0<<<BPG,TPB>>>(dev_StatesSpace);

    // 1-step Capturable Basin
    for (size_t i = 0; i < N_INPUT; i++) {
        step_N<<<BPG,TPB>>>(dev_StatesSpace, inputSpace[i], 1,
                            dev_cpR, dev_cpTh, dev_stepR, dev_stepTh);
    }

    // 2-step Capturable Basin
    for (size_t i = 0; i < N_INPUT; i++) {
        step_N<<<BPG,TPB>>>(dev_StatesSpace, inputSpace[i], 2,
                            dev_cpR, dev_cpTh, dev_stepR, dev_stepTh);
    }

    HANDLE_ERROR(cudaMemcpy(statesSpace, dev_StatesSpace, N_ENTIRE*sizeof(States),
                            cudaMemcpyDeviceToHost));


    std::vector<States> entire;
    for (size_t i = 0; i < N_ENTIRE; i++) {
        entire.push_back(statesSpace[i]);
    }

    writeFile(entire, "entire.csv");


    cudaFree( dev_StatesSpace );
    cudaFree( dev_InputSpace );
    cudaFree( dev_cpR );
    cudaFree( dev_cpTh );
    cudaFree( dev_stepR );
    cudaFree( dev_stepTh );

    delete [] statesSpace;
    delete [] inputSpace;


    return 0;

}
