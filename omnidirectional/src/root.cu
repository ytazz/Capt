#include "capturability.h"

int main(void)
{
    cudaSetDevice(1);
    //Xorgで使用しているGPUはkernel計算に制限時間があるため

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

    HANDLE_ERROR(cudaMemcpy(dev_cpR, cpR, N_CP_R*sizeof(float),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_cpTh, cpTh, N_CP_TH*sizeof(float),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_stepR, stepR, N_FOOT_R*sizeof(float),
                            cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(dev_stepTh, stepTh, N_FOOT_TH*sizeof(float),
                            cudaMemcpyHostToDevice));

    Data *dataSet = new Data[N_STATE];
    initializing(dataSet, cpR, cpTh, stepR, stepTh);

    Data *dev_dataSet;
    HANDLE_ERROR(cudaMalloc((void **)&dev_dataSet, N_STATE*sizeof(Data)));
    HANDLE_ERROR(cudaMemcpy(dev_dataSet, dataSet, N_STATE*sizeof(Data),
                            cudaMemcpyHostToDevice));

    step_1<<<BPG,TPB>>>(dev_dataSet);

    // 2-step Capturable Basin
    step_N<<<BPG,TPB>>>(dev_dataSet, 2, dev_cpR, dev_cpTh, dev_stepR, dev_stepTh);

    // 3-step Capturable Basin
    step_N<<<BPG,TPB>>>(dev_dataSet, 3, dev_cpR, dev_cpTh, dev_stepR, dev_stepTh);

    // 4-step Capturable Basin
    step_N<<<BPG,TPB>>>(dev_dataSet, 4, dev_cpR, dev_cpTh, dev_stepR, dev_stepTh);


    HANDLE_ERROR(cudaMemcpy(dataSet, dev_dataSet, N_STATE*sizeof(Data),
    cudaMemcpyDeviceToHost));

    writeData(dataSet, "data.csv");

    cudaFree( dev_dataSet );
    cudaFree( dev_cpR );
    cudaFree( dev_cpTh );
    cudaFree( dev_stepR );
    cudaFree( dev_stepTh );

    delete [] dataSet;

    return 0;

}




















//
