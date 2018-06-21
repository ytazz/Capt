#include "src/capturability.h"

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
    HANDLE_ERROR(cudaMemcpy(statesSpace, dev_StatesSpace, N_ENTIRE*sizeof(States),
                            cudaMemcpyDeviceToHost));
                            

    // 1-step Capturable Basin
    // States *temp1 = new States[N_ENTIRE];
    // States *dev_Temp1;
    // HANDLE_ERROR( cudaMalloc( (void **)&dev_Temp1, N_ENTIRE*sizeof(States) ) );
    // step_N<<<BPG,TPB>>>(dev_Temp1, dev_StatesSpace, dev_InputSpace,
    //                     set0, n0,
    //                     dev_cpR, dev_cpTh, dev_stepR, dev_stepTh);
    // HANDLE_ERROR( cudaMemcpy( temp1, dev_Temp1, N_ENTIRE*sizeof(States),
    //                           cudaMemcpyDeviceToHost ) );
    // int n1 = getLength(temp1, N_ENTIRE);
    // States *set1 = new States[n1];
    // getSortedArray(set1, temp1, N_ENTIRE);
    // cudaFree( dev_Temp1 );
    // delete [] temp1;


    writeFile(set0, n0, "zero_.csv");
    // writeFile(set1, n1, "one_n.csv");


    cudaFree( dev_StatesSpace );
    cudaFree( dev_InputSpace );
    cudaFree( dev_cpR );
    cudaFree( dev_cpTh );
    cudaFree( dev_stepR );
    cudaFree( dev_stepTh );

    delete [] statesSpace;
    delete [] inputSpace;
    delete [] set0;
    // delete [] set1;

    return 0;

}
