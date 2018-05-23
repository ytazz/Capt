#include "src/capturability.h"

int main(void)
{
    float cpX[numGrid], cpY[numGrid], stepX[numGrid], stepY[numGrid];
    linspace(cpX, CP_MIN.x, CP_MAX.x);
    linspace(cpY, CP_MIN.y, CP_MAX.y);
    linspace(stepX, FOOT_MIN.x, FOOT_MAX.x);
    linspace(stepY, FOOT_MIN.y, FOOT_MAX.y);

    States *dev_statesSpace, *dev_hat_statesSpace;
    HANDLE_ERROR( cudaMalloc( (void **)&dev_statesSpace, N*sizeof(States) ) );
    HANDLE_ERROR( cudaMalloc( (void **)&dev_hat_statesSpace, N*sizeof(States) ) );

    States *statesSpace = new States[N];
    makeStatesSpace(statesSpace, cpX, cpY, stepX, stepY);

    HANDLE_ERROR( cudaMemcpy( dev_statesSpace, statesSpace, N*sizeof(States),
                              cudaMemcpyHostToDevice ) );

    transf<<<blocksPerGrid,threadsPerBlock>>>(dev_hat_statesSpace, dev_statesSpace);

    States *hat_statesSpace = new States[N];
    HANDLE_ERROR( cudaMemcpy( hat_statesSpace, dev_hat_statesSpace, N*sizeof(States),
                              cudaMemcpyDeviceToHost ) );

    writeFile(hat_statesSpace);



    // long int row = 0;
    // for (size_t i = 0; i < numGrid; i++) {
    //     for (size_t j = 0; j < numGrid; j++) {
    //         for (size_t k = 0; k < numGrid; k++) {
    //             for (size_t l = 0; l < numGrid; l++) {
    //                 printf("%lf, %lf, %lf, %lf\n",
    //                        hat_statesSpace[row].step.x, hat_statesSpace[row].step.y, hat_statesSpace[row].cp.x, hat_statesSpace[row].cp.y);
    //                 row++;
    //             }
    //         }
    //     }
    // }



    cudaFree( dev_statesSpace );
    cudaFree( dev_hat_statesSpace );
    delete [] statesSpace;
    delete [] hat_statesSpace;
    return 0;

}
