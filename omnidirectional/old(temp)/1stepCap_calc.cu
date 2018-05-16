#include "src/capturability.h"
#include "src/param.h"
#include "src/common/nvidia.h"
#include <vector>

int main (void) {
    FILE *fp;
    fp = fopen("1step_capturability.csv", "w");

    float *dev_hat_cp_x, *dev_hat_cp_y, *dev_hat_step_x, *dev_hat_step_y;
    HANDLE_ERROR( cudaMalloc( (void **)&dev_hat_cp_x, numGrid * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void **)&dev_hat_cp_y, numGrid * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void **)&dev_hat_step_x, numGrid * sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void **)&dev_hat_step_y, numGrid * sizeof(float) ) );

    float hat_cp_x[numGrid], hat_cp_y[numGrid], hat_step_x[numGrid], hat_step_y[numGrid];
    linspace(hat_cp_x, cp_min_x, cp_max_x);
    linspace(hat_cp_y, cp_min_y, cp_max_y);
    linspace(hat_step_x, step_min_x, step_max_x);
    linspace(hat_step_y, step_min_y, step_max_y);

    float *dev_set_O, *dev_set_P1;
    HANDLE_ERROR( cudaMalloc( (void **)&dev_set_O, N*M*sizeof(float) ) );
    HANDLE_ERROR( cudaMalloc( (void **)&dev_set_P1, N*M*sizeof(float) ) );
    float *set_O = new float[N*M];
    float *set_P1 = new float[N*M];
    cartesianProduct4(set_O, hat_cp_x, hat_cp_y, hat_step_x, hat_step_y);

    HANDLE_ERROR( cudaMemcpy( dev_hat_step_x, hat_step_x, numGrid * sizeof(float),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_hat_step_y, hat_step_y, numGrid * sizeof(float),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_set_O, set_O, N*M*sizeof(float),
                              cudaMemcpyHostToDevice ) );

    oneStepCapturabilityKernel<<<blocksPerGrid,threadsPerBlock>>>(dev_set_P1, dev_set_O);

    HANDLE_ERROR( cudaMemcpy( set_P1, dev_set_P1, N*M*sizeof(float),
                              cudaMemcpyDeviceToHost ) );

    // long int row = 0;
    // for (size_t i = 0; i < numGrid; i++) {
    //     for (size_t j = 0; j < numGrid; j++) {
    //         for (size_t k = 0; k < numGrid; k++) {
    //             for (size_t l = 0; l < numGrid; l++) {
    //                 printf("%lf, %lf, %lf, %lf\n",
    //                        set_P1[row*4 + 0], set_P1[row*4 + 1], set_P1[row*4 + 2], set_P1[row*4 + 3]);
    //                 row++;
    //             }
    //         }
    //     }
    // }

    std::vector<float> set_P1_out;
    for (size_t i = 0; i < N; i++) {
        if (set_P1[i*4 + 0] == 0.0 &&
            set_P1[i*4 + 1] == 0.0 &&
            set_P1[i*4 + 2] == 0.0 &&
            set_P1[i*4 + 3] == 0.0) {
        }else{
            set_P1_out.push_back(set_P1[i*4 + 0]);
            set_P1_out.push_back(set_P1[i*4 + 1]);
            set_P1_out.push_back(set_P1[i*4 + 2]);
            set_P1_out.push_back(set_P1[i*4 + 3]);
        }
    }

    for (size_t i = 0; i < set_P1_out.size()/4; i++) {
        fprintf(fp, "%lf, %lf, %lf, %lf\n",
                set_P1_out[i*4 + 0],
                set_P1_out[i*4 + 1],
                set_P1_out[i*4 + 2],
                set_P1_out[i*4 + 3]);
    }

    fclose(fp);


    cudaFree( dev_hat_cp_x );
    cudaFree( dev_hat_cp_y );
    cudaFree( dev_hat_step_x );
    cudaFree( dev_hat_step_x );
    cudaFree( dev_set_O );
    cudaFree( dev_set_P1 );
    delete [] set_O;
    delete [] set_P1;
    return 0;
}
