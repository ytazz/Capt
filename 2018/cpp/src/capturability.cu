/**
   \author GWANWOO KIM
 */
#include "param.h"
#include "common/nvidia.h"
#include <cuda.h>

using namespace std;

void linspace(float c[], float a, float b) {
    if (b > a) {
        float h = (b - a)/(numGrid - 1);
        for (size_t i = 0; i < numGrid; i++) {
            c[i] = a + i*h;
        }
    }else{
        printf("%lf should be bigger than %lf \n", b, a);
    }
}

void stateSpace4(float set[], float a[], float b[], float c[], float d[] ) {
    long int row = 0;
    for (size_t i = 0; i < numGrid; i++) {
        for (size_t j = 0; j < numGrid; j++) {
            for (size_t k = 0; k < numGrid; k++) {
                for (size_t l = 0; l < numGrid; l++) {
                    set[row*4 + 0] = a[l];
                    set[row*4 + 1] = b[k];
                    set[row*4 + 2] = c[j];
                    set[row*4 + 3] = d[i];
                    // printf("%lf, %lf, %lf, %lf\n",
                    //        set[row*4 + 0], set[row*4 + 1], set[row*4 + 2], set[row*4 + 3]);
                    row++;
                }
            }
        }
    }
}

__device__ bool inPreviousSet( float cpx, float cpy ){
    if (cpx > -0.1 && cpx < 0.1 && cpy > -0.1 && cpy < 0.1) {
        return 1;
    }else{
        return 0;
    }
}

__device__ void calcStates( float *ncpx, float *ncpy,
                            float cpx, float cpy, float stx, float sty,
                            float input_x, float input_y ) {
    float deltaT = sqrt((input_x - stx)*(input_x - stx) + (input_y-sty)*(input_y-sty))/1.4;
    *ncpx = (cpx - input_x)*expf(omega*deltaT) + input_x;
    *ncpy = (cpy - input_y)*expf(omega*deltaT) + input_y;
}


__global__ void kernel(float *oset, float *iset, float *hat_input_x, float *hat_input_y ) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float ncpx, ncpy;

    while (tid < N) {
        for (int i = 0; i < numGrid; i++) {
            for (int j = 0; j < numGrid; j++) {
                calcStates( &ncpx, &ncpy,
                            iset[tid*4 + 0], iset[tid*4 + 1], iset[tid*4 + 2], iset[tid*4 + 3],
                            hat_input_x[i], hat_input_y[j]);

                if (inPreviousSet(ncpx, ncpy)) {
                    oset[tid*4 + 0] = iset[tid*4 + 0];
                    oset[tid*4 + 1] = iset[tid*4 + 1];
                    oset[tid*4 + 2] = iset[tid*4 + 2];
                    oset[tid*4 + 3] = iset[tid*4 + 3];
                }
            }
        }
        tid += blockDim.x * gridDim.x;
    }

}

int main (void) {
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
    stateSpace4(set_O, hat_cp_x, hat_cp_y, hat_step_x, hat_step_y);

    HANDLE_ERROR( cudaMemcpy( dev_hat_step_x, hat_step_x, numGrid * sizeof(float),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_hat_step_y, hat_step_y, numGrid * sizeof(float),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_set_O, set_O, N*M*sizeof(float),
                              cudaMemcpyHostToDevice ) );

    kernel<<<blocksPerGrid,threadsPerBlock>>>(dev_set_P1, dev_set_O,
                                              dev_hat_step_x, dev_hat_step_y);

    HANDLE_ERROR( cudaMemcpy( set_P1, dev_set_P1, N*M*sizeof(float),
                              cudaMemcpyDeviceToHost ) );

    long int row = 0;
    for (size_t i = 0; i < numGrid; i++) {
        for (size_t j = 0; j < numGrid; j++) {
            for (size_t k = 0; k < numGrid; k++) {
                for (size_t l = 0; l < numGrid; l++) {
                    printf("%lf, %lf, %lf, %lf\n",
                           set_P1[row*4 + 0], set_P1[row*4 + 1], set_P1[row*4 + 2], set_P1[row*4 + 3]);
                    row++;
                }
            }
        }
    }

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
