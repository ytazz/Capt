/**
   \author GWANWOO KIM
 */
#include "robotParam.h"
#include "gpgpu.h"
#include <vector>
#include <limits>

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

void stateSpace(float set[], float a[], float b[], float c[], float d[] ) {
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

    long int N = (long int)numGrid*numGrid*numGrid*numGrid;
    int M = 4;
    float *dev_set_O;
    HANDLE_ERROR( cudaMalloc( (void **)&dev_set_O, N*M*sizeof(float) ) );
    float *set_O = new float[N*M];
    stateSpace(set_O, hat_cp_x, hat_cp_y, hat_step_x, hat_step_y);

    HANDLE_ERROR( cudaMemcpy( dev_hat_cp_x, hat_cp_x, numGrid * sizeof(float),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_hat_cp_y, hat_cp_y, numGrid * sizeof(float),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_hat_step_x, hat_step_x, numGrid * sizeof(float),
                              cudaMemcpyHostToDevice ) );
    HANDLE_ERROR( cudaMemcpy( dev_hat_step_y, hat_step_y, numGrid * sizeof(float),
                              cudaMemcpyHostToDevice ) );
      

    cudaFree( dev_hat_cp_x );
    cudaFree( dev_hat_cp_y );
    cudaFree( dev_hat_step_x );
    cudaFree( dev_hat_step_x );
    cudaFree( dev_set_O );
    delete [] set_O;
    return 0;
}
