/**
   \author GWANWOO KIM
 */
#include "param.h"
#include "common/nvidia.h"
#include <cuda.h>
#include <vector>

using namespace std;

//ncpx, ncpy : next capture point x, y
//dstx, dsty : desired step x, y

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

void cartesianProduct4(float set[], float a[], float b[], float c[], float d[] ) {
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

__device__ float linearEquation(float x, float ax, float ay, float bx, float by)
{
    float y = (by - ay)/(bx - ax) * (x - ax) + ay;
    return y;
}

__device__ bool inSupportPolygon(float cpx, float cpy, float stx, float sty) {
    /*
       swing footの目標着地点と支持脚の間(支持多角形の中に)にCapture Pointが存在すれば追加的な踏み出し不要
     */
    float supportFoot[4][2];
    supportFoot[0][0] = -XFOOTSIZE;
    supportFoot[0][1] = -YFOOTSIZE;
    supportFoot[1][0] = -XFOOTSIZE;
    supportFoot[1][1] = +YFOOTSIZE;
    supportFoot[2][0] = +XFOOTSIZE;
    supportFoot[2][1] = -YFOOTSIZE;
    supportFoot[3][0] = +XFOOTSIZE;
    supportFoot[3][1] = +YFOOTSIZE;

    float swingFoot[4][2];
    swingFoot[0][0] = stx - XFOOTSIZE;
    swingFoot[0][1] = sty - YFOOTSIZE;
    swingFoot[1][0] = stx - XFOOTSIZE;
    swingFoot[1][1] = sty + YFOOTSIZE;
    swingFoot[2][0] = stx + XFOOTSIZE;
    swingFoot[2][1] = sty - YFOOTSIZE;
    swingFoot[3][0] = stx + XFOOTSIZE;
    swingFoot[3][1] = sty + YFOOTSIZE;

    if (stx > 0) {
        if (supportFoot[0][0] < cpx && swingFoot[3][0] > cpx &&
            supportFoot[0][1] < cpy && swingFoot[3][1] > cpy) {
            float y1, y2;
            y1 = linearEquation(cpx, supportFoot[2][0], supportFoot[2][1],
                                swingFoot[2][0], swingFoot[2][1]);
            y2 = linearEquation(cpx, supportFoot[1][0], supportFoot[1][1],
                                swingFoot[1][0], swingFoot[1][1]);
            if (y1 < cpy && y2 > cpy) {
                return 1;
            }else{
                return 0;
            }
        }else{
            return 0;
        }
    } else if (stx == 0.0) {
        if (supportFoot[0][0] < cpx && swingFoot[3][0] > cpx &&
            supportFoot[0][1] < cpy && swingFoot[3][1] > cpy) {
            return 1;
        }else{
            return 0;
        }
    }else{
        if (swingFoot[1][0] < cpx && supportFoot[2][0] > cpx &&
            supportFoot[2][1] < cpy && swingFoot[1][1] > cpy) {
            float y1, y2;
            y1 = linearEquation(cpx, supportFoot[0][0], supportFoot[0][1],
                                swingFoot[0][0], swingFoot[0][1]);
            y2 = linearEquation(cpx, supportFoot[3][0], supportFoot[3][1],
                                swingFoot[3][0], swingFoot[3][1]);
            if (y1 < cpy && y2 > cpy) {
                return 1;
            }else{
                return 0;
            }
        }else{
            return 0;
        }
    }
}

__device__ void calcStates( float *ncpx, float *ncpy, float *nstx, float *nsty,
                            float cpx, float cpy, float stx, float sty,
                            float dstx, float dsty ) {

    float deltaT = MINIMUM_STEPPING_TIME
                   + sqrtf((dstx-stx)*(dstx-stx) + (dsty-sty)*(dsty-sty))/FOOTVEL;

    *ncpx = cpx*expf(OMEGA*deltaT) - dstx;
    *ncpy = -(cpy*expf(OMEGA*deltaT) - dsty);
    *nstx = -dstx;
    *nsty = dsty;
}

__global__ void oneStepCapturabilityKernel(float *set_P1, float *set_O) {

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float ncpx, ncpy, nstx, nsty;

    while (tid < N) {
        calcStates( &ncpx, &ncpy, &nstx, &nsty,
                    set_O[tid*4 + 0], set_O[tid*4 + 1],
                    0.1, 0.2,
                    set_O[tid*4 + 2], set_O[tid*4 + 3]);

        if (inSupportPolygon(ncpx, ncpy, nstx, nsty)) {
            set_P1[tid*4 + 0] = set_O[tid*4 + 0];
            set_P1[tid*4 + 1] = set_O[tid*4 + 1];
            set_P1[tid*4 + 2] = set_O[tid*4 + 2];
            set_P1[tid*4 + 3] = set_O[tid*4 + 3];
        }else{
            set_P1[tid*4 + 0] = 0.0;
            set_P1[tid*4 + 1] = 0.0;
            set_P1[tid*4 + 2] = 0.0;
            set_P1[tid*4 + 3] = 0.0;
        }
        tid += blockDim.x * gridDim.x;
    }
}


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
