#ifndef __capturability_H__
#define __capturability_H__

void linspace(float c[], float a, float b);
void cartesianProduct4(float set[], float a[], float b[], float c[], float d[] );

__device__ float linearEquation(float x, float ax, float ay, float bx, float by);

__device__ bool inSupportPolygon(float cpx, float cpy, float stx, float sty) ;

__device__ void calcStates( float *ncpx, float *ncpy, float *nstx, float *nsty,
                            float cpx, float cpy, float stx, float sty,
                            float dstx, float dsty );

__global__ void oneStepCapturabilityKernel(float *set_P1, float *set_O);

#else
#endif
