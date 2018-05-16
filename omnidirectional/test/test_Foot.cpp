#include "param.h"
#include <vector>

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

float linearEquation(float x, float ax, float ay, float bx, float by)
{
    float y = (by - ay)/(bx - ax) * (x - ax) + ay;
    return y;
}

bool isInSupportPolygon(float cpx, float cpy, float stx, float sty) {
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


int main( void ) {
    FILE *fp;
    fp = fopen("0step_capturability.csv", "w");

    float x[numGrid], y[numGrid];
    linspace(x, cp_min_x, cp_max_x);
    linspace(y, cp_min_y, cp_max_y);

    float swFt_X = 0.1;
    float swFt_Y = 0.2;

    std::vector<float> result;

    for (size_t i = 0; i < numGrid; i++) {
        for (size_t j = 0; j < numGrid; j++) {
            if (isInSupportPolygon(x[i], y[j], swFt_X, swFt_Y)) {
                result.push_back(x[i]);
                result.push_back(y[j]);
            }
        }
    }

    for (size_t i = 0; i < result.size()/2; i++) {
        fprintf(fp, "%lf, %lf\n", result[i*2 + 0], result[i*2 + 1]);
    }

    fclose(fp);
    return 0;
}
