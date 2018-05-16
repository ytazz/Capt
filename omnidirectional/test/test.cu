#include <../src/capturability.h>
#include <../src/param.h>
#include <iostream>

int main(void)
{
    TwoDim p1 = {1.0, 1.0};
    float resultX;
    float resultY;

    rotation_inv(&resultX, &resultY, p1.x, p1.y, 45*PI/180);

    std::cout << resultX << ',' << resultY << '\n';

    return 0;
}
