#include <../src/capturability.h>
#include <iostream>

int main(void)
{
    TwoDim p1 = {1.0, 0.0};
    TwoDim r1;

    rotation_inv(&r1, p1, 45*PI/180);

    std::cout << r1.x << ',' << r1.y << '\n';
    std::cout << calcTheta(p1) * 180/PI << '\n';


    return 0;
}
