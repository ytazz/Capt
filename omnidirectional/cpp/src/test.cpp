#include "param.h"
#include <vector>


int main( void ) {
    float dstx = 1.0;
    float dsty = 1.0;
    float stx = 0.0;
    float sty = 0.0;
    float deltaT = MINIMUM_STEPPING_TIME
                   + sqrt((dstx-stx)*(dstx-stx) + (dsty-sty)*(dsty-sty))/FOOTVEL;
    printf("%lf\n", deltaT);
    return 0;
}
