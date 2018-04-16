/**
   \author GWANWOO KIM
 */
#include "robotParam.h"
#include <vector>
#include <limits>

using namespace std;

int main (void) {
    long int N =
        (long int)numGrid*numGrid*numGrid*numGrid*numGrid*numGrid;
    printf("%ld\n", N);

    int max1 = std::numeric_limits<int>::max();
    long int max2 = std::numeric_limits<long int>::max();
    long long int max3 = std::numeric_limits<long long int>::max();
    unsigned long long int max4 = std::numeric_limits<unsigned long long int>::max();
    printf("%d\n", max1);
    printf("%ld\n", sizeof(int));
    printf("%ld\n", max2);
    printf("%ld\n", sizeof(long int));
    printf("%lld\n", max3);
    printf("%ld\n", sizeof(long long int));
    printf("%llu\n", max4);
    printf("%ld\n", sizeof(unsigned long long int));
    return 0;
}
