#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <iostream>

struct Point {
    double x, y;

    bool operator <(const Point &p) const {
        return x < p.x || (x == p.x && y < p.y);
    }
    //sortのため
};

double cross(const Point &O, const Point &A, const Point &B)
{
    return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}

// Returns a list of points on the convex hull in counter-clockwise order.
// Note: the last point in the returned list is the same as the first one.
thrust::host_vector<Point> convex_hull(thrust::host_vector<Point> P)
{
    size_t n = P.size(), k = 0;
    if (n <= 3) return P;
    thrust::host_vector<Point> H(2*n);

    // Sort points lexicographically
    thrust::sort(P.begin(), P.end());

    // Build lower hull
    for (size_t i = 0; i < n; ++i) {
        while (k >= 2 && cross(H[k-2], H[k-1], P[i]) <= 0) k--;
        H[k++] = P[i];
    }


    // Build upper hull
    for (size_t i = n-1, t = k+1; i > 0; --i) {
        while (k >= t && cross(H[k-2], H[k-1], P[i-1]) <= 0) k--;
        H[k++] = P[i-1];
    }

    H.resize(k-1);
    return H;
}


int main(void)
{
    thrust::host_vector<Point> D(8);
    thrust::host_vector<Point> R;

    // // initialize individual elements
    D[0] = {
        0.0,
        2.0
    };
    D[1] = {
        1.0,
        2.0
    };
    D[2] = {
        0.0,
        1.0
    };
    D[3] = {
        1.0,
        1.0
    };
    D[4] = {
        -1.0,
        1.0
    };
    D[5] = {
        -2.0,
        1.0
    };
    D[6] = {
        -2.0,
        2.0
    };
    D[7] = {
        -1.0,
        2.0
    };

    // print contents of H
    for(int i = 0; i < D.size(); i++)
        std::cout << "x:" << D[i].x << " y:" << D[i].y << std::endl;

    std::cout << "-----------------------------------" << std::endl;

    R = convex_hull(D);
    for(int i = 0; i < R.size(); i++)
        std::cout << "x:" << R[i].x << " y:" << R[i].y << std::endl;


    return 0;
}
