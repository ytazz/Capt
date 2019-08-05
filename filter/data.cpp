#include <iostream>
#include <math.h>
#include <stdio.h>

using namespace std;

int main(int argc, char const *argv[]) {

  const int num_data = 1000;
  const int err_type = 100;
  const double err_scale = 0.1;

  double real[num_data];
  for (int i = 0; i < num_data; i++) {
    double x = (double)i / num_data * 10;
    real[i] = 1 / (1.0 + exp(-x)) - 0.5;
  }

  double err[num_data];
  for (int i = 0; i < num_data; i++) {
    err[i] = err_scale * (rand() % err_type - err_type / 2.0) / err_type;
  }

  double data[num_data];
  for (int i = 0; i < num_data; i++) {
    data[i] = real[i] + err[i];
  }

  FILE *fp = fopen("data.csv", "w");
  for (int i = 0; i < num_data; i++) {
    fprintf(fp, "%lf,", real[i]);
    fprintf(fp, "%lf\n", data[i]);
  }

  return 0;
}