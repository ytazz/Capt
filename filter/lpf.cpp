#include <iostream>
#include <math.h>
#include <stdio.h>

using namespace std;

int main(int argc, char const *argv[]) {

  const int num_data = 1000;
  const double k = 1;

  double real[num_data];
  double data[num_data];
  double lpf[num_data];
  double buf[2];
  FILE *fp;

  if ((fp = fopen("data.csv", "r")) == NULL) {
    printf("Error: Couldn't find the file data.csv\n");
    exit(EXIT_FAILURE);
  } else {
    int i = 0;
    while (fscanf(fp, "%lf,%lf", &buf[0], &buf[1]) != EOF) {
      real[i] = buf[0];
      data[i] = buf[1];
      if (i == 0)
        lpf[i] = data[0];
      else
        lpf[i] = k * lpf[i - 1] + (1 - k) * data[i];

      i++;
    }
    fclose(fp);
  }

  FILE *fp_out = fopen("LPF.csv", "w");
  for (int i = 0; i < num_data; i++) {
    fprintf(fp_out, "%d,", i);
    fprintf(fp_out, "%lf,", real[i]);
    fprintf(fp_out, "%lf,", data[i]);
    fprintf(fp_out, "%lf\n", lpf[i]);
  }

  return 0;
}