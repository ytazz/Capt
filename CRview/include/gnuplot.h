#pragma once
#ifndef GNUPLOT_H
#define GNUPLOT_H
#include <stdio.h>
#include <string>
#include <iostream>
#include <map>

using namespace std;

class gnuplot {
public:
  gnuplot();
  ~gnuplot();
  void operator () (const string &command);
  FILE *gp;
  map<int, string> int_color;
};

#endif // GNUPLOT_H
