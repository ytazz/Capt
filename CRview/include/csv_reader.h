#ifndef CSV_READER_H
#define CSV_READER_H

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include "../include/data_struct.h"

class CSVReader
{
  string fileName;
  string delimeter;
public:
  CSVReader(string file, string delm=",") : fileName(file), delimeter(delm){
  }
  vector<Data> getData();
  int readLine(FILE *fp, Data *buf);
};

#endif
