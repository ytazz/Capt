#pragma once
#ifndef DATA_STRUCT_H
#define DATA_STRUCT_H
#include <iostream>
#include <string>
#include <vector>

using namespace std;

struct PolarCoord {
  float r, th;
};

struct State {
  PolarCoord icp;    // instantaneous capture point
  PolarCoord swf;    // current swing foot position
  int n;             //n-step capturable

  bool operator == (State b) {
    return (icp.r == b.icp.r)
           && (icp.th == b.icp.th)
           && (swf.r == b.swf.r)
           && (swf.th == b.swf.th);
  }
};

struct Input {
  PolarCoord dsf;    // desired swing foot position (in capture region)
  int n;             // n-step
};

// struct Data {
//  State state;    // a state
//  vector<Input> cr; // capture region
// };

struct Data {
  State state;
  Input input;
};
#endif // !
