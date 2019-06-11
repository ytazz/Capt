#include "friction_filter.h"

namespace CA {

FrictionFilter::FrictionFilter(Capturability capturability)
    : capturability(capturability){};

FrictionFilter::~FrictionFilter(){};

void FrictionFilter::setCaptureRegion(std::vector<Input> capture_region) {
  for (size_t i = 0; i < capture_region.size(); i++) {
    this->capture_region.push_back(capture_region[i]);
  }
}

std::vector<Input> FrictionFilter::getCaptureRegion(State state, vec2_t com,
                                                    float mu) {
  std::vector<Input> modified_cr; // cr = capture region

  bool flag = false;
  float d[2];
  for (size_t i = 0; i < capture_region.size(); i++) {
    flag = false;
    d[0] = 0;
    modified_cr.push_back(capture_region[i]);
  }

  return modified_cr;
}

} // namespace CA