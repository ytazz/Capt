#ifndef __CA_H__
#define __CA_H__

#include "analysis.h"
#include "capturability.h"
#include "friction_filter.h"
#include "grid.h"
#include "input.h"
#include "kinematics.h"
#include "loader.h"
#include "model.h"
#include "monitor.h"
#include "param.h"
#include "pendulum.h"
#include "planning.h"
#include "polygon.h"
#include "state.h"
#include "swing_foot.h"
#include "trajectory.h"
#include "vector.h"

namespace CA {

enum Phase { DSP, SSP_R, SSP_L };

} // namespace CA

#endif // __CA_H__