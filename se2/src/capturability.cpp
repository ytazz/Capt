#define _CRT_SECURE_NO_WARNINGS

#include "capturability.h"

#include <limits>
#include <map>
#include <set>
using namespace std;

namespace Capt {

const int nmax = 10;

const float inf = numeric_limits<float>::max();
const float eps = 1.0e-5f;

CaptureState::CaptureState(int _swf_id, int _icp_id, int _nstep, Capturability* cap){
  swf_id = _swf_id;
  icp_id = _icp_id;
  nstep  = _nstep;

  vec3_t swf = cap->grid->xyz[cap->swf_to_xyz[swf_id]];
  vec2_t icp = cap->grid->xy [icp_id];
  swf_to_icp_id = cap->grid->xy.toIndex(cap->grid->xy.round(vec2_t(icp.x() - swf.x(), icp.y() - swf.y())));
}

Capturability::Capturability(Model* model, Param* param) {
  this->model    = model;
  this->param    = param;

  grid  = new Grid (param);
  swing = new Swing(model);

  model->read(&g, "gravity");
  model->read(&h, "com_height");
  T = sqrt(h/g);

  // exc
  param->read(&swf_x.min, "swf_x_min");
  param->read(&swf_x.max, "swf_x_max");
  param->read(&swf_y.min, "swf_y_min");
  param->read(&swf_y.max, "swf_y_max");
  // exc
  param->read(&exc_x.min, "exc_x_min");
  param->read(&exc_x.max, "exc_x_max");
  param->read(&exc_y.min, "exc_y_min");
  param->read(&exc_y.max, "exc_y_max");
  // cop
  param->read(&cop_x.min, "cop_x_min");
  param->read(&cop_x.max, "cop_x_max");
  param->read(&cop_y.min, "cop_y_min");
  param->read(&cop_y.max, "cop_y_max");
  // icp
  param->read(&icp_x.min, "icp_x_min");
  param->read(&icp_x.max, "icp_x_max");
  param->read(&icp_y.min, "icp_y_min");
  param->read(&icp_y.max, "icp_y_max");

  cap_basin.resize(nmax);

  step_weight = 1.0f;
  swf_weight  = 1.0f;
  icp_weight  = 1.0f;
}

Capturability::~Capturability() {
}

bool Capturability::isSteppable(vec2_t swf){
  // steppable if swf is inside swing region and outside the exc region
  return  ( (swf_x.min - eps <= swf.x() && swf.x() <= swf_x.max + eps) &&
            (swf_y.min - eps <= swf.y() && swf.y() <= swf_y.max + eps) )
             &&
         !( (exc_x.min + eps <= swf.x() && swf.x() <= exc_x.max - eps) &&
            (exc_y.min + eps <= swf.y() && swf.y() <= exc_y.max - eps) );
}

bool Capturability::isInsideSupport(vec2_t cop, float margin){
  return (cop.x() >= cop_x.min - margin &&
          cop.x() <= cop_x.max + margin &&
          cop.y() >= cop_y.min - margin &&
          cop.y() <= cop_y.max + margin );
}

Input Capturability::calcInput(const State& st, const State& stnext){
  Input in;
  in.swf = vec2_t(-stnext.swf.x(), stnext.swf.y());
  swing->set(st.swf, vec2Tovec3(in.swf));
  float tau   = swing->getDuration();
  float alpha = exp(tau/T);
  vec2_t diff(stnext.icp.x() - stnext.swf.x(), -(stnext.icp.y() - stnext.swf.y()));
  in.cop = (1.0f/(1.0f - alpha))*(diff - alpha*st.icp);
  return in;
}

void Capturability::calcFeasibleIcpRange(int swf_id, const CaptureState& csnext, pair<vec2_t, vec2_t>& icp_range){
  int tau_id = duration_map[swf_to_xyz.size()*swf_id + csnext.swf_id];
  icp_range = icp_map[grid->xy.num()*tau_id + csnext.swf_to_icp_id];
}

void Capturability::calcDurationMap(){
  printf(" calc duration map\n");

  vec3_t swf0, swf1;
  int nswf = (int)swf_to_xyz.size();
  duration_map.resize(nswf*nswf);
  for(int i = 0; i < nswf; i++){
    swf0 = grid->xyz[swf_to_xyz[i]];

    for(int j = 0; j < nswf; j++){
      swf1 = grid->xyz[swf_to_xyz[j]];

      swing->set(swf0, vec3_t(-swf1.x(), swf1.y(), 0.0f));
      duration_map[nswf*i + j] = grid->t.round(swing->getDuration());
    }
  }

  printf(" done: %d x %d entries\n", nswf, nswf);
}

void Capturability::calcIcpMap(){
  vec2_t cop_min(cop_x.min, cop_y.min);
  vec2_t cop_max(cop_x.max, cop_y.max);

  printf(" calc icp map\n");
  {
    icp_map.resize(grid->x.num*grid->y.num*grid->t.num);

    float  tau, tau_min, tau_max;
    float  alpha_min, alpha_max;
    float  alpha_min_inv, alpha_max_inv;
    vec2_t mu;
    vec2_t icp_min, icp_min0, icp_min1;
    vec2_t icp_max, icp_max0, icp_max1;
    for(int t_id = 0; t_id < grid->t.num; t_id++)
    for(int x_id = 0; x_id < grid->x.num; x_id++)
    for(int y_id = 0; y_id < grid->y.num; y_id++) {
      tau           = grid->t.val[t_id];
      tau_min       = std::max(tau - grid->t.stp, grid->t.min);
      tau_max       = std::min(tau + grid->t.stp, grid->t.max);
      alpha_min     = exp(tau_min/T);
      alpha_max     = exp(tau_max/T);
      alpha_min_inv = 1.0f/alpha_min;
      alpha_max_inv = 1.0f/alpha_max;
      mu            = vec2_t(grid->x.val[x_id], -1.0f*grid->y.val[y_id]);

      icp_min0 = (1.0f - alpha_min_inv)*cop_min + (alpha_min_inv)*mu;
      icp_max0 = (1.0f - alpha_min_inv)*cop_max + (alpha_min_inv)*mu;
      icp_min1 = (1.0f - alpha_max_inv)*cop_min + (alpha_max_inv)*mu;
      icp_max1 = (1.0f - alpha_max_inv)*cop_max + (alpha_max_inv)*mu;

      icp_min.x() = icp_x.min;
      icp_max.x() = icp_x.max;
      icp_min.y() = icp_y.min;
      icp_max.y() = icp_y.max;
      icp_min.x() = std::max(icp_min.x(), icp_min0.x());
      icp_min.x() = std::max(icp_min.x(), icp_min1.x());
      icp_min.y() = std::max(icp_min.y(), icp_min0.y());
      icp_min.y() = std::max(icp_min.y(), icp_min1.y());
      icp_max.x() = std::min(icp_max.x(), icp_max0.x());
      icp_max.x() = std::min(icp_max.x(), icp_max1.x());
      icp_max.y() = std::min(icp_max.y(), icp_max0.y());
      icp_max.y() = std::min(icp_max.y(), icp_max1.y());

      int idx = grid->xy.num()*t_id + grid->xy.toIndex(Index2D(x_id, y_id));
      icp_map[idx].first  = icp_min;
      icp_map[idx].second = icp_max;
    }
  }
  printf(" done: %d entries\n", (int)icp_map.size());
}

void Capturability::analyze(){
  printf(" Analysing ...... \n");
  printf(" grid size: x %d  y %d  z %d  t %d\n", grid->x.num, grid->y.num, grid->z.num, grid->t.num);

  printf(" enum valid stepping positions\n");
  {
    swf_to_xyz.clear();
    xyz_to_swf.clear();

    Index3D idx3;
    for(idx3[0] = 0; idx3[0] < grid->x.num; idx3[0]++)
    for(idx3[1] = 0; idx3[1] < grid->y.num; idx3[1]++)
    for(idx3[2] = 0; idx3[2] < grid->z.num; idx3[2]++) {
      // [x,y] in valid stepping range and z is zero
      if( isSteppable(vec2_t(grid->x.val[idx3[0]], grid->y.val[idx3[1]])) )
        swf_to_xyz.push_back(grid->xyz.toIndex(idx3));
    }

    xyz_to_swf.resize(grid->xyz.num(), -1);
    for(int swf_id = 0; swf_id < (int)swf_to_xyz.size(); swf_id++)
      xyz_to_swf[swf_to_xyz[swf_id]] = swf_id;
  }
  printf(" done: %d entries\n", (int)swf_to_xyz.size());

  calcDurationMap();
  calcIcpMap();

  // (swf_id, icp_id) -> nstep
  typedef map< pair<int,int>, int> NstepMap;
  NstepMap  nstep_map;

  printf(" calc 0 step basin\n");
  int  icp_x_id_min, icp_x_id_max;
  int  icp_y_id_min, icp_y_id_max;

  for(int swf_id = 0; swf_id < (int)swf_to_xyz.size(); swf_id++) {
    Index3D swf_idx3;
    grid->xyz.fromIndex(swf_to_xyz[swf_id], swf_idx3);

    // z should be zero
    if(swf_idx3[2] != 0)
      continue;

    grid->x.indexRange(cop_x.min, cop_x.max, icp_x_id_min, icp_x_id_max);
    grid->y.indexRange(cop_y.min, cop_y.max, icp_y_id_min, icp_y_id_max);

    for(int icp_x_id = icp_x_id_min; icp_x_id < icp_x_id_max; icp_x_id++)
    for(int icp_y_id = icp_y_id_min; icp_y_id < icp_y_id_max; icp_y_id++){
      int icp_id = grid->xy.toIndex(Index2D(icp_x_id, icp_y_id));
      cap_basin[0].push_back(CaptureState(swf_id, icp_id, 0, this));
      nstep_map[make_pair(swf_id, icp_id)] = 0;
    }
  }
  printf(" done: %d entries\n", (int)cap_basin[0].size());

  State st, stnext;
  pair<vec2_t, vec2_t> icp_range;
  std::set<int>  icp_id_valid;
  int n = 1;
  while(n < nmax){
    printf(" calc %d step basin\n", n);
    bool added = false;

    // enumerate possible current swing foot pos
    for(int swf_id = 0; swf_id < (int)swf_to_xyz.size(); swf_id++){

      icp_id_valid.clear();
      for(int basin_id = 0; basin_id < (int)cap_basin[n-1].size(); basin_id++){
        CaptureState& csnext = cap_basin[n-1][basin_id];

        calcFeasibleIcpRange(swf_id, csnext, icp_range);

        grid->x.indexRange(icp_range.first.x(), icp_range.second.x(), icp_x_id_min, icp_x_id_max);
        grid->y.indexRange(icp_range.first.y(), icp_range.second.y(), icp_y_id_min, icp_y_id_max);

        Index2D idx2;
        for(idx2[0] = icp_x_id_min; idx2[0] < icp_x_id_max; idx2[0]++)
        for(idx2[1] = icp_y_id_min; idx2[1] < icp_y_id_max; idx2[1]++){
          int icp_id = grid->xy.toIndex(idx2);
          icp_id_valid.insert(icp_id);
        }
      }

      Index3D swf_idx3;
      grid->xyz.fromIndex(swf_to_xyz[swf_id], swf_idx3);

      if(swf_idx3[2] != 0)
        continue;

      for(int icp_id : icp_id_valid){
        NstepMap::iterator it = nstep_map.find(make_pair(swf_id, icp_id));
        if(it == nstep_map.end()){
          cap_basin[n].push_back(CaptureState(swf_id, icp_id, n, this));
          nstep_map[make_pair(swf_id, icp_id)] = n;
          added = true;
        }
      }
    }

    if(!added)
      break;

    printf("  %d\n", (int)cap_basin[n].size());

    n++;
  }
  printf("Done!\n");
}

template<class T>
void saveArray(const string& filename, const vector<T>& arr){
  FILE* fp = fopen(filename.c_str(), "wb");
  fwrite(&arr[0], sizeof(T), arr.size(), fp);
  fclose(fp);
}

template<class T>
bool loadArray(const string& filename, vector<T>& arr){
  FILE* fp = fopen(filename.c_str(), "rb");
  if(!fp)
    return false;
  fseek(fp, 0, SEEK_END);
  int sz = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  int nelem = sz/(sizeof(T));
  arr.resize(nelem);
  fread(&arr[0], sizeof(T), arr.size(), fp);
  fclose(fp);
  return true;
}

void Capturability::save(const string& basename){
  stringstream ss;
  for(int n = 0; n < (int)cap_basin.size(); n++){
    ss.str("");
    ss << basename << "basin" << n << ".bin";
    saveArray(ss.str(), cap_basin[n]);
  }
  saveArray(basename + "swf_to_xyz.bin"  , swf_to_xyz  );
  saveArray(basename + "xyz_to_swf.bin"  , xyz_to_swf  );
  saveArray(basename + "duration_map.bin", duration_map);
  saveArray(basename + "icp_map.bin"     , icp_map     );
}

void Capturability::load(const string& basename) {
  stringstream ss;
  for(int n = 0; n < nmax; n++){
    ss.str("");
    ss << basename << "basin" << n << ".bin";
    loadArray(ss.str(), cap_basin[n]);

    // create index
    cap_basin[n].swf_index.resize(grid->xyz.num());
    fill(cap_basin[n].swf_index.begin(), cap_basin[n].swf_index.end(), make_pair(-1, -1));

    int swf_id   = -1;
    int idxBegin = 0;
    int i;
    for(i = 0; i < (int)cap_basin[n].size(); i++){
      CaptureState& cs = cap_basin[n][i];
      if(cs.swf_id != swf_id){
        if(swf_id != -1)
          cap_basin[n].swf_index[swf_id] = make_pair(idxBegin, i);

        swf_id   = cs.swf_id;
        idxBegin = i;
      }
      cap_basin[n].swf_index[swf_id] = make_pair(idxBegin, i);
    }
  }
  loadArray(basename + "swf_to_xyz.bin"  , swf_to_xyz  );
  loadArray(basename + "xyz_to_swf.bin"  , xyz_to_swf  );
  loadArray(basename + "duration_map.bin", duration_map);
  loadArray(basename + "icp_map.bin"     , icp_map     );
}

void Capturability::getCaptureBasin(State st, int nstep, CaptureBasin& basin){
  basin.clear();

  Index3D swf_idx3 = grid->xyz.round(st.swf);
  int swf_id = xyz_to_swf[grid->xyz.toIndex(swf_idx3)];
  if(swf_id == -1)
    return;

  pair<vec2_t, vec2_t> icp_range;

  for(int n = 0; n < nmax; n++){
    //printf("n: %d\n", n);
    if(nstep != -1 && nstep != n)
      continue;
    if((int)cap_basin.size() < n+1 || cap_basin[n].empty())
      continue;

    for(int basin_id = 0; basin_id < (int)cap_basin[n].size(); basin_id++){
      CaptureState& csnext = cap_basin[n][basin_id];
      calcFeasibleIcpRange(swf_id, csnext, icp_range);
      if( icp_range.first.x() <= st.icp.x() && st.icp.x() <= icp_range.second.x() &&
          icp_range.first.y() <= st.icp.y() && st.icp.y() <= icp_range.second.y() )
          basin.push_back(csnext);
    }
  }
}

bool Capturability::isCapturable(int swf_id, int icp_id, int& nstep) {
  for(int n = 0; n < nmax; n++){
    if(nstep != -1 && nstep != n)
      continue;
    if(cap_basin[n].swf_index.empty())
      continue;

    pair<int,int> idx = cap_basin[n].swf_index[swf_id];

    for(int i = idx.first; i < idx.second; i++){
      CaptureState& cs = cap_basin[n][i];
      if( cs.icp_id == icp_id ){
        if(nstep == -1)
          nstep = n;
        return true;
      }
    }
  }
  return false;
}

bool Capturability::findNearest(const State& st, const State& stnext, CaptureState& cs){
  float d_min = inf;
  float d_swf = 0.0f;
  float d_icp = 0.0f;
  float d     = 0.0f;
  int swf_id_prev;
  int ntested = 0;
  int ncomped = 0;

  Index3D swf_idx3 = grid->xyz.round(st.swf);
  int swf_id = xyz_to_swf[grid->xyz.toIndex(swf_idx3)];
  if(swf_id == -1)
    return false;

  for(int n = 0; n < nmax; n++){
    if(step_weight*n >= d_min)
      continue;
    if((int)cap_basin.size() < n+1 || cap_basin[n].empty())
      continue;

    swf_id_prev = -1;
    int tau_id = -1;
    for(int basin_id = 0; basin_id < (int)cap_basin[n].size(); basin_id++){
      CaptureState& csnext = cap_basin[n][basin_id];
      if(csnext.swf_id != swf_id_prev){
        d_swf  = (grid->xyz[swf_to_xyz[csnext.swf_id]] - stnext.swf).squaredNorm();
        tau_id = duration_map[swf_to_xyz.size()*swf_id + csnext.swf_id];
        swf_id_prev = csnext.swf_id;
      }
      if(step_weight*n + swf_weight*d_swf >= d_min)
        continue;

      pair<vec2_t,vec2_t>& icp_range = icp_map[grid->xy.num()*tau_id + csnext.swf_to_icp_id];

      if( icp_range.first.x() <= st.icp.x() && st.icp.x() < icp_range.second.x() &&
          icp_range.first.y() <= st.icp.y() && st.icp.y() < icp_range.second.y() ){
        d_icp = (grid->xy[csnext.icp_id] - stnext.icp).squaredNorm();
        d     = step_weight*n + swf_weight*d_swf + icp_weight*d_icp;
        if( d < d_min ){
          cs    = csnext;
          d_min = d;
        }
        ncomped++;
      }
      ntested++;
    }
  }
  printf("d_min: %f\n", d_min);

  return d_min != inf;
}

} // namespace Capt