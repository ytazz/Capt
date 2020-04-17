#include "capturability.h"

#include <limits>
#include <map>
using namespace std;

namespace Capt {

const int nmax = 10;

const float inf = numeric_limits<float>::max();
const float eps = 1.0e-5f;

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

bool Capturability::isInsideSupport(vec2_t cop){
  return (cop.x() >= cop_x.min - eps &&
          cop.x() <= cop_x.max + eps &&
          cop.y() >= cop_y.min - eps &&
          cop.y() <= cop_y.max + eps );
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

pair<vec2_t, vec2_t> Capturability::calcFeasibleIcpRange(vec3_t swf, int next_swf_id, int next_icp_id){
  //printf("calc feasible icp range:\n");
  State stnext;
  stnext.swf = grid->xyz[next_swf_id];
  stnext.icp = grid->xy [next_icp_id];

  Index3D idx3;
  idx3[0] = grid->x.round(stnext.icp.x() - stnext.swf.x());
  idx3[1] = grid->y.round(stnext.icp.y() - stnext.swf.y());

  vec3_t swfrel(swf.x() + stnext.swf.x(), swf.y() - stnext.swf.y(), swf.z());
  //printf(" get duration map\n");
  idx3[2] = duration_map[grid->xyz.toIndex(grid->xyz.round(swfrel))];

  //printf(" get icp map\n");
  return icp_map[grid->xyt.toIndex(idx3)];
}

void Capturability::analyze(){
  printf("  Analysing ...... \n");
  fflush(stdout);

  printf(" grid size: x %d  y %d  z %d  t %d\n", grid->x.num, grid->y.num, grid->z.num, grid->t.num);

  printf(" enum valid stepping positions\n");
  // [x,y,z] in valid stepping range
  swf_id_valid.clear();
  for(int x_id = 0; x_id < grid->x.num; x_id++)
  for(int y_id = 0; y_id < grid->y.num; y_id++)
  for(int z_id = 0; z_id < grid->z.num; z_id++) {
    if( isSteppable(vec2_t(grid->x.val[x_id], grid->y.val[y_id])) )
      swf_id_valid.push_back(grid->xyz.toIndex(Index3D(x_id, y_id, z_id)));
  }
  printf(" done: %d entries\n", swf_id_valid.size());

  printf(" calc duration map\n");
  {
    duration_map.resize(grid->x.num*grid->y.num*grid->z.num);
    for(int x_id = 0; x_id < grid->x.num; x_id++)
    for(int y_id = 0; y_id < grid->y.num; y_id++)
    for(int z_id = 0; z_id < grid->z.num; z_id++) {
      //printf("%d %d %d\n", x_id, y_id, z_id);
      swing->set(vec3_t(grid->x.val[x_id], grid->y.val[y_id], grid->z.val[z_id]), vec3_t(0.0, 0.0, 0.0));
      duration_map[grid->xyz.toIndex(Index3D(x_id, y_id, z_id))] = grid->t.round(swing->getDuration());
    }
  }
  printf(" done: %d entries\n", duration_map.size());

  vec2_t cop_min(cop_x.min, cop_y.min);
  vec2_t cop_max(cop_x.max, cop_y.max);

  printf(" calc icp map\n");
  {
    icp_map.resize(grid->x.num*grid->y.num*grid->t.num);
    float  tau;
    float  alpha;
    vec2_t mu;
    for(int x_id = 0; x_id < grid->x.num; x_id++)
    for(int y_id = 0; y_id < grid->y.num; y_id++)
    for(int t_id = 0; t_id < grid->t.num; t_id++) {
      tau   = grid->t.val[t_id];
      alpha = exp(tau/T);
      mu    = vec2_t(grid->x.val[x_id], -1.0f*grid->y.val[y_id]);

      int idx = grid->xyt.toIndex(Index3D(x_id, y_id, t_id));
      icp_map[idx].first  = (1.0f/alpha)*((alpha - 1.0f)*cop_min + mu);
      icp_map[idx].second = (1.0f/alpha)*((alpha - 1.0f)*cop_max + mu);
      //printf("icp min: %f,%f\n", icp_map[idx].first .x(), icp_map[idx].first .y());
      //printf("icp max: %f,%f\n", icp_map[idx].second.x(), icp_map[idx].second.y());
    }
  }
  printf(" done: %d entries\n", icp_map.size());

  // (swf_id, icp_id) -> nstep
  typedef map< pair<int,int>, int> NstepMap;
  NstepMap  nstep_map;

  printf(" calc 0 step basin\n");
  int  icp_x_id_min, icp_x_id_max;
  int  icp_y_id_min, icp_y_id_max;

  for(int swf_id : swf_id_valid) {
    Index3D swf_idx3;
    grid->xyz.fromIndex(swf_id, swf_idx3);
    // should be on the ground
    if(swf_idx3[2] != 0)
      continue;

    grid->x.indexRange(cop_x.min, cop_x.max, icp_x_id_min, icp_x_id_max);
    grid->y.indexRange(cop_y.min, cop_y.max, icp_y_id_min, icp_y_id_max);
    //cap_trans[0].push_back(CaptureTransition(swf_id, icp_x_id_min, icp_x_id_max, icp_y_id_min, icp_y_id_max, -1, -1));

    for(int icp_x_id = icp_x_id_min; icp_x_id < icp_x_id_max; icp_x_id++)
    for(int icp_y_id = icp_y_id_min; icp_y_id < icp_y_id_max; icp_y_id++){
      int icp_id = grid->xy.toIndex(Index2D(icp_x_id, icp_y_id));
      cap_basin[0].push_back(CaptureState(swf_id, icp_id, 0));
      nstep_map[make_pair(swf_id, icp_id)] = 0;
    }
  }
  printf(" done: %d entries\n", (int)cap_basin[0].size());

  State st, stnext;
  int n = 1;
  while(n <= nmax){
    printf(" calc %d step basin\n", n);
    bool found = false;

    // enumerate possible current swing foot pos
    for(int swf_id : swf_id_valid){
      //printf("swf_id: %d\n", swf_id);
      st.swf = grid->xyz[swf_id];

      // enumerate states in (N-1)-step capture basin as next state
      for(CaptureState& csnext : cap_basin[n-1]){
        pair<vec2_t, vec2_t> icp_range = calcFeasibleIcpRange(st.swf, csnext.swf_id, csnext.icp_id);

        grid->x.indexRange(icp_range.first.x(), icp_range.second.x(), icp_x_id_min, icp_x_id_max);
        grid->y.indexRange(icp_range.first.y(), icp_range.second.y(), icp_y_id_min, icp_y_id_max);

        // skip if valid icp range is empty
        if(icp_x_id_min == icp_x_id_max || icp_y_id_min == icp_y_id_max)
          continue;

        //cap_trans[n].push_back(CaptureTransition(swf_id, icp_x_id_min, icp_x_id_max, icp_y_id_min, icp_y_id_max, csnext.swf_id, csnext.icp_id));
        //printf("icp range: %f-%f  %f-%f\n", icp_range.first.x(), icp_range.second.x(), icp_range.first.y(), icp_range.second.y());
        //printf("icp index range: %d-%d  %d-%d\n", icp_x_id_min, icp_x_id_max, icp_y_id_min, icp_y_id_max);
        Index2D idx2;
        for(idx2[0] = icp_x_id_min; idx2[0] < icp_x_id_max; idx2[0]++)
        for(idx2[1] = icp_y_id_min; idx2[1] < icp_y_id_max; idx2[1]++){
          int icp_id   = grid->xy.toIndex(idx2);
          //printf("swf_id %d  icp_id %d\n", swf_id, icp_id);
          NstepMap::iterator it = nstep_map.find(make_pair(swf_id, icp_id));
          if(it == nstep_map.end()){
            //printf("found\n");
            nstep_map.insert(make_pair(make_pair(swf_id, icp_id), n));
            cap_basin[n].push_back(CaptureState(swf_id, icp_id, n));
            found = true;
          }
        }
      }
    }

    if(!found)
      break;

    printf("  %d\n", (int)cap_basin[n].size());

    n++;
  }
  printf("Done!\n");
}

void Capturability::saveBasin(string file_name, int n, bool binary){
  FILE *fp = fopen(file_name.c_str(), binary ? "wb" : "w");

  // Data
  if(binary){
    fwrite(&cap_basin[n][0], sizeof(CaptureState), cap_basin[n].size(), fp);
  }
  else{
    for(CaptureState& cs : cap_basin[n]){
      fprintf(fp, "%d,%d,%d\n", cs.swf_id, cs.icp_id, cs.nstep);
    }
    //for(CaptureState& cs : cap_basin[n]){
    //  fprintf(fp, "%d,%f,%f,%f,%f\n", cs.swf_id, cs.icp_min.x(), cs.icp_min.y(), cs.icp_max.x(), cs.icp_max.y());
    //}
  }
  fclose(fp);
}
/*
void Capturability::saveTrans(std::string file_name, int n, bool binary){
  FILE *fp = fopen(file_name.c_str(), binary ? "wb" : "w");

  // Data
  if(binary){
    fwrite(&cap_trans[n][0], sizeof(CaptureTransition), cap_trans[n].size(), fp);
  }
  else{
    for(CaptureTransition& tr : cap_trans[n]){
      fprintf(fp, "%d,%d,%d,%d,%d,%d,%d\n",
        tr.swf_id, tr.icp_x_id_min, tr.icp_x_id_max, tr.icp_y_id_min, tr.icp_y_id_max, tr.next_swf_id, tr.next_icp_id);
      //fprintf(fp, "%d,%f,%f,%f,%f,%d\n",
      //  tr.swf_id, tr.icp_min.x(), tr.icp_min.y(), tr.icp_max.x(), tr.icp_max.y(), tr.next_id);
    }
  }
  fclose(fp);
}
*/
void Capturability::loadBasin(string file_name, int n, bool binary) {
  FILE *fp = fopen(file_name.c_str(), (binary ? "rb" : "r"));
  if ( fp == NULL) {
    printf("Error: Couldn't find the file \"%s\"\n", file_name.c_str() );
    exit(EXIT_FAILURE);
  }
  printf("Found database.\n");

  if(binary){
    fseek(fp, 0, SEEK_END);
    int sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    int nelem = sz/(sizeof(CaptureState));
    cap_basin[n].resize(nelem);

    size_t nread = fread(&cap_basin[n][0], sizeof(CaptureState), cap_basin[n].size(), fp);
  }
  else{
    CaptureState cs;
    //while (fscanf(fp, "%d,%f,%f,%f,%f", &cs.swf_id, &cs.icp_min.x(), &cs.icp_min.y(), &cs.icp_max.x(), &cs.icp_max.y()) != EOF) {
    //  cap_basin[n].push_back(cs);
    //}
    while (fscanf(fp, "%d,%d,%d", &cs.swf_id, &cs.icp_id, &cs.nstep) != EOF) {
      cap_basin[n].push_back(cs);
    }
  }
  fclose(fp);

  // create index
  cap_basin[n].swf_index.resize(grid->x.num*grid->y.num*grid->z.num);
  fill(cap_basin[n].swf_index.begin(), cap_basin[n].swf_index.end(), make_pair(-1, -1));

  int swf_id   = -1;
  int idxBegin = 0;
  int i;
  for(i = 0; i < cap_basin[n].size(); i++){
    CaptureState& cs = cap_basin[n][i];
    if(cs.swf_id != swf_id){
      if(swf_id != -1)
        cap_basin[n].swf_index[swf_id] = make_pair(idxBegin, i);

      swf_id   = cs.swf_id;
      idxBegin = i;
    }
    cap_basin[n].swf_index[swf_id] = make_pair(idxBegin, i);
  }

  printf("Read success! (%d data)\n", (int)cap_basin[n].size());
}
/*
void Capturability::loadTrans(std::string file_name, int n, bool binary) {
  FILE *fp = fopen(file_name.c_str(), (binary ? "rb" : "r"));
  if ( fp == NULL) {
    printf("Error: Couldn't find the file \"%s\"\n", file_name.c_str() );
    exit(EXIT_FAILURE);
  }
  printf("Found database.\n");

  CaptureTransition tr;
  if(binary){
    fseek(fp, 0, SEEK_END);
    int sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    int nelem = sz/(sizeof(CaptureTransition));
    cap_trans[n].resize(nelem);
    size_t nread = fread(&cap_trans[n][0], sizeof(CaptureTransition), cap_trans[n].size(), fp);
  }
  else{
    //while (fscanf(fp, "%d,%f,%f,%f,%f,%d",
    //  &tr.swf_id, &tr.icp_min.x(), &tr.icp_min.y(), &tr.icp_max.x(), &tr.icp_max.y(), &tr.next_id) != EOF) {
    //  cap_trans[n].push_back(tr);
    //}
    while (fscanf(fp, "%d,%d,%d,%d,%d,%d,%d",
      &tr.swf_id, &tr.icp_x_id_min, &tr.icp_x_id_max, &tr.icp_y_id_min, &tr.icp_y_id_max, &tr.next_swf_id, &tr.next_icp_id) != EOF) {
      cap_trans[n].push_back(tr);
    }
  }
  fclose(fp);

  // create index
  cap_trans[n].swf_index.resize(grid->swf_num);
  fill(cap_trans[n].swf_index.begin(), cap_trans[n].swf_index.end(), std::make_pair(-1, -1));

  int swf_id   = -1;
  int idxBegin = 0;
  int i;
  for(i = 0; i < cap_trans[n].size(); i++){
    CaptureTransition& ct = cap_trans[n][i];
    if(ct.swf_id != swf_id){
      if(swf_id != -1)
        cap_trans[n].swf_index[swf_id] = std::make_pair(idxBegin, i);

      swf_id   = ct.swf_id;
      idxBegin = i;
    }
    cap_trans[n].swf_index[swf_id] = std::make_pair(idxBegin, i);
  }

  printf("Read success! (%d data)\n", (int)cap_trans[n].size());
}
*/
void Capturability::getCaptureBasin(State st, int nstep, CaptureBasin& basin){
  basin.clear();

  for(int n = 0; n < nmax; n++){
    if(nstep != -1 && nstep != n)
      continue;

    for(CaptureState& csnext : cap_basin[n]){
      pair<vec2_t, vec2_t> icp_range = calcFeasibleIcpRange(st.swf, csnext.swf_id, csnext.icp_id);
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

  for(int n = 0; n < nmax; n++){
    //if(nstep != -1 && nstep != n)
    //  continue;

    for(CaptureState& csnext : cap_basin[n]){
      pair<vec2_t, vec2_t> icp_range = calcFeasibleIcpRange(st.swf, csnext.swf_id, csnext.icp_id);
      if( icp_range.first.x() <= st.icp.x() && st.icp.x() <= icp_range.second.x() &&
          icp_range.first.y() <= st.icp.y() && st.icp.y() <= icp_range.second.y() ){
        float d_swf = (grid->xyz[csnext.swf_id] - stnext.swf).norm();
        float d_icp = (grid->xy [csnext.icp_id] - stnext.icp).norm();
        float d = step_weight*n + swf_weight*d_swf + icp_weight*d_icp;
        if( d < d_min ){
          cs    = csnext;
          d_min = d;
        }
      }
    }
  }
  //printf("d_swf_min: %f  d_icp_min: %f\n", d_swf_min, d_icp_min);
  printf("d_min: %f\n", d_min);

  return d_min != inf;
}

} // namespace Capt