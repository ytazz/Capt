#include "capturability.h"

namespace Capt {

const int nmax = 10;

bool CaptureRange::includes(const CaptureRange& r){
  return ( (swf_id == r.swf_id) &&
           (icp_min.x() <= r.icp_min.x() && r.icp_max.x() <= icp_max.x()) &&
           (icp_min.y() <= r.icp_min.y() && r.icp_max.y() <= icp_max.y())
         );
}
/*
bool CaptureBasin::includes(const CaptureRange& r){
  for(CaptureRange& _r : *this){
    if(_r.includes(r))
      return true;
  }
  return false;
}

void CaptureBasin::add(const CaptureRange& r){
  std::vector<CaptureRange> tmp;
  for(CaptureRange& _r : *this){
    if(!r.includes(_r))
      tmp.push_back(_r);
  }
  tmp.push_back(r);
  swap(tmp);
}
*/
Capturability::Capturability(Model* model, Param* param, Grid* grid, Swing* swing) {
  this->model    = model;
  this->param    = param;
  this->grid     = grid;
  this->swing    = swing;

  model->read(&g, "gravity");
  model->read(&h, "com_height");
  T = sqrt(h/g);

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
  cap_trans.resize(nmax);
}

Capturability::~Capturability() {
}

bool Capturability::isSteppable(vec2_t swf){
  // steppable if swf is outside the exc region
  return !(swf.x() >= exc_x.min &&
           swf.x() <= exc_x.max &&
           swf.y() >= exc_y.min &&
           swf.y() <= exc_y.max);
}

bool Capturability::isInsideSupport(vec2_t cop){
  return (cop.x() >= cop_x.min &&
          cop.x() <= cop_x.max &&
          cop.y() >= cop_y.min &&
          cop.y() <= cop_y.max);
}

Input Capturability::calcInput(State& st, State& stnext){
  Input in;
  in.swf = vec2_t(-stnext.swf.x(), stnext.swf.y());
  swing->set(st.swf, vec2Tovec3(in.swf));
  float tau   = swing->getDuration();
  float alpha = exp(tau/T);
  vec2_t diff(stnext.icp.x() - stnext.swf.x(), -(stnext.icp.y() - stnext.swf.y()));
  in.cop = (1.0f/(1.0f - alpha))*(diff - alpha*st.icp);
  return in;
}

void Capturability::analyze(){
  printf("  Analysing ...... \n");
  fflush(stdout);

  printf(" enum valid states\n");
  // enumerate all valid states
  int icp_num   = grid->getNumIcp();
  int swf_num   = grid->getNumSwf();
  int state_num = grid->getNumState();
  printf("icp_num: %d  swf_num: %d  state_num: %d\n", icp_num, swf_num, state_num);

  swf_id_valid.clear();
  for(int swf_id = 0; swf_id < swf_num; swf_id++) {
    if( isSteppable(vec3Tovec2(grid->swf[swf_id])) )
      swf_id_valid.push_back(swf_id);
  }

  std::vector<int> nstep(state_num);
  fill(nstep.begin(), nstep.end(), -1);

  printf(" calc 0 step basin\n");
  // calculate 0-step capture basin
  CaptureBasin cur;
  vec2_t cop_min(cop_x.min, cop_y.min);
  vec2_t cop_max(cop_x.max, cop_y.max);
  int    icp_x_id_min, icp_x_id_max;
  int    icp_y_id_min, icp_y_id_max;

  for(int swf_id : swf_id_valid) {
    // should be on the ground
    if(grid->swf[swf_id].z() != 0.0f)
      continue;

    grid->icp_x.indexRange(cop_x.min, cop_x.max, icp_x_id_min, icp_x_id_max);
    grid->icp_y.indexRange(cop_y.min, cop_y.max, icp_y_id_min, icp_y_id_max);
    cap_trans[0].push_back(CaptureTransition(swf_id, icp_x_id_min, icp_x_id_max, icp_y_id_min, icp_y_id_max, -1));

    for(int icp_x_id = icp_x_id_min; icp_x_id < icp_x_id_max; icp_x_id++)
    for(int icp_y_id = icp_y_id_min; icp_y_id < icp_y_id_max; icp_y_id++){
      int state_id = grid->getStateIndex(grid->getIcpIndex(icp_x_id, icp_y_id), swf_id);
      cap_basin[0].push_back(CaptureState(state_id, 0));
      nstep[state_id] = 0;
    }

    //int cur_id = (int)cap_basin[0].size();
    //cur.swf_id  = swf_id;
    //cur.icp_min = cop_min;
    //cur.icp_max = cop_max;

    //cap_basin[0].push_back(cur);
    //cap_trans[0].push_back(CaptureTransition(cur_id, -1, 0));
  }
  printf("  %d\n", (int)cap_basin[0].size());

  vec3_t swf, swf_next;
  vec2_t mu;
  vec2_t icp_min;
  vec2_t icp_max;
  float  tau;
  float  alpha;
  int n = 1;
  while(n <= nmax){
    printf(" calc %d step basin\n", n);
    bool found = false;

    // enumerate states in (N-1)-step capture basin as next state
    for(CaptureState& ctnext : cap_basin[n-1]){
      State& stnext = grid->state[ctnext.state_id];

      // enumerate possible current swing foot pos
      for(int swf_id : swf_id_valid){
        swf = grid->swf[swf_id];

        swing->set(swf, vec3_t(-stnext.swf.x(), stnext.swf.y(), 0.0f));
        tau   = swing->getDuration();
        alpha = exp(tau/T);
        mu.x()  =  (stnext.icp.x() - stnext.swf.x());
        mu.y()  = -(stnext.icp.y() - stnext.swf.y());
        icp_min = (1.0f/alpha)*((alpha - 1.0f)*cop_min + mu);
        icp_max = (1.0f/alpha)*((alpha - 1.0f)*cop_max + mu);
        grid->icp_x.indexRange(icp_min.x(), icp_max.x(), icp_x_id_min, icp_x_id_max);
        grid->icp_y.indexRange(icp_min.y(), icp_max.y(), icp_y_id_min, icp_y_id_max);

        cap_trans[n].push_back(CaptureTransition(swf_id, icp_x_id_min, icp_x_id_max, icp_y_id_min, icp_y_id_max, ctnext.state_id));

        for(int icp_x_id = icp_x_id_min; icp_x_id < icp_x_id_max; icp_x_id++)
        for(int icp_y_id = icp_y_id_min; icp_y_id < icp_y_id_max; icp_y_id++){
          int state_id = grid->getStateIndex(grid->getIcpIndex(icp_x_id, icp_y_id), swf_id);

          if(nstep[state_id] == -1){
            nstep[state_id] = n;
            cap_basin[n].push_back(CaptureState(state_id, n));
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

  /*
  while(n <= nmax){
    printf(" calc %d step basin\n", n);
    bool found = false;

    // enumerate states in (N-1)-step capture basin as next state
    for(CaptureRange& next : cap_basin[n-1]){
      swf_next = grid->swf[next.swf_id];

      // enumerate possible current swing foot pos
      for(int swf_id : swf_id_valid){
        //printf("swf_id: %d\n", swf_id);
        //State& st = grid->state[state_id];
        swf = grid->swf[swf_id];

        // calculate step duration
        swing->set(swf, swf_next);
        tau   = swing->getDuration();
        alpha = exp(tau/T);

        // calculate feasible icp range
        //mu.x() =  (stnext.icp.x() - stnext.swf.x());
        //mu.y() = -(stnext.icp.y() - stnext.swf.y());
        icp_min = (1.0f/alpha)*((alpha - 1.0f)*cop_min + vec2_t(next.icp_min.x() - swf_next.x(), -next.icp_max.y() + swf_next.y());
        icp_max = (1.0f/alpha)*((alpha - 1.0f)*cop_max + vec2_t(next.icp_max.x() - swf_next.x(), -next.icp_min.y() + swf_next.y());
        icp_min.x() = std::min(std::max(icp_x.min, icp_min.x()), icp_x.max);
        icp_min.y() = std::min(std::max(icp_y.min, icp_min.y()), icp_y.max);
        icp_max.x() = std::min(std::max(icp_x.min, icp_max.x()), icp_x.max);
        icp_max.y() = std::min(std::max(icp_y.min, icp_max.y()), icp_y.max);

        if( icp_min.x() < icp_max.x() && icp_min.y() < icp_max.y() ){
          cur.swf_id  = swf_id;
          cur_icp_min = icp_min;
          cur_icp_max = icp_max;
          cap_basin[n].push_back(cur);
          cap_range[n].push_back(cur, next, n);
        }

        // get grid index range
        //grid->icp_x.indexRange(icp_min.x(), icp_max.x(), icp_x_id_min, icp_x_id_max);
        //grid->icp_y.indexRange(icp_min.y(), icp_max.y(), icp_y_id_min, icp_y_id_max);
        //printf("icp_x_id: %d %d\n", icp_x_id_min, icp_x_id_max);
        //printf("icp_y_id: %d %d\n", icp_y_id_min, icp_y_id_max);

        //for(int icp_x_id = icp_x_id_min; icp_x_id < icp_x_id_max; icp_x_id++)
        //for(int icp_y_id = icp_y_id_min; icp_y_id < icp_y_id_max; icp_y_id++){
        //  icp_id   = grid->getIcpIndex  (icp_x_id, icp_y_id);
        //  state_id = grid->getStateIndex(icp_id, swf_id);
        //  //printf("icp_id: %d\n", icp_id);
        //  //printf("state_id: %d\n", state_id);

        //  if(nstep[state_id] == -1){
        //    nstep[state_id] = n;
        //    found = true;
        //    cap_basin[n].push_back(state_id);
        //  }
        //}
        //if( (icp_x_id_min < icp_x_id_max) &&
        //    (icp_y_id_min < icp_y_id_max) )
        //  cap_range[n].push_back(CaptureRange(icp_x_id_min, icp_x_id_max, icp_y_id_min, icp_y_id_max, swf_id, next_id, n));
      }
    }

    if(!found)
      break;

    printf("  %d\n", (int)cap_basin[n].size());

    n++;
  }
  */
  printf("Done!\n");
}

void Capturability::saveBasin(std::string file_name, int n, bool binary){
  FILE *fp = fopen(file_name.c_str(), binary ? "wb" : "w");

  // Data
  if(binary){
    fwrite(&cap_basin[n][0], sizeof(CaptureState), cap_basin[n].size(), fp);
  }
  else{
    for(CaptureState& st : cap_basin[n]){
      fprintf(fp, "%d,%d\n", st.state_id, st.nstep);
    }
  }
  fclose(fp);
}

void Capturability::saveTrans(std::string file_name, int n, bool binary){
  FILE *fp = fopen(file_name.c_str(), binary ? "wb" : "w");

  // Data
  if(binary){
    fwrite(&cap_trans[n][0], sizeof(CaptureTransition), cap_trans[n].size(), fp);
  }
  else{
    for(CaptureTransition& tr : cap_trans[n]){
      fprintf(fp, "%d,%d,%d,%d,%d,%d\n", tr.swf_id, tr.icp_x_id_min, tr.icp_x_id_max, tr.icp_y_id_min, tr.icp_y_id_max, tr.next_id);
    }
  }
  fclose(fp);
}

void Capturability::loadBasin(std::string file_name, int n, bool binary) {
  FILE *fp = fopen(file_name.c_str(), (binary ? "rb" : "r"));
  if ( fp == NULL) {
    printf("Error: Couldn't find the file \"%s\"\n", file_name.c_str() );
    exit(EXIT_FAILURE);
  }
  printf("Found database.\n");

  int state_id;
  if(binary){
    fseek(fp, 0, SEEK_END);
    int sz = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    int nelem = sz/(sizeof(CaptureState));
    cap_basin[n].resize(nelem);

    size_t nread = fread(&cap_basin[n][0], sizeof(CaptureState), cap_basin[n].size(), fp);
  }
  else{
    CaptureState st;
    while (fscanf(fp, "%d,%d", &st.state_id, &st.nstep) != EOF) {
      cap_basin[n].push_back(st);
    }
  }
  fclose(fp);

  printf("Read success! (%d data)\n", (int)cap_basin[n].size());
}

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
    while (fscanf(fp, "%d,%d,%d,%d,%d,%d", &tr.swf_id, &tr.icp_x_id_min, &tr.icp_x_id_max, &tr.icp_y_id_min, &tr.icp_y_id_max, &tr.next_id) != EOF) {
      cap_trans[n].push_back(tr);
    }
  }
  fclose(fp);

  printf("Read success! (%d data)\n", (int)cap_trans[n].size());
}

void Capturability::getCaptureBasin(State st, int nstep, CaptureBasin& basin){
  basin.clear();
  int icp_x_id = grid->icp_x.round(st.icp.x());
  int icp_y_id = grid->icp_y.round(st.icp.y());
  int swf_x_id = grid->swf_x.round(st.swf.x());
  int swf_y_id = grid->swf_y.round(st.swf.y());
  int swf_z_id = grid->swf_z.round(st.swf.z());
  int swf_id = grid->getSwfIndex(swf_x_id, swf_y_id, swf_z_id);

  for(int n = 0; n < nmax; n++){
    if(nstep != -1 && nstep != n)
      continue;

    for(CaptureTransition& tr : cap_trans[n]){
      if( (tr.swf_id == swf_id) &&
          (tr.icp_x_id_min <= icp_x_id && icp_x_id < tr.icp_x_id_max) &&
          (tr.icp_y_id_min <= icp_y_id && icp_y_id < tr.icp_y_id_max) )
        basin.push_back(CaptureState(tr.next_id, n-1));
    }
  }
}

/*
void Capturability::getCaptureRange(vec2_t icp, vec3_t swf, int nstep, std::vector<CaptureRange>& region){
  region.clear();
  int swf_id = grid->getSwfIndex(grid->swf_x.round(swf.x()), grid->swf_y.round(swf.y()), grid->swf_z.round(swf.z()));

  for(int n = 0; n < nmax; n++){
    for(CaptureRange& r : cap_range[n]){
      if( r.cur.swf_id == swf_id &&
          (r.cur.icp_min.x() <= icp.x() && icp.x() <= r.cur.icp_max.x()) &&
          (r.cur.icp_min.y() <= icp.y() && icp.y() <= r.cur.icp_max.y()) &&
          (nstep == -1 || r.nstep <= nstep) )
        region.push_back(r);
    }
  }
}
*/
/*
void Capturability::getCaptureRegion(int state_id, int nstep, std::vector<CaptureRegion>& region) {
  region.clear();

  for(int n = 0; n < nmax; n++){
    for(CaptureRegion& r : cap_region[n]){
      if( r.state_id == state_id &&
          (nstep == -1 || r.nstep <= nstep) )
        region.push_back(r);
    }
  }
}
*/
bool Capturability::isCapturable(int state_id, int nstep) {
  for(int n = 0; n < nmax; n++){
    if(nstep != -1 && nstep != n)
      continue;

    for(CaptureState& ct : cap_basin[n]){
      if(ct.state_id == state_id)
        return true;
    }
  }
  return false;
}

} // namespace Capt