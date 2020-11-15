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

CaptureState::CaptureState(int _swg_id, int _icp_id, int _nstep, Capturability* cap){
  swg_id = _swg_id;
  icp_id = _icp_id;
  nstep  = _nstep;

  vec3_t swg = cap->grid->xyz[cap->swg_to_xyz[swg_id]];
  vec2_t icp = cap->grid->xy [icp_id];
  swg_to_icp_id = cap->grid->xy.toIndex(cap->grid->xy.round(vec2_t(icp.x() - swg.x(), icp.y() - swg.y())));
}

Capturability::Capturability(Model* model, Param* param) {
  this->model    = model;
  this->param    = param;

  grid  = new Grid (param);
  swing = new Swing(model);

  cap_basin.resize(nmax);

  step_weight = 1.0f;
  swg_weight  = 1.0f;
  icp_weight  = 1.0f;
}

Capturability::~Capturability() {
}

void Capturability::Read(XMLNode* node){
  node->Get(g, ".gravity"   );
  node->Get(h, ".com_height");
  T = sqrt(h/g);

  // exc
  node->Get(swg_x.min, ".swg_x_min");
  node->Get(swg_x.max, ".swg_x_max");
  node->Get(swg_y.min, ".swg_y_min");
  node->Get(swg_y.max, ".swg_y_max");
  // exc
  node->Get(exc_x.min, ".exc_x_min");
  node->Get(exc_x.max, ".exc_x_max");
  node->Get(exc_y.min, ".exc_y_min");
  node->Get(exc_y.max, ".exc_y_max");
  // cop
  node->Get(cop_x.min, ".cop_x_min");
  node->Get(cop_x.max, ".cop_x_max");
  node->Get(cop_y.min, ".cop_y_min");
  node->Get(cop_y.max, ".cop_y_max");
  // icp
  node->Get(icp_x.min, ".icp_x_min");
  node->Get(icp_x.max, ".icp_x_max");
  node->Get(icp_y.min, ".icp_y_min");
  node->Get(icp_y.max, ".icp_y_max");

}

bool Capturability::isSteppable(vec2_t p_swg, real_t r_swg){
  // steppable if swg is inside swing region and outside the exc region
  return  ( (swg_x.min - eps <= p_swg.x() && p_swg.x() <= swg_x.max + eps) &&
            (swg_y.min - eps <= p_swg.y() && p_swg.y() <= swg_y.max + eps) &&
	        (swg_r.min - eps <= r_swg     && r_swg     <= swg_r.max + eps))
             &&
         !( (exc_x.min + eps <= swg.x() && swg.x() <= exc_x.max - eps) &&
            (exc_y.min + eps <= swg.y() && swg.y() <= exc_y.max - eps) );
}

bool Capturability::isInsideSupport(vec2_t cop, float margin){
  return (cop.x() >= cop_x.min - margin &&
          cop.x() <= cop_x.max + margin &&
          cop.y() >= cop_y.min - margin &&
          cop.y() <= cop_y.max + margin );
}

Input Capturability::calcInput(const State& st, const State& stnext){
  Input in;
  in.swg = vec2_t(-stnext.swg.x(), stnext.swg.y());
  swing->set(st.swg, vec2Tovec3(in.swg));
  float tau   = swing->getDuration();
  float alpha = exp(tau/T);
  vec2_t diff(stnext.icp.x() - stnext.swg.x(), -(stnext.icp.y() - stnext.swg.y()));
  in.cop = (1.0f/(1.0f - alpha))*(diff - alpha*st.icp);
  return in;
}

void Capturability::calcFeasibleIcpRange(int swg_id, const CaptureState& csnext, pair<vec2_t, vec2_t>& icp_range){
  int tau_id = duration_map[swg_to_xyzr.size()*swg_id + csnext.swg_id];
  icp_range = icp_map[grid->xy.num()*tau_id + csnext.swg_to_icp_id];
}

void Capturability::calcDurationMap(){
  printf(" calc duration map\n");

  vec4_t swg0, swg1;
  int nswg = (int)swg_to_xyzr.size();
  duration_map.resize(nswg*nswg);
  for(int i = 0; i < nswg; i++){
    swg0 = grid->xyzr[swg_to_xyzr[i]];

    for(int j = 0; j < nswg; j++){
      swg1 = grid->xyzr[swg_to_xyzr[j]];

      swing->set(swg0, vec3_t(-swg1.x(), swg1.y(), 0.0f));
      duration_map[nswg*i + j] = grid->t.round(swing->getDuration());
    }
  }

  printf(" done: %d x %d entries\n", nswg, nswg);
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
		swg_to_xyzr.clear();
		xyzr_to_swg.clear();

		Index4D idx4;
		for(idx4[0] = 0; idx4[0] < grid->x.num; idx4[0]++)
		for(idx4[1] = 0; idx4[1] < grid->y.num; idx4[1]++)
		for(idx4[2] = 0; idx4[2] < grid->z.num; idx4[2]++) 
		for(idx4[3] = 0; idx4[3] < grid->r.num; idx4[3]++) {
			// [x,y] in valid stepping range and z is zero
			if( isSteppable( vec2_t(grid->x.val[idx4[0]], grid->y.val[idx4[1]]), grid->r.val[idx4[3]]) )
				swg_to_xyzr.push_back(grid->xyzr.toIndex(idx4));
		}

		xyzr_to_swg.resize(grid->xyzr.num(), -1);
	    for(int swg_id = 0; swg_id < (int)swg_to_xyzr.size(); swg_id++)
			xyzr_to_swg[swg_to_xyzr[swg_id]] = swg_id;
	}
	printf(" done: %d entries\n", (int)swg_to_xyzr.size());

	calcDurationMap();
	calcIcpMap();

	// (swg_id, icp_id) -> nstep
	typedef map< pair<int,int>, int> NstepMap;
	NstepMap  nstep_map;

	printf(" calc 0 step basin\n");
	int  icp_x_id_min, icp_x_id_max;
	int  icp_y_id_min, icp_y_id_max;

	for(int swg_id = 0; swg_id < (int)swg_to_xyzr.size(); swg_id++) {
		Index$D swg_idx4;
		grid->xyzr.fromIndex(swg_to_xyzr[swg_id], swg_idx4);

		// z should be zero
		if(swg_idx4[2] != 0)
			continue;

		grid->x.indexRange(cop_x.min, cop_x.max, icp_x_id_min, icp_x_id_max);
		grid->y.indexRange(cop_y.min, cop_y.max, icp_y_id_min, icp_y_id_max);

		for(int icp_x_id = icp_x_id_min; icp_x_id < icp_x_id_max; icp_x_id++)
		for(int icp_y_id = icp_y_id_min; icp_y_id < icp_y_id_max; icp_y_id++){
			int icp_id = grid->xy.toIndex(Index2D(icp_x_id, icp_y_id));
			cap_basin[0].push_back(CaptureState(swg_id, icp_id, 0, this));
			nstep_map[make_pair(swg_id, icp_id)] = 0;
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
		for(int swg_id = 0; swg_id < (int)swg_to_xyzr.size(); swg_id++){

			icp_id_valid.clear();
			for(int basin_id = 0; basin_id < (int)cap_basin[n-1].size(); basin_id++){
				CaptureState& csnext = cap_basin[n-1][basin_id];

				calcFeasibleIcpRange(swg_id, csnext, icp_range);

				grid->x.indexRange(icp_range.first.x(), icp_range.second.x(), icp_x_id_min, icp_x_id_max);
				grid->y.indexRange(icp_range.first.y(), icp_range.second.y(), icp_y_id_min, icp_y_id_max);

				Index2D idx2;
				for(idx2[0] = icp_x_id_min; idx2[0] < icp_x_id_max; idx2[0]++)
				for(idx2[1] = icp_y_id_min; idx2[1] < icp_y_id_max; idx2[1]++){
					int icp_id = grid->xy.toIndex(idx2);
					icp_id_valid.insert(icp_id);
				}
			}

			Index3D swg_idx3;
			grid->xyz.fromIndex(swg_to_xyz[swg_id], swg_idx3);

			if(swg_idx3[2] != 0)
				continue;

			for(int icp_id : icp_id_valid){
				NstepMap::iterator it = nstep_map.find(make_pair(swg_id, icp_id));
				if(it == nstep_map.end()){
					cap_basin[n].push_back(CaptureState(swg_id, icp_id, n, this));
					nstep_map[make_pair(swg_id, icp_id)] = n;
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
  saveArray(basename + "swg_to_xyz.bin"  , swg_to_xyz  );
  saveArray(basename + "xyz_to_swg.bin"  , xyz_to_swg  );
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
    cap_basin[n].swg_index.resize(grid->xyz.num());
    fill(cap_basin[n].swg_index.begin(), cap_basin[n].swg_index.end(), make_pair(-1, -1));

    int swg_id   = -1;
    int idxBegin = 0;
    int i;
    for(i = 0; i < (int)cap_basin[n].size(); i++){
      CaptureState& cs = cap_basin[n][i];
      if(cs.swg_id != swg_id){
        if(swg_id != -1)
          cap_basin[n].swg_index[swg_id] = make_pair(idxBegin, i);

        swg_id   = cs.swg_id;
        idxBegin = i;
      }
      cap_basin[n].swg_index[swg_id] = make_pair(idxBegin, i);
    }
  }
  loadArray(basename + "swg_to_xyz.bin"  , swg_to_xyz  );
  loadArray(basename + "xyz_to_swg.bin"  , xyz_to_swg  );
  loadArray(basename + "duration_map.bin", duration_map);
  loadArray(basename + "icp_map.bin"     , icp_map     );
}

void Capturability::getCaptureBasin(State st, int nstep, CaptureBasin& basin){
  basin.clear();

  Index3D swg_idx3 = grid->xyz.round(st.swg);
  int swg_id = xyz_to_swg[grid->xyz.toIndex(swg_idx3)];
  if(swg_id == -1)
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
      calcFeasibleIcpRange(swg_id, csnext, icp_range);
      if( icp_range.first.x() <= st.icp.x() && st.icp.x() <= icp_range.second.x() &&
          icp_range.first.y() <= st.icp.y() && st.icp.y() <= icp_range.second.y() )
          basin.push_back(csnext);
    }
  }
}

bool Capturability::isCapturable(int swg_id, int icp_id, int& nstep) {
  for(int n = 0; n < nmax; n++){
    if(nstep != -1 && nstep != n)
      continue;
    if(cap_basin[n].swg_index.empty())
      continue;

    pair<int,int> idx = cap_basin[n].swg_index[swg_id];

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
  float d_swg = 0.0f;
  float d_icp = 0.0f;
  float d     = 0.0f;
  int swg_id_prev;
  int ntested = 0;
  int ncomped = 0;

  Index3D swg_idx3 = grid->xyz.round(st.swg);
  int swg_id = xyz_to_swg[grid->xyz.toIndex(swg_idx3)];
  if(swg_id == -1)
    return false;

  for(int n = 0; n < nmax; n++){
    if(step_weight*n >= d_min)
      continue;
    if((int)cap_basin.size() < n+1 || cap_basin[n].empty())
      continue;

    swg_id_prev = -1;
    int tau_id = -1;
    for(int basin_id = 0; basin_id < (int)cap_basin[n].size(); basin_id++){
      CaptureState& csnext = cap_basin[n][basin_id];
      if(csnext.swg_id != swg_id_prev){
        d_swg  = (grid->xyz[swg_to_xyz[csnext.swg_id]] - stnext.swg).squaredNorm();
        tau_id = duration_map[swg_to_xyz.size()*swg_id + csnext.swg_id];
        swg_id_prev = csnext.swg_id;
      }
      if(step_weight*n + swg_weight*d_swg >= d_min)
        continue;

      pair<vec2_t,vec2_t>& icp_range = icp_map[grid->xy.num()*tau_id + csnext.swg_to_icp_id];

      if( icp_range.first.x() <= st.icp.x() && st.icp.x() < icp_range.second.x() &&
          icp_range.first.y() <= st.icp.y() && st.icp.y() < icp_range.second.y() ){
        d_icp = (grid->xy[csnext.icp_id] - stnext.icp).squaredNorm();
        d     = step_weight*n + swg_weight*d_swg + icp_weight*d_icp;
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