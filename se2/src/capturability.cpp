#define _CRT_SECURE_NO_WARNINGS

#include "capturability.h"

#include <limits>
#include <map>
#include <set>
using namespace std;

#include <sbtimer.h>
static Scenebuilder::Timer timer;

namespace Capt {

const int nmax = 10;

const real_t inf = numeric_limits<real_t>::max();
const real_t eps = 1.0e-5f;

CaptureState::CaptureState(int _swg_id, int _icp_id, int _nstep, Capturability* cap){
	swg_id = _swg_id;
	icp_id = _icp_id;
	nstep  = _nstep;
}

Capturability::Capturability() {
	grid  = new Grid ();
	swing = new Swing();

	cap_basin.resize(nmax);

	g = 9.8;
	h = 1.0;

	step_weight = 1.0;
	swg_weight  = 1.0;
	icp_weight  = 1.0;
	tau_weight  = 1.0;
}

Capturability::~Capturability() {
}

void Capturability::Read(Scenebuilder::XMLNode* node){
	node->Get(g, ".gravity"   );
	node->Get(h, ".com_height");
	T = sqrt(h/g);

	node->Get(swg_near , ".swg_near" );
	node->Get(swg_far  , ".swg_far"  );
	node->Get(swg_angle, ".swg_angle");
	
	// cop
	cop_x.Read(node->GetNode("cop_x"));
	cop_y.Read(node->GetNode("cop_y"));
	// icp
	icp_x.Read(node->GetNode("icp_x"));
	icp_y.Read(node->GetNode("icp_y"));

	// grid
	grid ->Read(node->GetNode("grid"));

	// swing
	swing->Read(node->GetNode("swing"));

	node->Get(step_weight, ".step_weight");
    node->Get(swg_weight , ".swg_weight" );
    node->Get(icp_weight , ".icp_weight" );
    node->Get(tau_weight , ".tau_weight" );

}

bool Capturability::IsSteppable(const vec2_t& p_swg, real_t r_swg){
	real_t dist  = p_swg.norm();
	real_t angle = atan2(p_swg[1], p_swg[0]);
	return ( swg_near           - eps <= dist          && dist          < swg_far          + eps &&
		     pi/2.0 - swg_angle - eps <= angle - r_swg && angle - r_swg < pi/2 + swg_angle + eps );
}

bool Capturability::IsInsideSupport(const vec2_t& cop, real_t margin){
  return (cop.x >= cop_x.min - margin &&
          cop.x <= cop_x.max + margin &&
          cop.y >= cop_y.min - margin &&
          cop.y <= cop_y.max + margin );
}

void Capturability::CalcInput(const State& st, const State& stnext, Input& in){
	mat2_t R       = mat2_t::Rot(stnext.swg[2]);
	vec2_t p_land  = -(R*vec2_t(stnext.swg.x, -stnext.swg.y));
	real_t r_land  =  stnext.swg[2];
	in.land        =  vec3_t(p_land.x, p_land.y, r_land);

	real_t alpha = exp(in.tau/T);
	vec2_t mu;
	CalcMu(stnext.swg, stnext.icp, mu);
	in.cop = (alpha*st.icp - mu)/(alpha - 1.0);
}

bool UpdateDurationRange(real_t mu, real_t cop, real_t icp, bool min_or_max, vec2_t& ainv_range){
	if(mu == cop){
		if( ( min_or_max && icp < mu) ||
			(!min_or_max && icp > mu) )
			return false;
	}
	else{	
		real_t tmp = (icp - cop)/(mu - cop);
		if(min_or_max){
			if(mu < cop)
				 ainv_range[0] = std::max(ainv_range[0], tmp);
			else ainv_range[1] = std::min(ainv_range[1], tmp);
		}
		else{
			if(mu < cop)
				 ainv_range[1] = std::min(ainv_range[1], tmp);
			else ainv_range[0] = std::max(ainv_range[0], tmp);
		}
	}
	return true;
}

void Capturability::CalcMu(const vec3_t& swg, const vec2_t& icp, vec2_t& mu){
	mu = mat2_t::Rot(swg[2])*vec2_t(icp.x - swg.x, -1.0*(icp.y - swg.y));
}

void Capturability::CalcTauRange(const vec2_t& ainv_range, vec2_t& tau_range){
	tau_range[0] = -T*log(ainv_range[1]);
	tau_range[1] = -T*log(ainv_range[0]);
}

void Capturability::CalcAinvRange(const vec2_t& tau_range, vec2_t& ainv_range){
	ainv_range[0] = exp(-tau_range[1]/T);
	ainv_range[1] = exp(-tau_range[0]/T);
}

bool Capturability::CalcFeasibleAinvRange(const vec2_t& mu, const vec2_t& icp, vec2_t& ainv_range){
	if(!UpdateDurationRange(mu.x, cop_x.min, icp.x, true , ainv_range)) return false;
	if(!UpdateDurationRange(mu.x, cop_x.max, icp.x, false, ainv_range)) return false;
	if(!UpdateDurationRange(mu.y, cop_y.min, icp.y, true , ainv_range)) return false;
	if(!UpdateDurationRange(mu.y, cop_y.max, icp.y, false, ainv_range)) return false;

	return ainv_range[0] < ainv_range[1];
}
/*
void Capturability::CalcFeasibleIcpRange(const vec2_t& mu, real_t ainv, pair<vec2_t, vec2_t>& icp_range){
	icp_range.first = vec2_t(
		(1.0 - ainv)*cop_x.min + (ainv)*mu.x,
		(1.0 - ainv)*cop_y.min + (ainv)*mu.y
		);
	icp_range.second = vec2_t(
		(1.0 - ainv)*cop_x.max + (ainv)*mu.x,
		(1.0 - ainv)*cop_y.max + (ainv)*mu.y
		);
}
*/
void Capturability::CalcFeasibleIcpRange(const vec2_t& mu, const vec2_t& ainv_range, pair<vec2_t, vec2_t>& icp_range){
	icp_range.first = vec2_t(
		std::min(
			(1.0 - ainv_range[0])*cop_x.min + (ainv_range[0])*mu.x,
			(1.0 - ainv_range[1])*cop_x.min + (ainv_range[1])*mu.x ),
		std::min(
			(1.0 - ainv_range[0])*cop_y.min + (ainv_range[0])*mu.y,
			(1.0 - ainv_range[1])*cop_y.min + (ainv_range[1])*mu.y )
		);
	icp_range.second = vec2_t(
		std::max(
			(1.0 - ainv_range[0])*cop_x.max + (ainv_range[0])*mu.x,
			(1.0 - ainv_range[1])*cop_x.max + (ainv_range[1])*mu.x ),
		std::max(
			(1.0 - ainv_range[0])*cop_y.max + (ainv_range[0])*mu.y,
			(1.0 - ainv_range[1])*cop_y.max + (ainv_range[1])*mu.y )
		);
}

void Capturability::CalcFeasibleMuRange(const vec2_t& icp, const vec2_t& ainv_range, std::pair<vec2_t, vec2_t>& mu_range ){
	vec2_t a_range;
	a_range[0] = 1.0/ainv_range[1];
	a_range[1] = 1.0/ainv_range[0];

	mu_range.first = vec2_t(
		std::min(
			(1.0 - a_range[0])*cop_x.max + (a_range[0])*icp.x,
			(1.0 - a_range[1])*cop_x.max + (a_range[1])*icp.x ),
		std::min(
			(1.0 - a_range[0])*cop_y.max + (a_range[0])*icp.y,
			(1.0 - a_range[1])*cop_y.max + (a_range[1])*icp.y )
		);
	mu_range.second = vec2_t(
		std::max(
			(1.0 - a_range[0])*cop_x.min + (a_range[0])*icp.x,
			(1.0 - a_range[1])*cop_x.min + (a_range[1])*icp.x ),
		std::max(
			(1.0 - a_range[0])*cop_y.min + (a_range[0])*icp.y,
			(1.0 - a_range[1])*cop_y.min + (a_range[1])*icp.y )
		);
}	

real_t Capturability::CalcMinDuration(const vec3_t& swg0, const vec3_t& swg1){
	mat2_t R      = mat2_t::Rot(swg1[2]);
	vec2_t p_land = -(R*vec2_t(swg1.x, -swg1.y));

	swing->SetSwg (vec3_t(swg0.x, swg0.y, 0.0), swg0[2]);
	swing->SetLand(vec3_t(p_land.x, p_land.y, 0.0), swg1[2]);

	return swing->GetMinDuration();
}

void Capturability::EnumReachable(const vector< pair<int, real_t> >& seed, vector<bool>& swg_id_array){
	struct CompByTau{
		const vector<real_t>& _tau_map;

		bool operator()(int lhs, int rhs) const { return _tau_map[lhs] > _tau_map[rhs]; }

		CompByTau(const vector<real_t>& _map):_tau_map(_map){}
	};

	// xyr_id to tau
	static vector<real_t>  tau_map;
	if(tau_map.empty())
		tau_map.resize(grid->xyr.Num());

	fill(tau_map.begin(), tau_map.end(), 0.0);
	fill(swg_id_array.begin(), swg_id_array.end(), false);

	CompByTau            comp(tau_map);
	set<int, CompByTau>  queue(comp);
	vector<Index3D>      indices3;

	vec3_t swg1;
	vec2_t dp;
	real_t dr;
	real_t tau;

	for(pair<int, real_t> s : seed){
		int    swg_id = s.first;
		real_t tau0   = s.second - swing->dsp_duration;

		swg1 = grid->xyr[swg_to_xyr[swg_id]];

		// landing pose equivalent to swg1
		mat2_t R      = mat2_t::Rot(swg1[2]);
		vec2_t p_land = -(R*vec2_t(swg1.x, -swg1.y));
		real_t r_land = swg1[2];

		// enum neighbor grid points
		grid->xyr.Neighbors(vec3_t(p_land.x, p_land.y, r_land), indices3);
		for(Index3D& idx3 : indices3){
			int i = grid->xyr.ToIndex(idx3);
			dp.x = grid->x.val[idx3[0]] - p_land.x;
			dp.y = grid->y.val[idx3[1]] - p_land.y;
			dr   = grid->r.val[idx3[2]] - r_land;

			int swg_id = xyr_to_swg[i];
			if(swg_id == -1)
				continue;
			
			real_t tau = tau0 - std::max(1.5*(dp.norm()/swing->v_max), 1.5*(std::abs(dr)/swing->w_max));
			if(tau > tau_map[i]){
				swg_id_array[swg_id] = true;
				tau_map[i] = tau;
				queue.insert(i);
			}
		}
	}

	while(!queue.empty()){
		int i = *(queue.begin());
		queue.erase(queue.begin());

		Index3D idx3;
		grid->xyr.FromIndex(i, idx3);

		Index3D idx3_adj;
		for(int j0 = -1; j0 <= 1; j0++)
		for(int j1 = -1; j1 <= 1; j1++)
		for(int j2 = -1; j2 <= 1; j2++){
			if(j0 == 0 && j1 == 0 && j2 == 0)
				continue;
			
			idx3_adj[0] = idx3[0] + j0;
			idx3_adj[1] = idx3[1] + j1;
			idx3_adj[2] = idx3[2] + j2;
			if( !(0 <= idx3_adj[0] && idx3_adj[0] < grid->x.num) ||
				!(0 <= idx3_adj[1] && idx3_adj[1] < grid->y.num) ||
				!(0 <= idx3_adj[2] && idx3_adj[2] < grid->r.num) )
				continue;

			int i_adj = grid->xyr.ToIndex(idx3_adj);
			int swg_id = xyr_to_swg[i_adj];
			if(swg_id == -1)
				continue;

			dp.x = grid->x.stp*j0;
			dp.y = grid->y.stp*j1;
			dr   = grid->r.stp*j2;
			tau  = tau_map[i] - std::max(1.5*(dp.norm()/swing->v_max), 1.5*(std::abs(dr)/swing->w_max));
			if(tau > tau_map[i_adj]){
				swg_id_array[swg_id] = true;
				tau_map[i_adj] = tau;
				queue.insert(i_adj);
			}
		}

	}
}

void Capturability::Analyze(){
	printf(" Analysing ...... \n");
	printf(" grid size: x %d  y %d  r %d  ainv %d\n", grid->x.num, grid->y.num, grid->r.num, grid->ainv.num);

	printf(" enum valid stepping positions\n");
	{
		swg_to_xyr.clear();
		xyr_to_swg.clear();

		Index3D idx3;
		for(idx3[0] = 0; idx3[0] < grid->x.num; idx3[0]++)
		for(idx3[1] = 0; idx3[1] < grid->y.num; idx3[1]++)
		for(idx3[2] = 0; idx3[2] < grid->r.num; idx3[2]++) {
			if( IsSteppable( vec2_t(grid->x.val[idx3[0]], grid->y.val[idx3[1]]), grid->r.val[idx3[2]]) ){
				swg_to_xyr.push_back(grid->xyr.ToIndex(idx3));
			}
		}

		xyr_to_swg.resize(grid->xyr.Num(), -1);
	    for(int swg_id = 0; swg_id < (int)swg_to_xyr.size(); swg_id++)
			xyr_to_swg[swg_to_xyr[swg_id]] = swg_id;
	}
	printf(" done: %d entries\n", (int)swg_to_xyr.size());
	
	// (swg_id, icp_id) -> nstep
	typedef map< pair<int,int>, int> NstepMap;
	NstepMap  nstep_map;

	printf(" calc 0 step basin\n");
	int  icp_x_id_min, icp_x_id_max;
	int  icp_y_id_min, icp_y_id_max;

	for(int swg_id = 0; swg_id < (int)swg_to_xyr.size(); swg_id++) {
		Index3D swg_idx3;
		grid->xyr.FromIndex(swg_to_xyr[swg_id], swg_idx3);
		vec3_t swg = grid->xyr[swg_idx3];
		
		grid->x.IndexRange(cop_x.min, cop_x.max, icp_x_id_min, icp_x_id_max);
		grid->y.IndexRange(cop_y.min, cop_y.max, icp_y_id_min, icp_y_id_max);

		for(int icp_x_id = icp_x_id_min; icp_x_id < icp_x_id_max; icp_x_id++)
		for(int icp_y_id = icp_y_id_min; icp_y_id < icp_y_id_max; icp_y_id++){
			Index2D icp_idx2(icp_x_id, icp_y_id);
			int icp_id = grid->xy.ToIndex(icp_idx2);
			cap_basin[0].push_back(CaptureState(swg_id, icp_id, 0, this));
			nstep_map[make_pair(swg_id, icp_id)] = 0;
		}
	}

	printf(" done: %d entries\n", (int)cap_basin[0].size());

	vec2_t icp;
	real_t tau;
	vec2_t mu;
	State  stnext;
	pair<vec2_t, vec2_t> icp_range;
	vec2_t ainv_range;

	int n = 1;
	while(n < nmax){
		printf(" calc %d step basin\n", n);
		bool added = false;

		printf("  first phase\n");
		
		vector< vector< pair<int, real_t> > > swg_tau_array;
		swg_tau_array.resize(grid->xy.Num());

		for(int basin_id = 0; basin_id < (int)cap_basin[n-1].size(); basin_id++){
			CaptureState& csnext = cap_basin[n-1][basin_id];
			stnext.swg = grid->xyr[swg_to_xyr[csnext.swg_id]];
			stnext.icp = grid->xy [csnext.icp_id];

			CalcMu(stnext.swg, stnext.icp, mu);
			CalcFeasibleIcpRange(mu, vec2_t(0.0, 1.0), icp_range);
			grid->x.IndexRange(icp_range.first.x, icp_range.second.x, icp_x_id_min, icp_x_id_max);
			grid->y.IndexRange(icp_range.first.y, icp_range.second.y, icp_y_id_min, icp_y_id_max);

			Index2D idx2;
			for(idx2[0] = icp_x_id_min; idx2[0] < icp_x_id_max; idx2[0]++)
			for(idx2[1] = icp_y_id_min; idx2[1] < icp_y_id_max; idx2[1]++){
				icp = grid->xy[idx2];
				// if icp is inside cop range, then it implies 0-step cap, so it can be skipped.
				if(IsInsideSupport(icp, eps)){
					continue;
				}
				// calc ainv_range to capture this icp
				ainv_range = vec2_t(0.0, 1.0);
				if(!CalcFeasibleAinvRange(mu, icp, ainv_range))
					continue;

				// minimum duration required
				tau = -T*log(ainv_range[0]);

				int icp_id = grid->xy.ToIndex(idx2);
				if(!swg_tau_array[icp_id].empty() && swg_tau_array[icp_id].back().first == csnext.swg_id){
					swg_tau_array[icp_id].back().second = std::max(swg_tau_array[icp_id].back().second, tau);
				}
				else{
					swg_tau_array[icp_id].push_back(make_pair(csnext.swg_id, tau));
				}
			}
		}

		printf("  first phase done\n");
		printf("  second phase\n");
		
		vector<bool> swg_id_array;
		swg_id_array.resize(swg_to_xyr.size());
			
		for(int icp_id = 0; icp_id < grid->xy.Num(); icp_id++){
			timer.CountUS();
			// create bitmap of swg
			EnumReachable(swg_tau_array[icp_id], swg_id_array);
			//for(pair<int, real_t>& swg_tau : swg_tau_array[icp_id]){
			//	EnumReachable(grid->xyr[swg_to_xyr[swg_tau.first]], swg_tau.second, swg_id_array);
			//}
			int T1 = timer.CountUS();

			for(int swg_id = 0; swg_id < swg_to_xyr.size(); swg_id++){
				if(!swg_id_array[swg_id])
					continue;

				NstepMap::iterator it = nstep_map.find(make_pair(swg_id, icp_id));
				if(it == nstep_map.end()){
					cap_basin[n].push_back(CaptureState(swg_id, icp_id, n, this));
					nstep_map[make_pair(swg_id, icp_id)] = n;
					added = true;
				}
			}
			int T2 = timer.CountUS();
			DSTR << T1 << " " << T2 << endl;
		}

		printf("  second phase done\n");
		
		if(!added)
			break;

		//CreateMuIndex(cap_basin[n]);
		printf("  %d\n", (int)cap_basin[n].size());

		n++;
	}
	printf("Done!\n");
}

template<class T>
void SaveArray(const string& filename, const vector<T>& arr){
	FILE* fp = fopen(filename.c_str(), "wb");
	fwrite(&arr[0], sizeof(T), arr.size(), fp);
	fclose(fp);
}

template<class T>
bool LoadArray(const string& filename, vector<T>& arr){
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

void Capturability::Save(const string& basename){
	stringstream ss;
	for(int n = 0; n < (int)cap_basin.size(); n++){
		if(cap_basin[n].empty())
			break;

		ss.str("");
		ss << basename << "basin" << n << ".bin";
		SaveArray(ss.str(), cap_basin[n]);
	}
	SaveArray(basename + "swg_to_xyr.bin"  , swg_to_xyr);
	SaveArray(basename + "xyr_to_swg.bin"  , xyr_to_swg);
	//SaveArray(basename + "duration_map.bin", duration_map);
	//SaveArray(basename + "icp_map.bin"     , icp_map     );
}

void Capturability::Load(const string& basename) {
	stringstream ss;
	for(int n = 0; n < nmax; n++){
		ss.str("");
		ss << basename << "basin" << n << ".bin";
		LoadArray(ss.str(), cap_basin[n]);

		//CreateMuIndex(cap_basin[n]);
	}

	LoadArray(basename + "swg_to_xyr.bin" , swg_to_xyr );
	LoadArray(basename + "xyr_to_swg.bin" , xyr_to_swg );
	//LoadArray(basename + "duration_map.bin", duration_map);
	//LoadArray(basename + "icp_map.bin"     , icp_map     );
}

void Capturability::GetCaptureBasin(const State& st, int nstepMin, int nstepMax, CaptureBasin& basin, vector<vec2_t>& tau_range_valid){
	basin.clear();

	real_t tau_min;
	vec2_t tau_range;
	vec2_t ainv_range;
	vec2_t mu;
	State  stnext;

	for(int n = nstepMin; n <= nstepMax; n++){
		if((int)cap_basin.size() < n+1)
			break;

		for(int basin_id = 0; basin_id < (int)cap_basin[n].size(); basin_id++){
			CaptureState& csnext = cap_basin[n][basin_id];
			stnext.swg = grid->xyr[swg_to_xyr[csnext.swg_id]];
			stnext.icp = grid->xy [csnext.icp_id];

			//CalcFeasibleIcpRange(swg_id, csnext, icp_range);
			//if( icp_range.first.x <= st.icp.x && st.icp.x <= icp_range.second.x &&
			//	icp_range.first.y <= st.icp.y && st.icp.y <= icp_range.second.y ){
			tau_min = CalcMinDuration(st.swg, stnext.swg);
			ainv_range[0] = 0.0;
			ainv_range[1] = exp(-tau_min/T);
			CalcMu(stnext.swg, stnext.icp, mu);
			
			if(CalcFeasibleAinvRange(mu, st.icp, ainv_range)){
				CalcTauRange(ainv_range, tau_range);
				basin.push_back(csnext);
				tau_range_valid.push_back(tau_range);
			}
		}
	}
}

/*
bool Capturability::IsCapturable(int swg_id, int icp_id, int& nstep) {
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
*/
bool Capturability::FindNearest(const State& st, const Input& in_ref, const State& stnext_ref, CaptureState& cs_opt, real_t& tau_opt, int& n_opt){
	real_t d_opt = inf;
	real_t d_swg = 0.0;
	real_t d_tau = 0.0;
	real_t d_icp = 0.0;
	real_t d     = 0.0;
	int    n0 = 0, n1 = 0, n2 = 0, n3 = 0, n4 = 0;
	State  stnext;
	vec2_t mu;
	real_t tau_min;
	vec2_t tau_range;
	vec2_t ainv_range;
	
	tau_opt = 0.0;
	n_opt   = 0;

	pair<vec2_t, vec2_t>  mu_range;

	ainv_range[0] = 0.01;
	ainv_range[1] = exp(-swing->dsp_duration/T);
	CalcFeasibleMuRange(st.icp, ainv_range, mu_range);
	//grid->x.IndexRange(mu_range.first.x, mu_range.second.x, mu_x_id_min, mu_x_id_max);
	//grid->y.IndexRange(mu_range.first.y, mu_range.second.y, mu_y_id_min, mu_y_id_max);

	for(int n = 0; n < nmax; n++){
		if(step_weight*n >= d_opt)
			break;
		if((int)cap_basin.size() < n+1 || cap_basin[n].empty())
			break;

		int swg_id_prev = -1;
		for(int basin_id = 0; basin_id < (int)cap_basin[n].size(); basin_id++){
			CaptureState& csnext = cap_basin[n][basin_id];
			//grid->xy.FromIndex(csnext.mu_id, mu_idx2);
			//if( !(mu_x_id_min <= mu_idx2[0] && mu_idx2[0] < mu_x_id_max) ||
			//	!(mu_y_id_min <= mu_idx2[1] && mu_idx2[1] < mu_y_id_max) )
			//	continue;

			stnext.swg = grid->xyr[swg_to_xyr[csnext.swg_id]];
			stnext.icp = grid->xy [csnext.icp_id];
			CalcMu(stnext.swg, stnext.icp, mu);
			n0++;

			if( !(mu_range.first[0] <= mu.x && mu.x <= mu_range.second[0]) ||
				!(mu_range.first[1] <= mu.y && mu.y <= mu_range.second[1]) )
				continue;

			n1++;
			if(csnext.swg_id != swg_id_prev){
				d_swg  = (stnext.swg - stnext_ref.swg).square();
				swg_id_prev = csnext.swg_id;
			}
			if(step_weight*n + swg_weight*d_swg >= d_opt)
				continue;

			n2++;
			ainv_range = vec2_t(0.0, 1.0);
			if(!CalcFeasibleAinvRange(mu, st.icp, ainv_range))
				continue;

			n3++;
			CalcTauRange(ainv_range, tau_range);
			tau_min = CalcMinDuration(st.swg, stnext.swg);
			
			if( tau_min > tau_range[1] )
				continue;
			
			n4++;
			real_t tau = std::min(std::max(tau_range[0], in_ref.tau), tau_range[1]);
			d_tau = (tau - in_ref.tau)*(tau - in_ref.tau);
			d_icp = (stnext.icp - stnext_ref.icp).square();
			d     = step_weight*n + swg_weight*d_swg + tau_weight*d_tau + icp_weight*d_icp;
			if( d < d_opt ){
				cs_opt  = csnext;
				tau_opt = tau;
				d_opt   = d;
				n_opt   = n;
			}
		}
	}

	printf("d_opt: %f  n_opt: %d  %d %d %d %d %d\n", d_opt, n_opt, n0, n1, n2, n3, n4);

	return d_opt != inf;
}

bool Capturability::Check(const State& st, Input& in, State& stnext, int& nstep, bool& modified){
	modified = false;

	bool next_ok;
	bool cop_ok;
	bool tau_ok;
	/*
	int next_swg_id = xyr_to_swg[grid->xyr.ToIndex(grid->xyr.Round(stnext.swg))];
	int next_icp_id = grid->xy.ToIndex(grid->xy.Round(stnext.icp));
	printf("next state id: %d,%d\n", next_swg_id, next_icp_id);
	
	if(next_swg_id == -1){
		printf("next swg is invalid\n");
		return false;
	}

	// check if next state is in capture basin
	nstep = -1;
	if(IsCapturable(next_swg_id, next_icp_id, nstep)){
		printf("next state is %d-step capturable\n", nstep);
		next_ok = true;
	}
	else{
		printf("next state is not capturable\n");
		next_ok = false;
	}
	*/
	// calculate cop
	CalcInput(st, stnext, in);
	// check if cop is inside support region
	//printf("cop(local): %f,%f\n", in.cop.x(), in.cop.y());
	if(IsInsideSupport(in.cop, 0.01)){
		printf("cop is inside support\n");
		cop_ok = true;
	}
	else{
		printf("cop is outside support\n");
		cop_ok = false;
	}
	
	/*
	// check duration
	swing->SetSwg (vec3_t(st.swg .x, st.swg .y, st.swg.z), st.swg[3] );
	swing->SetLand(vec3_t(in.land.x, in.land.y, 0.0     ), in.land[2]);
	real_t tau_min = swing->GetMinDuration();
	if(in.tau >= tau_min){
		tau_ok = true;
	}
	else{
		tau_ok = false;
		printf("step duration too small\n");
	}
	*/
	if(/*next_ok &&*/ cop_ok /*&& tau_ok*/){
		return true;
	}

	// find modified next state that can be transitioned from current state and is capturable
	CaptureState cs_opt;
	real_t tau_opt;
	if(!FindNearest(st, in, stnext, cs_opt, tau_opt, nstep)){
		printf("no capturable state found\n");
		return false;
	}
	printf("modified next state: %d,%d  %d-step capturable transition\n", cs_opt.swg_id, cs_opt.icp_id, cs_opt.nstep);

	stnext.swg = grid->xyr[swg_to_xyr[cs_opt.swg_id]];
	stnext.icp = grid->xy [cs_opt.icp_id];
	
	in.tau = tau_opt;
	CalcInput(st, stnext, in);
	
	modified = true;
	return true;
}

} // namespace Capt