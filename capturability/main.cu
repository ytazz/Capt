#include "cuda_analysis.cuh"
#include <chrono>

int main(void) {
  std::chrono::system_clock::time_point start, end_exe, end_save;
  start = std::chrono::system_clock::now();

  /* パラメータの読み込み */
  Capt::Model *cmodel = new Capt::Model("data/valkyrie.xml");
  Capt::Param *cparam = new Capt::Param("data/valkyrie_xy.xml");
  Capt::Grid  *cgrid  = new Capt::Grid(cparam);

  /* 前処理 */
  printf("*** Analysis ***\n");
  printf("  Initializing ... ");
  fflush(stdout);

  /* 各変数のsize */
  const int state_num = cgrid->getNumState();
  const int input_num = cgrid->getNumInput();
  const int grid_num  = state_num * input_num;

  int foot_r_num, foot_l_num;
  cmodel->read(&foot_r_num, "foot_r_convex_num");
  cmodel->read(&foot_l_num, "foot_l_convex_num");

  const int num_vertex = foot_r_num + foot_l_num;
  const int num_swf    = cgrid->getNumInput();

  /* メモリ管理 */
  Cuda::MemoryManager mm;
  mm.set(cmodel, cparam, cgrid);

  /* パラメータの用意 */
  // ホスト側
  Cuda::State   *state     = (Cuda::State*)malloc(sizeof( Cuda::State ) * state_num);
  Cuda::Input   *input     = (Cuda::Input*)malloc(sizeof( Cuda::Input ) * input_num );
  int           *trans     = (int*)malloc(sizeof( int ) * state_num * input_num );
  int           *basin     = (int*)malloc(sizeof( int ) * state_num );
  int           *nstep     = (int*)malloc(sizeof( int ) * grid_num );
  Cuda::Grid    *grid      = (Cuda::Grid*)malloc(sizeof( Cuda::Grid ) );
  Cuda::vec2_t  *foot_r    = (Cuda::vec2_t*)malloc(sizeof( Cuda::vec2_t ) * foot_r_num );
  Cuda::vec2_t  *foot_l    = (Cuda::vec2_t*)malloc(sizeof( Cuda::vec2_t ) * foot_l_num );
  Cuda::vec2_t  *convex    = (Cuda::vec2_t*)malloc(sizeof( Cuda::vec2_t ) * num_swf * num_vertex );
  Cuda::vec2_t  *cop       = (Cuda::vec2_t*)malloc(sizeof( Cuda::vec2_t ) * state_num );
  double        *step_time = (double*)malloc(sizeof( double ) * grid_num );
  Cuda::Physics *physics   = (Cuda::Physics*)malloc(sizeof( Cuda::Physics ) );
  mm.initHostState(state);
  mm.initHostTrans(trans);
  mm.initHostInput(input);
  mm.initHostBasin(basin);
  mm.initHostNstep(nstep);
  mm.initHostGrid(grid);
  mm.initHostFootR(foot_r);
  mm.initHostFootL(foot_l);
  mm.initHostConvex(convex);
  mm.initHostCop(cop);
  mm.initHostStepTime(step_time);
  mm.initHostPhysics(physics);
  // デバイス側
  Cuda::State   *dev_state;
  Cuda::Input   *dev_input;
  int           *dev_basin;
  int           *dev_nstep;
  int           *dev_trans;
  Cuda::Grid    *dev_grid;
  Cuda::vec2_t  *dev_foot_r;
  Cuda::vec2_t  *dev_foot_l;
  Cuda::vec2_t  *dev_convex;
  Cuda::vec2_t  *dev_cop;
  double        *dev_step_time;
  Cuda::Physics *dev_physics;
  HANDLE_ERROR(cudaMalloc( (void **)&dev_state, state_num * sizeof( Cuda::State ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_input, input_num * sizeof( Cuda::Input ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_trans, grid_num * sizeof( int ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_basin, state_num * sizeof( int ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_nstep, grid_num * sizeof( int ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_grid, sizeof( Cuda::Grid ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_foot_r, foot_r_num * sizeof( Cuda::vec2_t ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_foot_l, foot_l_num * sizeof( Cuda::vec2_t ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_convex, num_swf * num_vertex * sizeof( Cuda::vec2_t ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_cop, state_num * sizeof( Cuda::vec2_t ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_step_time, grid_num * sizeof( double ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_physics, sizeof( Cuda::Physics ) ) );
  // ホスト側からデバイス側にコピー
  mm.copyHostToDevState(state, dev_state);
  mm.copyHostToDevInput(input, dev_input);
  mm.copyHostToDevTrans(trans, dev_trans);
  mm.copyHostToDevBasin(basin, dev_basin);
  mm.copyHostToDevNstep(nstep, dev_nstep);
  mm.copyHostToDevGrid(grid, dev_grid);
  mm.copyHostToDevFootR(foot_r, dev_foot_r);
  mm.copyHostToDevFootL(foot_l, dev_foot_l);
  mm.copyHostToDevConvex(convex, dev_convex);
  mm.copyHostToDevCop(cop, dev_cop);
  mm.copyHostToDevStepTime(step_time, dev_step_time);
  mm.copyHostToDevPhysics(physics, dev_physics);

  printf("Done!\n");

  /* 状態遷移計算 */
  printf("  Calculating .... ");
  fflush(stdout);
  Cuda::calcCop << < BPG, TPB >> > ( dev_state, dev_grid, dev_foot_r, dev_cop );
  Cuda::calcStepTime << < BPG, TPB >> > ( dev_state, dev_input, dev_grid, dev_step_time, dev_physics );
  Cuda::calcBasin << < BPG, TPB >> > ( dev_state, dev_basin, dev_grid, dev_foot_r, dev_foot_l, dev_convex );
  Cuda::calcTrans << < BPG, TPB >> > ( dev_state, dev_input, dev_trans, dev_grid, dev_cop, dev_step_time, dev_physics );
  printf("Done!\n");

  /* 解析実行 */
  printf("  Analysing ...... ");
  fflush(stdout);

  int  step = 0;
  bool flag = true;
  while( flag ) {
    step++;
    Cuda::exeNstep << < BPG, TPB >> > ( step, dev_basin, dev_nstep, dev_trans, dev_grid );

    mm.copyDevToHostNstep(dev_nstep, nstep);
    flag = false;
    for(int id = 0; id < grid_num; id++) {
      if(nstep[id] == step)
        flag = true;
    }
  }
  step--;
  end_exe = std::chrono::system_clock::now();
  printf("Done!\n");

  /* 解析結果をデバイス側からホスト側にコピー */
  mm.copyDevToHostBasin(dev_basin, basin);
  mm.copyDevToHostNstep(dev_nstep, nstep);
  mm.copyDevToHostTrans(dev_trans, trans);
  mm.copyDevToHostCop(dev_cop, cop);
  mm.copyDevToHostStepTime(dev_step_time, step_time);

  /* ファイル書き出し */
  mm.saveBasin("gpu/Basin.csv", basin);
  mm.saveNstep("gpu/Nstep.csv", nstep, trans, step);
  mm.saveCop("gpu/Cop.csv", cop);
  mm.saveStepTime("gpu/StepTime.csv", step_time);
  end_save = std::chrono::system_clock::now();

  /* 処理時間 */
  int time_exe  = std::chrono::duration_cast<std::chrono::milliseconds>(end_exe - start).count();
  int time_save = std::chrono::duration_cast<std::chrono::milliseconds>(end_save - end_exe).count();
  int time_sum  = std::chrono::duration_cast<std::chrono::milliseconds>(end_save - start).count();
  printf("*** Time ***\n");
  printf("  exe : %7d [ms]\n", time_exe);
  printf("  save: %7d [ms]\n", time_save);
  printf("  sum : %7d [ms]\n", time_sum);

  /* メモリの開放 */
  // ホスト側
  delete state;
  delete input;
  delete basin;
  delete nstep;
  delete trans;
  delete grid;
  delete foot_r;
  delete foot_l;
  delete convex;
  delete cop;
  delete step_time;
  delete physics;
  // デバイス側
  cudaFree(dev_state);
  cudaFree(dev_input);
  cudaFree(dev_trans);
  cudaFree(dev_basin);
  cudaFree(dev_nstep);
  cudaFree(dev_grid);
  cudaFree(dev_foot_r);
  cudaFree(dev_foot_l);
  cudaFree(dev_convex);
  cudaFree(dev_cop);
  cudaFree(dev_step_time);
  cudaFree(dev_physics);

  return 0;
}