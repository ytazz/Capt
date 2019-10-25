#include "cuda_analysis.cuh"

const int BPG = 65535; // Blocks  Per Grid  (max: 65535)
const int TPB = 1024;  // Threads Per Block (max: 1024)

typedef Cuda::GridCartesian grid_t;
// typedef Cuda::GridPolar grid_t;

int main(void) {
  /* 前処理 */
  /* ---------------------------------------------------------------------- */
  printf("Prepare...\t");

  /* パラメータの読み込み */
  Capt::Model cmodel("data/nao.xml");
  Capt::Param cparam("data/nao_xy.xml");

  /* グリッド */
  Capt::Grid cgrid(cparam);
  const int  num_state  = cgrid.getNumState();
  const int  num_input  = cgrid.getNumInput();
  const int  num_grid   = num_state * num_input;
  const int  num_foot_r = cmodel.getVec("foot", "foot_r_convex").size();
  const int  num_foot_l = cmodel.getVec("foot", "foot_l_convex").size();
  const int  num_vertex = num_foot_r + num_foot_l;
  const int  num_swf    = cgrid.getNumInput();

  /* 解析条件 */
  Cuda::Condition cond;
  cond.model = &cmodel;
  cond.param = &cparam;
  cond.grid  = &cgrid;

  /* メモリ管理 */
  Cuda::MemoryManager mm;

  /* パラメータの用意 */
  // ホスト側
  Cuda::State   *state     = (Cuda::State*)malloc(sizeof( Cuda::State ) * num_state);
  Cuda::Input   *input     = (Cuda::Input*)malloc(sizeof( Cuda::Input ) * num_input );
  int           *trans     = (int*)malloc(sizeof( int ) * num_state * num_input );
  int           *basin     = (int*)malloc(sizeof( int ) * num_state );
  int           *nstep     = (int*)malloc(sizeof( int ) * num_grid );
  grid_t        *grid      = new grid_t;
  Cuda::Vector2 *foot_r    = new Cuda::Vector2[num_foot_r];
  Cuda::Vector2 *foot_l    = new Cuda::Vector2[num_foot_l];
  Cuda::Vector2 *convex    = new Cuda::Vector2[num_swf * num_vertex];
  Cuda::Vector2 *cop       = new Cuda::Vector2[num_state];
  double        *step_time = (double*)malloc(sizeof( double ) * num_grid );
  Cuda::Physics *physics   = new Cuda::Physics;
  mm.initHostState(state, cond);
  mm.initHostTrans(trans, cond);
  mm.initHostInput(input, cond);
  mm.initHostBasin(basin, cond);
  mm.initHostNstep(nstep, cond);
  mm.initHostGrid(grid, cond);
  mm.initHostFootR(foot_r, cond);
  mm.initHostFootL(foot_l, cond);
  mm.initHostConvex(convex, cond);
  mm.initHostCop(cop, cond);
  mm.initHostStepTime(step_time, cond);
  mm.initHostPhysics(physics, cond);
  mm.setGrid(grid);
  // デバイス側
  Cuda::State   *dev_state;
  Cuda::Input   *dev_input;
  int           *dev_basin;
  int           *dev_nstep;
  int           *dev_trans;
  grid_t        *dev_grid;
  Cuda::Vector2 *dev_foot_r;
  Cuda::Vector2 *dev_foot_l;
  Cuda::Vector2 *dev_convex;
  Cuda::Vector2 *dev_cop;
  double        *dev_step_time;
  Cuda::Physics *dev_physics;
  HANDLE_ERROR(cudaMalloc( (void **)&dev_state, num_state * sizeof( Cuda::State ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_input, num_input * sizeof( Cuda::Input ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_trans, num_grid * sizeof( int ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_basin, num_state * sizeof( int ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_nstep, num_grid * sizeof( int ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_grid, sizeof( grid_t ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_foot_r, num_foot_r * sizeof( Cuda::Vector2 ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_foot_l, num_foot_l * sizeof( Cuda::Vector2 ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_convex, num_swf * num_vertex * sizeof( Cuda::Vector2 ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_cop, num_state * sizeof( Cuda::Vector2 ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_step_time, num_grid * sizeof( double ) ) );
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

  printf("Done.\n");
  /* ------------------------------------------------------------------------ */

  /* 状態遷移計算 */
  /* ---------------------------------------------------------------------- */
  printf("Calculate...\t");
  // Cuda::calcCop << < BPG, TPB >> > ( dev_state, dev_grid, dev_foot_r, dev_cop );
  // Cuda::calcStepTime << < BPG, TPB >> > ( dev_state, dev_input, dev_grid, dev_step_time, dev_physics );
  Cuda::calcBasin << < BPG, TPB >> > ( dev_state, dev_basin, dev_grid, dev_foot_r, dev_foot_l, dev_convex );
  // Cuda::calcTrans << < BPG, TPB >> > ( dev_state, dev_input, dev_trans, dev_grid, dev_cop, dev_step_time, dev_physics );
  printf("Done.\n");
  /* ---------------------------------------------------------------------- */

  /* 解析実行 */
  /* ---------------------------------------------------------------------- */
  // printf("Execute...\n");
  // for(int N = 1; N <= NUM_STEP_MAX; N++) {
  //   printf("\t%d-step\n", N);
  //   Cuda::exeNStep << < BPG, TPB >> > ( N, dev_basin, dev_nstep, dev_trans, dev_grid );
  // }
  // printf("\t\tDone.\n");
  /* ---------------------------------------------------------------------- */

  /* 解析結果をデバイス側からホスト側にコピー */
  /* ---------------------------------------------------------------------- */
  HANDLE_ERROR(cudaMemcpy(basin, dev_basin, num_state * sizeof( int ),
                          cudaMemcpyDeviceToHost) );
  HANDLE_ERROR(cudaMemcpy(nstep, dev_nstep, num_grid * sizeof( int ),
                          cudaMemcpyDeviceToHost) );
  HANDLE_ERROR(cudaMemcpy(trans, dev_trans, num_grid * sizeof( int ),
                          cudaMemcpyDeviceToHost) );
  HANDLE_ERROR(cudaMemcpy(cop, dev_cop, num_state * sizeof( Cuda::Vector2 ),
                          cudaMemcpyDeviceToHost) );
  mm.copyDevToHostStepTime(dev_step_time, step_time);
  /* ---------------------------------------------------------------------- */

  /* ファイル書き出し */
  /* ---------------------------------------------------------------------- */
  printf("Output...\t");
  Cuda::outputBasin("csv/BasinGpu.csv", cond, basin);
  Cuda::outputNStep("csv/NstepGpu.csv", cond, nstep, trans);
  Cuda::outputCop("csv/Cop.csv", cond, cop);
  Cuda::outputStepTime("csv/StepTime.csv", cond, step_time);
  printf("Done.\n");
  /* ---------------------------------------------------------------------- */

  /* 終了処理 */
  /* ---------------------------------------------------------------------- */
  printf("Finish...\t");

  FILE *fp = fopen("debug.csv", "w");
  fprintf(fp, "id, x, y\n");
  for(int i = 0; i < num_swf * num_vertex; i++) {
    fprintf(fp, "%d, %lf, %lf\n", i, convex[i].x_, convex[i].y_);
  }
  fclose(fp);

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

  printf("Done.\n");
  /* ---------------------------------------------------------------------- */

  return 0;
}