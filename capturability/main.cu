#include "cuda_analysis.cuh"

const int BPG = 65535; // Blocks  Per Grid  (max: 65535)
const int TPB = 1024;  // Threads Per Block (max: 1024)

// typedef Cuda::GridCartesian grid_t;
typedef Cuda::GridPolar grid_t;

int main(void) {
  /* 前処理 */
  /* ---------------------------------------------------------------------- */
  printf("Prepare...\t");

  /* パラメータの読み込み */
  Capt::Model cmodel("nao.xml");
  Capt::Param cparam("analysis.xml");

  /* グリッド */
  Capt::Grid cgrid(cparam);
  const int  num_state = cgrid.getNumState();
  const int  num_input = cgrid.getNumInput();
  const int  num_grid  = num_state * num_input;

  /* 解析条件 */
  Cuda::Condition cond;
  cond.model = &cmodel;
  cond.param = &cparam;
  cond.grid  = &cgrid;

  /* メモリー管理 */
  Cuda::MemoryManager mm;

  /* パラメータの用意 */
  // ホスト側
  Cuda::State   *state   = (Cuda::State*)malloc(sizeof( Cuda::State ) * num_state);
  Cuda::Input   *input   = (Cuda::Input*)malloc(sizeof( Cuda::Input ) * num_input );
  int           *trans   = (int*)malloc(sizeof( int ) * num_state * num_input );
  int           *basin   = (int*)malloc(sizeof( int ) * num_state );
  int           *nstep   = (int*)malloc(sizeof( int ) * num_grid );
  grid_t        *grid    = new grid_t;
  Cuda::Vector2 *cop     = new Cuda::Vector2[num_state];
  Cuda::Physics *physics = new Cuda::Physics;
  mm.initHostState(state, cond);
  mm.initHostTrans(trans, cond);
  mm.initHostInput(input, cond);
  mm.initHostBasin(basin, cond);
  mm.initHostNstep(nstep, cond);
  mm.initHostGrid(grid, cond);
  mm.initCop(cop, cond);
  mm.initPhysics(physics, cond);
  mm.setGrid(grid);
  // デバイス側
  Cuda::State   *dev_state;
  Cuda::Input   *dev_input;
  int           *dev_basin;
  int           *dev_nstep;
  int           *dev_trans;
  grid_t        *dev_grid;
  Cuda::Vector2 *dev_cop;
  Cuda::Physics *dev_physics;
  // mm.initDevState(dev_state);
  // mm.initDevInput(dev_input);
  // mm.initDevTrans(dev_trans);
  // mm.initDevBasin(dev_basin);
  // mm.initDevNstep(dev_nstep);
  // mm.initDevGrid(dev_grid);
  // mm.initDevCop(dev_cop);
  // mm.initDevPhysics(dev_physics);
  HANDLE_ERROR(cudaMalloc( (void **)&dev_state, num_state * sizeof( Cuda::State ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_input, num_input * sizeof( Cuda::Input ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_trans, num_grid * sizeof( int ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_basin, num_state * sizeof( int ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_nstep, num_grid * sizeof( int ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_grid, sizeof( grid_t ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_cop, num_state * sizeof( Cuda::Vector2 ) ) );
  HANDLE_ERROR(cudaMalloc( (void **)&dev_physics, sizeof( Cuda::Physics ) ) );

  // ホスト側からデバイス側にコピー
  mm.copyHostToDevState(state, dev_state);
  mm.copyHostToDevInput(input, dev_input);
  mm.copyHostToDevTrans(trans, dev_trans);
  mm.copyHostToDevBasin(basin, dev_basin);
  mm.copyHostToDevNstep(nstep, dev_nstep);
  mm.copyHostToDevGrid(grid, dev_grid);
  mm.copyHostToDevCop(cop, dev_cop);
  mm.copyHostToDevPhysics(physics, dev_physics);

  printf("Done.\n");
  /* ------------------------------------------------------------------------ */

  /* 状態遷移計算 */
  /* ---------------------------------------------------------------------- */
  printf("Calculate...\t");

  Cuda::calcStateTrans << < BPG, TPB >> > ( dev_state, dev_input, dev_trans, dev_grid, dev_cop, dev_physics );
  HANDLE_ERROR(cudaMemcpy(trans, dev_trans, num_grid * sizeof( int ),
                          cudaMemcpyDeviceToHost) );

  printf("Done.\n");
  /* ---------------------------------------------------------------------- */

  /* 解析実行 */
  /* ---------------------------------------------------------------------- */
  printf("Execute...\n");

  printf("\t0-step\n");
  Cuda::exeZeroStep(cgrid, cmodel, basin);
  HANDLE_ERROR(cudaMemcpy(dev_basin, basin, num_state * sizeof( int ),
                          cudaMemcpyHostToDevice) );

  for(int N = 0; N < NUM_STEP_MAX; N++) {
    printf("\t%d-step\n", N);
    Cuda::exeNStep << < BPG, TPB >> > ( N, dev_basin, dev_nstep,
                                        dev_trans, dev_grid );
    HANDLE_ERROR(cudaMemcpy(basin, dev_basin, num_state * sizeof( int ),
                            cudaMemcpyDeviceToHost) );
  }

  printf("\t\tDone.\n");
  /* ---------------------------------------------------------------------- */

  /* 解析結果をデバイス側からホスト側にコピー */
  /* ---------------------------------------------------------------------- */
  HANDLE_ERROR(cudaMemcpy(basin, dev_basin, num_state * sizeof( int ),
                          cudaMemcpyDeviceToHost) );
  HANDLE_ERROR(cudaMemcpy(nstep, dev_nstep, num_grid * sizeof( int ),
                          cudaMemcpyDeviceToHost) );
  HANDLE_ERROR(cudaMemcpy(trans, dev_trans, num_grid * sizeof( int ),
                          cudaMemcpyDeviceToHost) );
  /* ---------------------------------------------------------------------- */

  /* ファイル書き出し */
  /* ---------------------------------------------------------------------- */
  printf("Output...\t");
  Cuda::outputBasin("BasinGpu.csv", cond, basin, false);
  Cuda::outputNStep("NstepGpu.csv", cond, nstep, trans, false);
  printf("Done.\n");
  /* ---------------------------------------------------------------------- */

  /* 終了処理 */
  /* ---------------------------------------------------------------------- */
  printf("Finish...\t");

  /* メモリの開放 */
  // ホスト側
  delete state;
  delete input;
  delete basin;
  delete nstep;
  delete trans;
  delete grid;
  delete cop;
  delete physics;
  // デバイス側
  cudaFree(dev_state);
  cudaFree(dev_input);
  cudaFree(dev_trans);
  cudaFree(dev_basin);
  cudaFree(dev_nstep);
  cudaFree(dev_grid);
  cudaFree(dev_cop);
  cudaFree(dev_physics);

  printf("Done.\n");
  /* ---------------------------------------------------------------------- */

  return 0;
}