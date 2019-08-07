#include "cuda_analysis.cuh"

const int BPG = 1024; // Blocks  Per Grid  (max: 65535)
const int TPB = 1024; // Threads Per Block (max: 1024)

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

  /* グリッド */
  // ホスト側
  Cuda::State *state   = new Cuda::State[num_state];
  Cuda::Input *input   = new Cuda::Input[num_input];
  int         *basin   = new int[num_state]; // if 0     => 0-step capturable
  int         *nstep   = new int[num_grid];  // if n(>1) => n-step capturable
  int         *next_id = new int[num_grid];  // next state id
  Cuda::Grid  *grid    = new Cuda::Grid;
  initState(state, next_id, cond);
  initInput(input, cond);
  initNstep(basin, nstep, cond);
  initGrid(grid, cond);
  // デバイス側
  Cuda::State *dev_state;
  Cuda::Input *dev_input;
  int         *dev_basin;
  int         *dev_nstep;
  int         *dev_next_id;
  Cuda::Grid  *dev_grid;
  HANDLE_ERROR(
    cudaMalloc((void **)&dev_state, num_state * sizeof(Cuda::State)));
  HANDLE_ERROR(
    cudaMalloc((void **)&dev_input, num_input * sizeof(Cuda::Input)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_basin, num_state * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_nstep, num_grid * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_next_id, num_grid * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_grid, sizeof(Cuda::Grid)));
  HANDLE_ERROR(cudaMemcpy(dev_state, state, num_state * sizeof(Cuda::State),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_input, input, num_input * sizeof(Cuda::Input),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_nstep, nstep, num_grid * sizeof(int),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_next_id, next_id, num_grid * sizeof(int),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(
    cudaMemcpy(dev_grid, grid, sizeof(Cuda::Grid), cudaMemcpyHostToDevice));

  /* CoP List */
  // ホスト側
  Cuda::Vector2 *cop = new Cuda::Vector2[num_state];
  initCop(cop, cond);
  // デバイス側
  Cuda::Vector2 *dev_cop;
  HANDLE_ERROR(
    cudaMalloc((void **)&dev_cop, num_state * sizeof(Cuda::Vector2)));
  HANDLE_ERROR(cudaMemcpy(dev_cop, cop, num_state * sizeof(Cuda::Vector2),
                          cudaMemcpyHostToDevice));

  /* 物理情報 */
  // ホスト側
  Cuda::Physics *physics = new Cuda::Physics;
  initPhysics(physics, cond);
  // デバイス側
  Cuda::Physics *dev_physics;
  HANDLE_ERROR(cudaMalloc((void **)&dev_physics, sizeof(Cuda::Physics)));
  HANDLE_ERROR(cudaMemcpy(dev_physics, physics, sizeof(Cuda::Physics),
                          cudaMemcpyHostToDevice));

  printf("Done.\n");
  /* ------------------------------------------------------------------------ */

  /* 状態遷移計算 */
  /* ---------------------------------------------------------------------- */
  printf("Calculate...\t");

  Cuda::calcStateTrans << < BPG, TPB >> > (dev_state, dev_input, dev_next_id, dev_grid, dev_cop, dev_physics);
  HANDLE_ERROR(cudaMemcpy(next_id, dev_next_id, num_grid * sizeof(int),
                          cudaMemcpyDeviceToHost));

  printf("Done.\n");
  /* ---------------------------------------------------------------------- */

  /* 解析実行 */
  /* ---------------------------------------------------------------------- */
  printf("Execute...\n");

  printf("\t0-step\n");
  Cuda::exeZeroStep(cgrid, cmodel, basin);
  HANDLE_ERROR(cudaMemcpy(dev_basin, basin, num_state * sizeof(int),
                          cudaMemcpyHostToDevice));

  int  N    = 1;
  bool flag = true;
  while (flag) {
    printf("\t%d-step\n", N);
    Cuda::exeNStep << < BPG, TPB >> > ( N, dev_basin, dev_nstep,
                                        dev_next_id, dev_grid);
    HANDLE_ERROR(cudaMemcpy(basin, dev_basin, num_state * sizeof(int),
                            cudaMemcpyDeviceToHost));

    flag = false;
    // for (int i = 0; i < num_state; i++) {
    //   if (basin[i] == -1) // basin[i] != prev_basin[i]とするべき
    //     flag = true;
    // }
    if(N < 3)
      flag = true;
    N++;
  }

  printf("\t\tDone.\n");
  /* ---------------------------------------------------------------------- */

  /* 解析結果をデバイス側からホスト側にコピー */
  /* ---------------------------------------------------------------------- */
  HANDLE_ERROR(cudaMemcpy(basin, dev_basin, num_state * sizeof(int),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(nstep, dev_nstep, num_grid * sizeof(int),
                          cudaMemcpyDeviceToHost));
  HANDLE_ERROR(cudaMemcpy(next_id, dev_next_id, num_grid * sizeof(int),
                          cudaMemcpyDeviceToHost));
  /* ---------------------------------------------------------------------- */

  /* ファイル書き出し */
  /* ---------------------------------------------------------------------- */
  printf("Output...\t");
  Cuda::outputBasin("Basin.csv", false, cond, basin);
  Cuda::outputNStep("Nstep.csv", false, cond, nstep, next_id);
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
  delete next_id;
  delete grid;
  delete cop;
  delete physics;
  // デバイス側
  cudaFree(dev_state);
  cudaFree(dev_input);
  cudaFree(dev_next_id);
  cudaFree(dev_basin);
  cudaFree(dev_nstep);
  cudaFree(dev_grid);
  cudaFree(dev_cop);
  cudaFree(dev_physics);

  printf("Done.\n");
  /* ---------------------------------------------------------------------- */

  return 0;
}