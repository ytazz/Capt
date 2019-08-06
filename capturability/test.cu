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
  Cuda::State *state     = new Cuda::State[num_state];
  Cuda::Input *input     = new Cuda::Input[num_input];
  int         *zero_step = new int[num_state]; // if 0     => 0-step capturable
  int         *n_step    = new int[num_grid];  // if n(>1) => n-step capturable
  int         *next_id   = new int[num_grid];  // next state id
  Cuda::Grid  *grid      = new Cuda::Grid;
  initState(state, next_id, cond);
  initInput(input, cond);
  initNstep(zero_step, n_step, cond);
  initGrid(grid, cond);
  // デバイス側
  Cuda::State *dev_state;
  Cuda::Input *dev_input;
  int         *dev_zero_step;
  int         *dev_n_step;
  int         *dev_next_id;
  Cuda::Grid  *dev_grid;
  HANDLE_ERROR(
    cudaMalloc((void **)&dev_state, num_state * sizeof(Cuda::State)));
  HANDLE_ERROR(
    cudaMalloc((void **)&dev_input, num_input * sizeof(Cuda::Input)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_zero_step, num_state * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_n_step, num_grid * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_next_id, num_grid * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_grid, sizeof(Cuda::Grid)));
  HANDLE_ERROR(cudaMemcpy(dev_state, state, num_state * sizeof(Cuda::State),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_input, input, num_input * sizeof(Cuda::Input),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_n_step, n_step, num_grid * sizeof(int),
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
  Cuda::exeZeroStep(cgrid, cmodel, zero_step);

  int  N    = 1;
  bool flag = false;
  while (flag) {
    printf("\t%d-step\n", N);
    Cuda::exeNStep << < BPG, TPB >> > ( N, dev_zero_step, dev_n_step,
                                        dev_next_id, dev_grid);
    HANDLE_ERROR(cudaMemcpy(n_step, dev_n_step, num_grid * sizeof(int),
                            cudaMemcpyDeviceToHost));
    flag = false;
    for (int i = 0; i < num_grid; i++) {
      if (n_step[i] == -1)
        flag = true;
    }
    N++;
  }

  printf("\t\tDone.\n");
  /* ---------------------------------------------------------------------- */


  /* ファイル書き出し */
  /* ---------------------------------------------------------------------- */
  printf("Output...\t");
  Cuda::outputZeroStep("0step.csv", false, cond, zero_step);
  Cuda::outputNStep("Nstep.csv", false, cond, n_step, next_id);
  printf("Done.\n");
  /* ---------------------------------------------------------------------- */

  /* 終了処理 */
  /* ---------------------------------------------------------------------- */
  printf("Finish...\t");

  /* メモリの開放 */
  // ホスト側
  delete state;
  delete input;
  delete zero_step;
  delete n_step;
  delete next_id;
  delete grid;
  delete cop;
  delete physics;
  // デバイス側
  cudaFree(dev_state);
  cudaFree(dev_input);
  cudaFree(dev_next_id);
  cudaFree(dev_zero_step);
  cudaFree(dev_n_step);
  cudaFree(dev_grid);
  cudaFree(dev_cop);
  cudaFree(dev_physics);

  printf("Done.\n");
  /* ---------------------------------------------------------------------- */

  return 0;
}