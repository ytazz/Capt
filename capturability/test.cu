#include "cuda_analysis.cuh"

const int BPG = 1024; // Blocks  Per Grid  (max: 65535)
const int TPB = 1024; // Threads Per Block (max: 1024)

int main(void) {
  /* 前処理 */
  /* ---------------------------------------------------------------------- */
  printf("Prepare...\t");

  /* パラメータの読み込み */
  CA::Model model("nao.xml");
  CA::Param param("analysis.xml");

  /* グリッド */
  CA::Grid grid(param);
  const int num_state = grid.getNumState();
  const int num_input = grid.getNumInput();
  const int num_grid = num_state * num_input;

  /* 解析条件 */
  Cuda::Condition cond;
  cond.model = &model;
  cond.param = &param;
  cond.grid = &grid;

  /* グリッド */
  // ホスト側
  Cuda::State *cstate = new Cuda::State[num_state];
  Cuda::Input *cinput = new Cuda::Input[num_input];
  int *cnstep = new int[num_grid];
  int *next_state_id = new int[num_grid];
  Cuda::Grid *cgrid = new Cuda::Grid;
  initState(cstate, next_state_id, cond);
  initInput(cinput, cond);
  initNstep(cnstep, cond);
  initGrid(cgrid, cond);
  // デバイス側
  Cuda::State *dev_cstate;
  Cuda::Input *dev_cinput;
  int *dev_cnstep;
  int *dev_next_state_id;
  Cuda::Grid *dev_cgrid;
  HANDLE_ERROR(
      cudaMalloc((void **)&dev_cstate, num_state * sizeof(Cuda::State)));
  HANDLE_ERROR(
      cudaMalloc((void **)&dev_cinput, num_input * sizeof(Cuda::Input)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_cnstep, num_grid * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_next_state_id, num_grid * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_cgrid, sizeof(Cuda::Grid)));
  HANDLE_ERROR(cudaMemcpy(dev_cstate, cstate, num_state * sizeof(Cuda::State),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_cinput, cinput, num_input * sizeof(Cuda::Input),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_cnstep, cnstep, num_grid * sizeof(int),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_next_state_id, next_state_id,
                          num_grid * sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(
      cudaMemcpy(dev_cgrid, cgrid, sizeof(Cuda::Grid), cudaMemcpyHostToDevice));

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

  /* 解析実行 */
  /* ---------------------------------------------------------------------- */
  printf("Execute...\t");

  Cuda::exeZeroStep(grid, model, cnstep, next_state_id);

  Cuda::exeNStep<<<BPG, TPB>>>(dev_cstate, dev_cinput, dev_cnstep,
                               dev_next_state_id, dev_cgrid, dev_cop,
                               dev_physics);

  printf("Done.\n");
  /* ---------------------------------------------------------------------- */

  // HANDLE_ERROR(cudaMemcpy(cnstep, dev_cnstep, num_grid * sizeof(int),
  //                         cudaMemcpyDeviceToHost));

  /* ファイル書き出し */
  /* ---------------------------------------------------------------------- */
  printf("Output...\t");
  output("result.csv", cond, cnstep, next_state_id);
  printf("Done.\n");
  /* ---------------------------------------------------------------------- */

  /* 終了処理 */
  /* ---------------------------------------------------------------------- */
  printf("Finish...\t");

  /* メモリの開放 */
  // ホスト側
  delete cstate;
  delete cinput;
  delete cnstep;
  delete next_state_id;
  delete cgrid;
  delete cop;
  delete physics;
  // デバイス側
  cudaFree(dev_cstate);
  cudaFree(dev_cinput);
  cudaFree(dev_next_state_id);
  cudaFree(dev_cnstep);
  cudaFree(dev_cgrid);
  cudaFree(dev_cop);
  cudaFree(dev_physics);

  printf("Done.\n");
  /* ---------------------------------------------------------------------- */

  return 0;
}