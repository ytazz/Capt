#include "cuda_analysis.cuh"
#include "grid.h"
#include "input.h"
#include "model.h"
#include "param.h"
#include "state.h"

const int BPG = 1024; // Blocks  Per Grid  (max: 65535)
const int TPB = 1024; // Threads Per Block (max: 1024)

using namespace CA;

int main(void) {
  /* 前処理 */
  /* ---------------------------------------------------------------------- */
  printf("Prepare...\t");

  /* パラメータの読み込み */
  Model model("nao.xml");
  Param param("analysis.xml");

  /* グリッド */
  Grid grid(param);
  const int num_state = grid.getNumState();
  const int num_input = grid.getNumInput();
  const int num_grid = num_state * num_input;

  /* 解析条件 */
  Condition cond;
  cond.model = &model;
  cond.param = &param;
  cond.grid = &grid;

  /* グリッド */
  // ホスト側
  CudaState *cstate = new CudaState[num_state];
  CudaInput *cinput = new CudaInput[num_input];
  int *cnstep = new int[num_grid];
  int *next_state_id = new int[num_grid];
  CudaGrid *cgrid = new CudaGrid;
  initState(cstate, next_state_id, cond);
  initInput(cinput, cond);
  initNstep(cnstep, cond);
  initGrid(cgrid, cond);
  // デバイス側
  CudaState *dev_cstate;
  CudaInput *dev_cinput;
  int *dev_cnstep;
  int *dev_next_state_id;
  CudaGrid *dev_cgrid;
  HANDLE_ERROR(cudaMalloc((void **)&dev_cstate, num_state * sizeof(CudaState)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_cinput, num_input * sizeof(CudaInput)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_cnstep, num_grid * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_next_state_id, num_grid * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_cgrid, sizeof(CudaGrid)));
  HANDLE_ERROR(cudaMemcpy(dev_cstate, cstate, num_state * sizeof(CudaState),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_cinput, cinput, num_input * sizeof(CudaInput),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_cnstep, cnstep, num_grid * sizeof(int),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_next_state_id, next_state_id,
                          num_grid * sizeof(int), cudaMemcpyHostToDevice));
  HANDLE_ERROR(
      cudaMemcpy(dev_cgrid, cgrid, sizeof(CudaGrid), cudaMemcpyHostToDevice));

  /* CoP List */
  // ホスト側
  CudaVector2 *cop = new CudaVector2[num_state];
  initCop(cop, cond);
  // デバイス側
  CudaVector2 *dev_cop;
  HANDLE_ERROR(cudaMalloc((void **)&dev_cop, num_state * sizeof(CudaVector2)));
  HANDLE_ERROR(cudaMemcpy(dev_cop, cop, num_state * sizeof(CudaVector2),
                          cudaMemcpyHostToDevice));

  printf("Done.\n");
  /* ------------------------------------------------------------------------ */

  /* 解析実行 */
  /* ---------------------------------------------------------------------- */
  printf("Execute...\t");

  exeZeroStep(grid, model, cnstep, next_state_id);
  // exeNStep<<<BPG, TPB>>>(dev_cstate, dev_cinput, dev_cnstep, dev_cfoot,
  //                           dev_cgrid);

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
  // デバイス側
  cudaFree(dev_cstate);
  cudaFree(dev_cinput);
  cudaFree(dev_next_state_id);
  cudaFree(dev_cnstep);
  cudaFree(dev_cgrid);

  printf("Done.\n");
  /* ---------------------------------------------------------------------- */

  return 0;
}