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

  /* グリッド */
  // ホスト側
  CudaState *cstate = new CudaState[num_state];
  CudaInput *cinput = new CudaInput[num_input];
  int *cnstep = new int[num_grid];
  int *next_state_id = new int[num_grid];
  CudaGrid *cgrid = new CudaGrid;
  setNstep(grid, cnstep);
  setState(grid, cstate);
  setInput(grid, cinput);
  init(next_state_id, num_grid);
  setGrid(grid, model, param, cgrid);
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

  FILE *fp_output = fopen("result.csv", "w");
  fprintf(fp_output, "%s,", "state_id");
  fprintf(fp_output, "%s,", "input_id");
  fprintf(fp_output, "%s,", "next_state_id");
  fprintf(fp_output, "%s,", "nstep");
  fprintf(fp_output, "\n");
  for (int state_id = 0; state_id < grid.getNumState(); state_id++) {
    for (int input_id = 0; input_id < grid.getNumInput(); input_id++) {
      int id = state_id * num_input + input_id;
      fprintf(fp_output, "%d,", state_id);
      fprintf(fp_output, "%d,", input_id);
      fprintf(fp_output, "%d,", next_state_id[id]);
      fprintf(fp_output, "%d,", cnstep[id]);
      fprintf(fp_output, "\n");
    }
  }
  fclose(fp_output);

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