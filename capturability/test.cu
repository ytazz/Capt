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
  /* パラメータの読み込み */
  Model model("nao.xml");
  Param param("analysis.xml");

  /* グリッド */
  Grid grid(param);
  const int num_state = grid.getNumState();
  const int num_input = grid.getNumInput();
  const int num_nstep = num_state * num_input;

  /* グリッド */
  // ホスト側
  CudaState *cstate = new CudaState[num_state];
  CudaInput *cinput = new CudaInput[num_input];
  int *cnstep = new int[num_nstep];
  CudaGrid *cgrid = new CudaGrid;
  initNstep(grid, cnstep);
  copyState(grid, cstate);
  copyInput(grid, cinput);
  copyGrid(grid, cgrid);
  // デバイス側
  CudaState *dev_cstate;
  CudaInput *dev_cinput;
  int *dev_cnstep;
  CudaGrid *dev_cgrid;
  HANDLE_ERROR(cudaMalloc((void **)&dev_cstate, num_state * sizeof(CudaState)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_cinput, num_input * sizeof(CudaInput)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_cnstep, num_nstep * sizeof(int)));
  HANDLE_ERROR(cudaMalloc((void **)&dev_cgrid, sizeof(CudaGrid)));
  HANDLE_ERROR(cudaMemcpy(dev_cstate, cstate, num_state * sizeof(CudaState),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_cinput, cinput, num_input * sizeof(CudaInput),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(cudaMemcpy(dev_cnstep, cnstep, num_nstep * sizeof(int),
                          cudaMemcpyHostToDevice));
  HANDLE_ERROR(
      cudaMemcpy(dev_cgrid, cgrid, sizeof(CudaGrid), cudaMemcpyHostToDevice));

  exeZeroStep<<<BPG, TPB>>>(dev_cstate, dev_cinput, dev_cnstep, dev_cgrid);

  HANDLE_ERROR(cudaMemcpy(cnstep, dev_cnstep, num_nstep * sizeof(int),
                          cudaMemcpyDeviceToHost));

  for (int i = 0; i < 10; i++) {
    printf("id: %d,\t nstep: %d\n", i, cnstep[i]);
  }

  /* メモリの開放 */
  // ホスト側
  delete cstate;
  delete cinput;
  delete cgrid;
  // デバイス側
  cudaFree(dev_cstate);
  cudaFree(dev_cinput);
  cudaFree(dev_cnstep);
  cudaFree(dev_cgrid);

  return 0;
}