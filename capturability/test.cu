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
  setNstep(grid, cnstep);
  setState(grid, cstate);
  setInput(grid, cinput);
  setGrid(grid, model, param, cgrid);
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

  /* 足形状 */
  const int num_foot = (int)model.getVec("foot", "foot_r").size();
  CudaVector2 *cfoot_r = new CudaVector2[num_foot];
  CudaVector2 *cfoot_l = new CudaVector2[num_foot];

  // ホスト側
  CudaVector2 *cfoot = new CudaVector2[2 * num_foot];
  setFoot(cfoot, cfoot_r, cfoot_l, num_foot);
  // デバイス側
  CudaVector2 *dev_cfoot;
  HANDLE_ERROR(
      cudaMalloc((void **)&dev_cfoot, 2 * num_foot * sizeof(CudaVector2)));
  HANDLE_ERROR(cudaMemcpy(dev_cfoot, cfoot, 2 * num_foot * sizeof(CudaVector2),
                          cudaMemcpyHostToDevice));

  // exeZeroStep<<<BPG, TPB>>>(dev_cstate, dev_cinput, dev_cnstep, dev_cfoot,
  //                           dev_cgrid);

  // HANDLE_ERROR(cudaMemcpy(cnstep, dev_cnstep, num_nstep * sizeof(int),
  //                         cudaMemcpyDeviceToHost));

  /* メモリの開放 */
  // ホスト側
  delete cstate;
  delete cinput;
  delete cnstep;
  delete cgrid;
  delete cfoot_r;
  delete cfoot_l;
  delete cfoot;
  // デバイス側
  cudaFree(dev_cstate);
  cudaFree(dev_cinput);
  cudaFree(dev_cnstep);
  cudaFree(dev_cgrid);
  cudaFree(dev_cfoot);

  return 0;
}