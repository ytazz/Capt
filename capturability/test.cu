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
  // ホスト側
  Grid *grid;
  grid = new Grid(param);
  CudaGrid *cgrid;
  cgrid = new CudaGrid();
  copyGrid(grid, cgrid);
  // デバイス側
  CudaGrid *dev_cgrid;
  HANDLE_ERROR(cudaMalloc((void **)&dev_cgrid, sizeof(CudaGrid)));
  HANDLE_ERROR(
      cudaMemcpy(dev_cgrid, cgrid, sizeof(CudaGrid), cudaMemcpyHostToDevice));

  exeZeroStep<<<BPG, TPB>>>(dev_cgrid);
  HANDLE_ERROR(
      cudaMemcpy(cgrid, dev_cgrid, sizeof(CudaGrid), cudaMemcpyDeviceToHost));

  for (int i = 0; i < grid->getNumState() * grid->getNumInput(); i++) {
    if (cgrid->nstep[i] == 0)
      printf("id: %d,\t nstep: %d\n", i, cgrid->nstep[i]);
  }
  std::cout << cgrid->num_icp_r << '\n';

  /* メモリの開放 */
  // ホスト側
  delete grid;
  delete cgrid;
  // デバイス側
  cudaFree(dev_cgrid);

  return 0;
}