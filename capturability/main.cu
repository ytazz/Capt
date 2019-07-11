#include "analysis_gpu.cuh"

const int BPG = 1024; // Blocks  Per Grid  (max: 65535)
const int TPB = 1024; // Threads Per Block (max: 1024)

int main(void) {
  /* パラメータの読み込み */
  Model model("nao.xml");
  Param param("analysis.xml");

  /* グリッド */
  // ホスト側
  Grid *grid;
  grid = new Grid(param);
  // デバイス側
  Grid *dev_grid;
  HANDLE_ERROR(cudaMalloc((void **)&dev_grid, sizeof(Grid)));
  // ホスト側からデバイス側にコピー
  HANDLE_ERROR(
      cudaMemcpy(dev_grid, grid, sizeof(Grid), cudaMemcpyHostToDevice));

  // グリッド数の確認
  const int num_data = grid->getNumState() * grid->getNumInput();
  std::cout << "num_state = " << grid->getNumState() << '\n';
  std::cout << "num_input = " << grid->getNumInput() << '\n';

  /* 解析結果保存用変数 */
  // ホスト側
  Capturability *capturability;
  capturability = new Capturability(model, param);
  // デバイス側
  Capturability *dev_capturability;
  HANDLE_ERROR(cudaMalloc((void **)&dev_capturability, sizeof(Capturability)));
  // ホスト側からデバイス側にコピー
  HANDLE_ERROR(cudaMemcpy(dev_capturability, capturability,
                          sizeof(Capturability), cudaMemcpyHostToDevice));

  /* 解析 */
  exeZero<<<BPG, TPB>>>(dev_capturability, dev_grid);

  /* 結果 */
  // デバイス側からホスト側にコピー
  HANDLE_ERROR(cudaMemcpy(capturability, dev_capturability,
                          sizeof(Capturability), cudaMemcpyDeviceToHost));

  /* メモリの開放 */
  // ホスト側
  delete grid;
  delete capturability;
  // デバイス側
  cudaFree(dev_grid);
  cudaFree(dev_capturability);

  return 0;
}