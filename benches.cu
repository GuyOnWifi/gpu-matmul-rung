#include "cublas_v2.h"
#include "matmul.cuh"
#include <nvbench/nvbench.cuh>

const int M = 4096;
const int N = 4096;
const int K = 4096;

void calc_flops(nvbench::state &state) {
  const double t =
      state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  auto &s = state.add_summary("mmm/flop_rate");
  s.set_string("name", "FLOP/s");
  s.set_string("hint", "item_rate"); // SI formatting
  s.set_float64("value", 2.0 * M * N * K / t);
}

void cublas(nvbench::state &state) {
  auto ptrs = init_matrixes_alloc(M, N, K);
  cublasHandle_t handle;
  cublasCreate(&handle);

  state.exec([ptrs, handle](nvbench::launch &launch) {
    matmulCublas(M, N, K, ptrs[0], ptrs[1], ptrs[2], handle);
  });

  calc_flops(state);
}
NVBENCH_BENCH(cublas);

void naive(nvbench::state &state) {
  auto ptrs = init_matrixes_alloc(M, N, K);

  state.exec([ptrs](nvbench::launch &launch) {
    matmulNaive(M, N, K, ptrs[0], ptrs[1], ptrs[2]);
  });

  calc_flops(state);
}
NVBENCH_BENCH(naive);

void GMEMCoalesced(nvbench::state &state) {
  auto ptrs = init_matrixes_alloc(M, N, K);

  state.exec([ptrs](nvbench::launch &launch) {
    matmulGMEMCoalesced(M, N, K, ptrs[0], ptrs[1], ptrs[2]);
  });

  calc_flops(state);
}
NVBENCH_BENCH(GMEMCoalesced);

void SMEMCoalesced(nvbench::state &state) {
  auto ptrs = init_matrixes_alloc(M, N, K);

  state.exec([ptrs](nvbench::launch &launch) {
    matmulSMEMCoalesced(M, N, K, ptrs[0], ptrs[1], ptrs[2]);
  });

  calc_flops(state);
}
NVBENCH_BENCH(SMEMCoalesced);

void BlockTiling1D(nvbench::state &state) {
  auto ptrs = init_matrixes_alloc(M, N, K);

  state.exec([ptrs](nvbench::launch &launch) {
    matmulBlockTiling1D(M, N, K, ptrs[0], ptrs[1], ptrs[2]);
  });

  calc_flops(state);
}
NVBENCH_BENCH(BlockTiling1D);

void BlockTiling2D(nvbench::state &state) {
  auto ptrs = init_matrixes_alloc(M, N, K);

  state.exec([ptrs](nvbench::launch &launch) {
    matmulBlockTiling2D(M, N, K, ptrs[0], ptrs[1], ptrs[2]);
  });

  calc_flops(state);
}
NVBENCH_BENCH(BlockTiling2D);

void BlockTiling2DTransposed(nvbench::state &state) {
  auto ptrs = init_matrixes_alloc(M, N, K);

  state.exec([ptrs](nvbench::launch &launch) {
    matmulBlockTiling2DTransposed(M, N, K, ptrs[0], ptrs[1], ptrs[2]);
  });

  calc_flops(state);
}
NVBENCH_BENCH(BlockTiling2DTransposed);
