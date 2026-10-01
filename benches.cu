#include "autotune.cuh"
#include "cublas_v2.h"
#include "matmul.cuh"
#include <nvbench/create.cuh>
#include <nvbench/detail/type_list_impl.cuh>
#include <nvbench/enum_type_list.cuh>
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

void VectorizedGMEM(nvbench::state &state) {
  auto ptrs = init_matrixes_alloc(M, N, K);

  state.exec([ptrs](nvbench::launch &launch) {
    matmulVectorizedGMEM(M, N, K, ptrs[0], ptrs[1], ptrs[2]);
  });

  calc_flops(state);
}
NVBENCH_BENCH(VectorizedGMEM);

void Warptiling(nvbench::state &state) {
  auto ptrs = init_matrixes_alloc(M, N, K);

  state.exec([ptrs](nvbench::launch &launch) {
    matmulWarptiling(M, N, K, ptrs[0], ptrs[1], ptrs[2]);
  });

  calc_flops(state);
}
NVBENCH_BENCH(Warptiling);

__global__ void triadKernel(float *c, const float *a, const float *b, float s,
                            size_t n) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    c[i] = a[i] + s * b[i];
}

void triad(nvbench::state &state) {
  const size_t n = 64ull * 1024 * 1024; // 256 MB per array — far beyond L2
  const size_t bytes = n * sizeof(float);

  float *a, *b, *c;
  cudaMalloc(&a, bytes);
  cudaMalloc(&b, bytes);
  cudaMalloc(&c, bytes);
  cudaMemset(a, 0, bytes);
  cudaMemset(b, 0, bytes);

  // nvbench does the GB/s arithmetic from these.
  state.add_global_memory_reads<float>(2 * n);
  state.add_global_memory_writes<float>(n);

  state.exec([=](nvbench::launch &launch) {
    triadKernel<<<(n + 255) / 256, 256, 0, launch.get_stream()>>>(c, a, b, 2.0f,
                                                                  n);
  });

  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
}
NVBENCH_BENCH(triad);
