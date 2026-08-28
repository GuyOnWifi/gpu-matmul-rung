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

/*
template <int BM, int BN, int BK, int TM, int TN>
void Autotuned(
    nvbench::state &state,
    nvbench::type_list<nvbench::enum_type<BM>, nvbench::enum_type<BN>,
                       nvbench::enum_type<BK>, nvbench::enum_type<TM>,
                       nvbench::enum_type<TN>>) {

  const int num_threads = BM * BN / TM / TN;
  constexpr int a_stride = 4 * num_threads / BK; // A rows per pass
  constexpr int b_stride = 4 * num_threads / BN; // B rows per pass
  constexpr bool valid =
      BK % 4 == 0 && BN % 4 == 0 && BM % TM == 0 && BN % TN == 0 &&
      num_threads >= 32 && num_threads <= 1024 && a_stride >= 1 &&
      BM % a_stride == 0 && b_stride >= 1 && BK % b_stride == 0 &&
      (BM * BK + BK * BN) * sizeof(float) <= 48 * 1024;

  if constexpr (!valid) {
    state.skip("invalid config");
    return;
  }

  auto ptrs = init_matrixes_alloc(M, N, K);

  state.exec([ptrs](nvbench::launch &launch) {
    matmulAutotuned<BM, BN, BK, TM, TN>(M, N, K, ptrs[0], ptrs[1], ptrs[2]);
  });

  calc_flops(state);
}

using MBlockList = nvbench::enum_type_list<64, 128>;
using NBlockList = nvbench::enum_type_list<64, 128>;
using KBlockList = nvbench::enum_type_list<4, 8, 16, 32>;
using MThreadList = nvbench::enum_type_list<4, 8>;
using NThreadList = nvbench::enum_type_list<4, 8>;

NVBENCH_BENCH_TYPES(Autotuned,
                    NVBENCH_TYPE_AXES(MBlockList, NBlockList, KBlockList,
                                      MThreadList, NThreadList))
    .set_type_axes_names({"BM", "BN", "BK", "TM", "TN"});
;
*/
