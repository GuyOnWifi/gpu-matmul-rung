// Config sweep for impls/warptile_tpl.cuh.
//
// One benchmark body, one type axis over config indices. The only generated
// artifact is autotune_cfgs.inc, which is a table of integers:
//     python3 gen_autotune.py 4096 > autotune_cfgs.inc
//
// Generation is needed because nvbench's type axes take a Cartesian product
// and ~75% of it is invalid -- and invalid configs don't merely skip at
// runtime, they fail to compile (a zero PACK_A is a division by zero inside
// the kernel body).

// #include "08_warptiling_tpl.cuh"
// #include "09_prefetch_tpl.cuh"
#include "10_prefetch_reg_tpl.cuh"
#include "cublas_v2.h"
#include "matmul.cuh"
#include <cmath>
#include <cstdio>
#include <nvbench/nvbench.cuh>
#include <string>
#include <vector>

const int M = 4096;
const int N = 4096;
const int K = 4096;

struct Cfg {
  int BM, BN, BK, TM, TN, WM, WN, WI;
};

#include "autotune_cfgs.inc"

static std::string cfg_name(const Cfg &c) {
  char buf[64];
  std::snprintf(buf, sizeof buf, "%dx%dx%d_t%dx%d_w%dx%d_i%d", c.BM, c.BN, c.BK,
                c.TM, c.TN, c.WM, c.WN, c.WI);
  return buf;
}

// Label the axis with the config rather than a bare index, so the results
// table is readable. One partial specialisation covers every index.
namespace nvbench {
template <int I> struct type_strings<nvbench::enum_type<I>> {
  static std::string input_string() { return cfg_name(CFGS[I]); }
  static std::string description() { return {}; }
};
} // namespace nvbench

// One allocation and one cuBLAS reference for the whole sweep.
struct Fixture {
  float *A, *B, *C;
  std::vector<float> ref;
  double rms;
  cublasHandle_t h;

  Fixture() {
    auto p = init_matrixes_alloc(M, N, K);
    A = p[0];
    B = p[1];
    C = p[2];
    cublasCreate(&h);
    matmulCublas(M, N, K, A, B, C, h);
    cudaDeviceSynchronize();
    ref.resize((size_t)M * N);
    cudaMemcpy(ref.data(), C, ref.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    double ss = 0;
    for (float v : ref)
      ss += (double)v * v;
    rms = std::sqrt(ss / ref.size());
  }
};

static Fixture &fixture() {
  static Fixture f;
  return f;
}

// max|diff| normalised by the RMS of the reference. A per-element relative
// error is useless here: with K=4096 the outputs pass through zero, so the
// denominator vanishes and every correct kernel looks broken.
static double nerr() {
  auto &f = fixture();
  static std::vector<float> got((size_t)M * N);
  cudaMemcpy(got.data(), f.C, got.size() * sizeof(float),
             cudaMemcpyDeviceToHost);
  double worst = 0;
  for (size_t i = 0; i < got.size(); i++)
    worst = std::max(worst, std::fabs((double)got[i] - (double)f.ref[i]));
  return worst / f.rms;
}

static void calc_flops(nvbench::state &state) {
  const double t =
      state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  auto &s = state.add_summary("mmm/flop_rate");
  s.set_string("name", "FLOP/s");
  s.set_string("hint", "item_rate");
  s.set_float64("value", 2.0 * M * N * K / t);
}

template <int I>
void sweep(nvbench::state &state, nvbench::type_list<nvbench::enum_type<I>>) {
  constexpr Cfg c = CFGS[I];
  auto &f = fixture();

  // Verify before timing: a config can be fast and wrong.
  cudaMemset(f.C, 0, (size_t)M * N * sizeof(float));
  launchPrefetchRegTpl<c.BM, c.BN, c.BK, c.TM, c.TN, c.WM, c.WN, c.WI>(
      M, N, K, f.A, f.B, f.C);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    cudaGetLastError();
    state.skip("launch failed");
    return;
  }
  if (nerr() >= 1e-4) {
    state.skip("wrong result");
    return;
  }

  state.exec([&f](nvbench::launch &) {
    launchPrefetchRegTpl<c.BM, c.BN, c.BK, c.TM, c.TN, c.WM, c.WN, c.WI>(
        M, N, K, f.A, f.B, f.C);
  });
  calc_flops(state);
}

NVBENCH_BENCH_TYPES(sweep,
                    NVBENCH_TYPE_AXES(nvbench::enum_type_list<CFG_INDICES>))
    .set_type_axes_names({"cfg"});

void ref_cublas(nvbench::state &state) {
  auto &f = fixture();
  state.exec(
      [&f](nvbench::launch &) { matmulCublas(M, N, K, f.A, f.B, f.C, f.h); });
  calc_flops(state);
}
NVBENCH_BENCH(ref_cublas);
