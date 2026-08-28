#include "cublas_v2.h"
#include <array>
#pragma once

std::array<float *, 3> init_matrixes(int M, int N, int K, float *A, float *B,
                                     const float *C);

void matmulCublas(int M, int N, int K, const float *A, const float *B, float *C,
                  cublasHandle_t handle);

std::array<float *, 3> init_matrixes_alloc(int M, int N, int K);

void matmulNaive(int M, int N, int K, const float *A, const float *B, float *C);

void matmulGMEMCoalesced(int M, int N, int K, const float *A, const float *B,
                         float *C);

void matmulSMEMCoalesced(int M, int N, int K, const float *A, const float *B,
                         float *C);

void matmulBlockTiling1D(int M, int N, int K, const float *A, const float *B,
                         float *C);

void matmulBlockTiling2D(int M, int N, int K, const float *A, const float *B,
                         float *C);

void matmulBlockTiling2DTransposed(int M, int N, int K, const float *A,
                                   const float *B, float *C);

void matmulVectorizedGMEM(int M, int N, int K, const float *A, const float *B,
                          float *C);

void matmulWarptiling(int M, int N, int K, const float *A, const float *B,
                      float *C);
