/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

#include <immintrin.h>
#include <avx2intrin.h>
#include <stdlib.h>

const char *dgemm_desc = "Simple blocked dgemm.";

double *B_SUB_ALGN;

double *B_L2_CACHED = NULL;
double *B_L1_CACHED = NULL;
double *A_L3_CACHED = NULL;

#if !defined(L1_M)
#define L1_M 32
#endif

#if !defined(L1_N)
#define L1_N 64
#endif

#if !defined(L1_K)
#define L1_K 32
#endif

#if !defined(L2_M)
#define L2_M 64
#endif

#if !defined(L2_N)
#define L2_N 64
#endif

#if !defined(L2_K)
#define L2_K 64
#endif

#if !defined(L3_M)
#define L3_M 128
#endif

#if !defined(L3_N)
#define L3_N 64
#endif

#if !defined(L3_K)
#define L3_K 128
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

static inline void populate_B_CACHED(int ldb, double *B, double *B_Cached, int M, int N)
{
  for (int i = 0; i < M; i++)
  {
    int I_N = i * N;
    int I_LDB = i * ldb;
    for (int j = 0; j < N; j++)
    {
      B_Cached[I_N + j] = B[I_LDB + j];
    }
  }
}




static inline void do_block_SIMD8x8(int lda, int ldb, int ldc, double *restrict A, double *restrict B, double *restrict C)
{

  register __m512d c00_c01_c02_c03 = _mm512_loadu_pd(C);
  register __m512d c10_c11_c12_c13 = _mm512_loadu_pd(C + ldc);
  register __m512d c20_c21_c22_c23 = _mm512_loadu_pd(C + 2 * ldc);
  register __m512d c30_c31_c32_c33 = _mm512_loadu_pd(C + 3 * ldc);
  register __m512d c40_c41_c42_c43 = _mm512_loadu_pd(C + 4 * ldc);
  register __m512d c50_c51_c52_c53 = _mm512_loadu_pd(C + 5 * ldc);
  register __m512d c60_c61_c62_c63 = _mm512_loadu_pd(C + 6 * ldc);
  register __m512d c70_c71_c72_c73 = _mm512_loadu_pd(C + 7 * ldc);


  for (int kk = 0; kk < 8; ++kk)
  {
    register __m512d a0x = _mm512_broadcast_sd(A + kk);
    register __m512d a1x = _mm512_broadcast_sd(A + kk + lda);
    register __m512d a2x = _mm512_broadcast_sd(A + kk + 2 * lda);
    register __m512d a3x = _mm512_broadcast_sd(A + kk + 3 * lda);
    register __m512d a4x = _mm512_broadcast_sd(A + kk + 4 * lda);
    register __m512d a5x = _mm512_broadcast_sd(A + kk + 5 * lda);
    register __m512d a6x = _mm512_broadcast_sd(A + kk + 6 * lda);
    register __m512d a7x = _mm512_broadcast_sd(A + kk + 7 * lda);

    register __m512d b = _mm512_loadu_pd(B + kk * ldb);

    c00_c01_c02_c03 = _mm512_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm512_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm512_fmadd_pd(a2x, b, c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm512_fmadd_pd(a3x, b, c30_c31_c32_c33);
    c40_c41_c42_c43 = _mm512_fmadd_pd(a4x, b, c40_c41_c42_c43);
    c50_c51_c52_c53 = _mm512_fmadd_pd(a5x, b, c50_c51_c52_c53);
    c60_c61_c62_c63 = _mm512_fmadd_pd(a6x, b, c60_c61_c62_c63);
    c70_c71_c72_c73 = _mm512_fmadd_pd(a7x, b, c70_c71_c72_c73);
  }
  _mm512_storeu_pd(C, c00_c01_c02_c03);
  _mm512_storeu_pd(C + ldc, c10_c11_c12_c13);
  _mm512_storeu_pd(C + 2 * ldc, c20_c21_c22_c23);
  _mm512_storeu_pd(C + 3 * ldc, c30_c31_c32_c33);
  _mm512_storeu_pd(C + 4 * ldc, c40_c41_c42_c43);
  _mm512_storeu_pd(C + 5 * ldc, c50_c51_c52_c53);
  _mm512_storeu_pd(C + 6 * ldc, c60_c61_c62_c63);
  _mm512_storeu_pd(C + 7 * ldc, c70_c71_c72_c73);
}












/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B using SIMD operations
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static inline void do_block_SIMD(int lda, int ldb, int ldc, int M, int N, int K, double *A, double *B, double *C)
{
  register __m256d c00_c01_c02_c03 = _mm256_loadu_pd(C);
  register __m256d c10_c11_c12_c13 = _mm256_loadu_pd(C + ldc);

  for (int kk = 0; kk < K; ++kk)
  {
    register __m256d a0x = _mm256_broadcast_sd(A + kk);
    register __m256d a1x = _mm256_broadcast_sd(A + kk + lda);

    register __m256d b = _mm256_loadu_pd(B + kk * ldb);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
  }

  _mm256_storeu_pd(C, c00_c01_c02_c03);
  _mm256_storeu_pd(C + ldc, c10_c11_c12_c13);
}

static inline void do_block_SIMD5x4(int lda, int ldb, int ldc, double *A, double *B, double *C)
{
  register int address = 0;
  register __m256d c00_c01_c02_c03 = _mm256_loadu_pd(C + address);
  address += ldc;
  register __m256d c10_c11_c12_c13 = _mm256_loadu_pd(C + address);
  address += ldc;
  register __m256d c20_c21_c22_c23 = _mm256_loadu_pd(C + address);
  address += ldc;
  register __m256d c30_c31_c32_c33 = _mm256_loadu_pd(C + address);
  address += ldc;
  register __m256d c40_c41_c42_c43 = _mm256_loadu_pd(C + address);

  for (int kk = 0; kk < 4; ++kk)
  {
    register __m256d a0x = _mm256_broadcast_sd(A + kk);
    register __m256d a1x = _mm256_broadcast_sd(A + kk + lda);
    register __m256d a2x = _mm256_broadcast_sd(A + kk + 2 * lda);
    register __m256d a3x = _mm256_broadcast_sd(A + kk + 3 * lda);
    register __m256d a4x = _mm256_broadcast_sd(A + kk + 4 * lda);

    register __m256d b = _mm256_loadu_pd(B + kk * ldb);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
    c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b, c40_c41_c42_c43);
  }

  _mm256_storeu_pd(C, c00_c01_c02_c03);
  _mm256_storeu_pd(C + ldc, c10_c11_c12_c13);
  _mm256_storeu_pd(C + 2 * ldc, c20_c21_c22_c23);
  _mm256_storeu_pd(C + 3 * ldc, c30_c31_c32_c33);
  _mm256_storeu_pd(C + 4 * ldc, c40_c41_c42_c43);
}

static inline void do_block_SIMD8x4(int lda, int ldb, int ldc, double *restrict A, double *restrict B, double *restrict C)
{
  A = __builtin_assume_aligned(A, 16);
  B = __builtin_assume_aligned(B, 16);
  C = __builtin_assume_aligned(C, 16);

  register __m256d c00_c01_c02_c03 = _mm256_loadu_pd(C);
  register __m256d c10_c11_c12_c13 = _mm256_loadu_pd(C + ldc);
  register __m256d c20_c21_c22_c23 = _mm256_loadu_pd(C + 2 * ldc);
  register __m256d c30_c31_c32_c33 = _mm256_loadu_pd(C + 3 * ldc);
  register __m256d c40_c41_c42_c43 = _mm256_loadu_pd(C + 4 * ldc);
  register __m256d c50_c51_c52_c53 = _mm256_loadu_pd(C + 5 * ldc);
  register __m256d c60_c61_c62_c63 = _mm256_loadu_pd(C + 6 * ldc);
  register __m256d c70_c71_c72_c73 = _mm256_loadu_pd(C + 7 * ldc);

#if 1
  for (int kk = 0; kk < 4; ++kk)
  {
    register __m256d a0x = _mm256_broadcast_sd(A + kk);
    register __m256d a1x = _mm256_broadcast_sd(A + kk + lda);
    register __m256d a2x = _mm256_broadcast_sd(A + kk + 2 * lda);
    register __m256d a3x = _mm256_broadcast_sd(A + kk + 3 * lda);
    register __m256d a4x = _mm256_broadcast_sd(A + kk + 4 * lda);
    register __m256d a5x = _mm256_broadcast_sd(A + kk + 5 * lda);
    register __m256d a6x = _mm256_broadcast_sd(A + kk + 6 * lda);
    register __m256d a7x = _mm256_broadcast_sd(A + kk + 7 * lda);

    register __m256d b = _mm256_loadu_pd(B + kk * ldb);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
    c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b, c40_c41_c42_c43);
    c50_c51_c52_c53 = _mm256_fmadd_pd(a5x, b, c50_c51_c52_c53);
    c60_c61_c62_c63 = _mm256_fmadd_pd(a6x, b, c60_c61_c62_c63);
    c70_c71_c72_c73 = _mm256_fmadd_pd(a7x, b, c70_c71_c72_c73);
  }
#else
  register __m256d a0x = _mm256_broadcast_sd(A);
  register __m256d a1x = _mm256_broadcast_sd(A + lda);
  register __m256d a2x = _mm256_broadcast_sd(A + 2 * lda);
  register __m256d a3x = _mm256_broadcast_sd(A + 3 * lda);
  register __m256d a4x = _mm256_broadcast_sd(A + 4 * lda);
  register __m256d a5x = _mm256_broadcast_sd(A + 5 * lda);
  register __m256d a6x = _mm256_broadcast_sd(A + 6 * lda);
  register __m256d a7x = _mm256_broadcast_sd(A + 7 * lda);
  register __m256d b = _mm256_loadu_pd(B);
  c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
  c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
  c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
  c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
  c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b, c40_c41_c42_c43);
  c50_c51_c52_c53 = _mm256_fmadd_pd(a5x, b, c50_c51_c52_c53);
  c60_c61_c62_c63 = _mm256_fmadd_pd(a6x, b, c60_c61_c62_c63);
  c70_c71_c72_c73 = _mm256_fmadd_pd(a7x, b, c70_c71_c72_c73);
  a0x = _mm256_broadcast_sd(A + 1);
  a1x = _mm256_broadcast_sd(A + 1 + lda);
  a2x = _mm256_broadcast_sd(A + 1 + 2 * lda);
  a3x = _mm256_broadcast_sd(A + 1 + 3 * lda);
  a4x = _mm256_broadcast_sd(A + 1 + 4 * lda);
  a5x = _mm256_broadcast_sd(A + 1 + 5 * lda);
  a6x = _mm256_broadcast_sd(A + 1 + 6 * lda);
  a7x = _mm256_broadcast_sd(A + 1 + 7 * lda);
  b = _mm256_loadu_pd(B + 1 * ldb);
  c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
  c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
  c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
  c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
  c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b, c40_c41_c42_c43);
  c50_c51_c52_c53 = _mm256_fmadd_pd(a5x, b, c50_c51_c52_c53);
  c60_c61_c62_c63 = _mm256_fmadd_pd(a6x, b, c60_c61_c62_c63);
  c70_c71_c72_c73 = _mm256_fmadd_pd(a7x, b, c70_c71_c72_c73);
  a0x = _mm256_broadcast_sd(A + 2);
  a1x = _mm256_broadcast_sd(A + 2 + lda);
  a2x = _mm256_broadcast_sd(A + 2 + 2 * lda);
  a3x = _mm256_broadcast_sd(A + 2 + 3 * lda);
  a4x = _mm256_broadcast_sd(A + 2 + 4 * lda);
  a5x = _mm256_broadcast_sd(A + 2 + 5 * lda);
  a6x = _mm256_broadcast_sd(A + 2 + 6 * lda);
  a7x = _mm256_broadcast_sd(A + 2 + 7 * lda);
  b = _mm256_loadu_pd(B + 2 * ldb);
  c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
  c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
  c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
  c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
  c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b, c40_c41_c42_c43);
  c50_c51_c52_c53 = _mm256_fmadd_pd(a5x, b, c50_c51_c52_c53);
  c60_c61_c62_c63 = _mm256_fmadd_pd(a6x, b, c60_c61_c62_c63);
  c70_c71_c72_c73 = _mm256_fmadd_pd(a7x, b, c70_c71_c72_c73);
  a0x = _mm256_broadcast_sd(A + 3);
  a1x = _mm256_broadcast_sd(A + 3 + lda);
  a2x = _mm256_broadcast_sd(A + 3 + 2 * lda);
  a3x = _mm256_broadcast_sd(A + 3 + 3 * lda);
  a4x = _mm256_broadcast_sd(A + 3 + 4 * lda);
  a5x = _mm256_broadcast_sd(A + 3 + 5 * lda);
  a6x = _mm256_broadcast_sd(A + 3 + 6 * lda);
  a7x = _mm256_broadcast_sd(A + 3 + 7 * lda);
  b = _mm256_loadu_pd(B + 3 * ldb);
  c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
  c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
  c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
  c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
  c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b, c40_c41_c42_c43);
  c50_c51_c52_c53 = _mm256_fmadd_pd(a5x, b, c50_c51_c52_c53);
  c60_c61_c62_c63 = _mm256_fmadd_pd(a6x, b, c60_c61_c62_c63);
  c70_c71_c72_c73 = _mm256_fmadd_pd(a7x, b, c70_c71_c72_c73);
#endif
  _mm256_storeu_pd(C, c00_c01_c02_c03);
  _mm256_storeu_pd(C + ldc, c10_c11_c12_c13);
  _mm256_storeu_pd(C + 2 * ldc, c20_c21_c22_c23);
  _mm256_storeu_pd(C + 3 * ldc, c30_c31_c32_c33);
  _mm256_storeu_pd(C + 4 * ldc, c40_c41_c42_c43);
  _mm256_storeu_pd(C + 5 * ldc, c50_c51_c52_c53);
  _mm256_storeu_pd(C + 6 * ldc, c60_c61_c62_c63);
  _mm256_storeu_pd(C + 7 * ldc, c70_c71_c72_c73);
}

static inline void do_block_SIMD8x4_(int lda, int ldb, int ldc, double *restrict A, double *restrict B, double *restrict C)
{
  A = __builtin_assume_aligned(A, 16);
  B = __builtin_assume_aligned(B, 16);
  C = __builtin_assume_aligned(C, 16);
  register double *address = C;
  register __m256d c00_c01_c02_c03 = _mm256_loadu_pd(C);
  address += ldc;
  register __m256d c10_c11_c12_c13 = _mm256_loadu_pd(address);
  address += ldc;
  register __m256d c20_c21_c22_c23 = _mm256_loadu_pd(address);
  address += ldc;
  register __m256d c30_c31_c32_c33 = _mm256_loadu_pd(address);
  address += ldc;
  register __m256d c40_c41_c42_c43 = _mm256_loadu_pd(address);
  address += ldc;
  register __m256d c50_c51_c52_c53 = _mm256_loadu_pd(address);
  address += ldc;
  register __m256d c60_c61_c62_c63 = _mm256_loadu_pd(address);
  address += ldc;
  register __m256d c70_c71_c72_c73 = _mm256_loadu_pd(address);

#if 1
  for (int kk = 0; kk < 4; ++kk)
  {
    address = A + kk;
    register __m256d a0x = _mm256_broadcast_sd(address);
    address += lda;
    register __m256d a1x = _mm256_broadcast_sd(address);
    address += lda;
    register __m256d a2x = _mm256_broadcast_sd(address);
    address += lda;
    register __m256d a3x = _mm256_broadcast_sd(address);
    address += lda;
    register __m256d a4x = _mm256_broadcast_sd(address);
    address += lda;
    register __m256d a5x = _mm256_broadcast_sd(address);
    address += lda;
    register __m256d a6x = _mm256_broadcast_sd(address);
    address += lda;
    register __m256d a7x = _mm256_broadcast_sd(address);

    register __m256d b = _mm256_loadu_pd(B + kk * ldb);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
    c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b, c40_c41_c42_c43);
    c50_c51_c52_c53 = _mm256_fmadd_pd(a5x, b, c50_c51_c52_c53);
    c60_c61_c62_c63 = _mm256_fmadd_pd(a6x, b, c60_c61_c62_c63);
    c70_c71_c72_c73 = _mm256_fmadd_pd(a7x, b, c70_c71_c72_c73);
  }
#else
  address = A;
  register __m256d a0x = _mm256_broadcast_sd(address);
  address += lda;
  register __m256d a1x = _mm256_broadcast_sd(address);
  address += lda;
  register __m256d a2x = _mm256_broadcast_sd(address);
  address += lda;
  register __m256d a3x = _mm256_broadcast_sd(address);
  address += lda;
  register __m256d a4x = _mm256_broadcast_sd(address);
  address += lda;
  register __m256d a5x = _mm256_broadcast_sd(address);
  address += lda;
  register __m256d a6x = _mm256_broadcast_sd(address);
  address += lda;
  register __m256d a7x = _mm256_broadcast_sd(address);
  address += lda;
  register __m256d b = _mm256_loadu_pd(B);

  c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
  c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
  c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
  c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
  c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b, c40_c41_c42_c43);
  c50_c51_c52_c53 = _mm256_fmadd_pd(a5x, b, c50_c51_c52_c53);
  c60_c61_c62_c63 = _mm256_fmadd_pd(a6x, b, c60_c61_c62_c63);
  c70_c71_c72_c73 = _mm256_fmadd_pd(a7x, b, c70_c71_c72_c73);
  address = A + 1;
  a0x = _mm256_broadcast_sd(address);
  address += lda;
  a1x = _mm256_broadcast_sd(address);
  address += lda;
  a2x = _mm256_broadcast_sd(address);
  address += lda;
  a3x = _mm256_broadcast_sd(address);
  address += lda;
  a4x = _mm256_broadcast_sd(address);
  address += lda;
  a5x = _mm256_broadcast_sd(address);
  address += lda;
  a6x = _mm256_broadcast_sd(address);
  address += lda;
  a7x = _mm256_broadcast_sd(address);

  b = _mm256_loadu_pd(B + 1 * ldb);
  c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
  c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
  c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
  c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
  c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b, c40_c41_c42_c43);
  c50_c51_c52_c53 = _mm256_fmadd_pd(a5x, b, c50_c51_c52_c53);
  c60_c61_c62_c63 = _mm256_fmadd_pd(a6x, b, c60_c61_c62_c63);
  c70_c71_c72_c73 = _mm256_fmadd_pd(a7x, b, c70_c71_c72_c73);

  address = A + 2;
  a0x = _mm256_broadcast_sd(address);
  address += lda;
  a1x = _mm256_broadcast_sd(address);
  address += lda;
  a2x = _mm256_broadcast_sd(address);
  address += lda;
  a2x = _mm256_broadcast_sd(address);
  address += lda;
  a3x = _mm256_broadcast_sd(address);
  address += lda;
  a4x = _mm256_broadcast_sd(address);
  address += lda;
  a5x = _mm256_broadcast_sd(address);
  address += lda;
  a6x = _mm256_broadcast_sd(address);
  address += lda;
  a7x = _mm256_broadcast_sd(address);

  b = _mm256_loadu_pd(B + 2 * ldb);
  c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
  c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
  c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
  c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
  c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b, c40_c41_c42_c43);
  c50_c51_c52_c53 = _mm256_fmadd_pd(a5x, b, c50_c51_c52_c53);
  c60_c61_c62_c63 = _mm256_fmadd_pd(a6x, b, c60_c61_c62_c63);
  c70_c71_c72_c73 = _mm256_fmadd_pd(a7x, b, c70_c71_c72_c73);

  address = A + 3;
  a0x = _mm256_broadcast_sd(address);
  address += lda;
  a1x = _mm256_broadcast_sd(address);
  address += lda;
  a2x = _mm256_broadcast_sd(address);
  address += lda;
  a3x = _mm256_broadcast_sd(address);
  address += lda;
  a4x = _mm256_broadcast_sd(address);
  address += lda;
  a5x = _mm256_broadcast_sd(address);
  address += lda;
  a6x = _mm256_broadcast_sd(address);
  address += lda;
  a7x = _mm256_broadcast_sd(address);
  b = _mm256_loadu_pd(B + 3 * ldb);
  c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
  c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
  c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
  c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
  c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b, c40_c41_c42_c43);
  c50_c51_c52_c53 = _mm256_fmadd_pd(a5x, b, c50_c51_c52_c53);
  c60_c61_c62_c63 = _mm256_fmadd_pd(a6x, b, c60_c61_c62_c63);
  c70_c71_c72_c73 = _mm256_fmadd_pd(a7x, b, c70_c71_c72_c73);
#endif

  address = C;
  _mm256_storeu_pd(C, c00_c01_c02_c03);
  address += ldc;
  _mm256_storeu_pd(address, c10_c11_c12_c13);
  address += ldc;
  _mm256_storeu_pd(address, c20_c21_c22_c23);
  address += ldc;
  _mm256_storeu_pd(address, c30_c31_c32_c33);
  address += ldc;
  _mm256_storeu_pd(address, c40_c41_c42_c43);
  address += ldc;
  _mm256_storeu_pd(address, c50_c51_c52_c53);
  address += ldc;
  _mm256_storeu_pd(address, c60_c61_c62_c63);
  address += ldc;
  _mm256_storeu_pd(address, c70_c71_c72_c73);
}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static inline void do_block_naive(int lda, int ldb, int ldc, int M, int N, int K, double *A, double *B, double *C)
{
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */
    for (int j = 0; j < N; ++j)
    {
      /* Compute C(i,j) */
      double cij = C[i * ldc + j];
      for (int k = 0; k < K; ++k)
#ifdef TRANSPOSE
        cij += A[i * lda + k] * B[j * lda + k];
#else
        cij += A[i * lda + k] * B[k * ldb + j];
#endif
      C[i * ldc + j] = cij;
    }
}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static inline void do_block(int lda, int ldb, int ldc, int M, int N, int K, double *A, double *B, double *C)
{
  /* For each row i of A */
  for (int i = 0; i < M; i += 2)
    /* For each column j of B */
    for (int k = 0; k < K; k += 4)
    {
      register int I = A + i * lda + k;
      for (int j = 0; j < N; j += 4)
      {
        /* Compute C(i,j) */
        int M_ = min(2, M - i);
        int N_ = min(4, N - j);
        int K_ = min(4, K - k);
        if (M_ == 2 && N_ == 4)
        {
          do_block_SIMD(lda, ldb, ldc, M_, N_, K_, I, B + k * ldb + j, C + i * ldc + j);
        }
        else
        {
          do_block_naive(lda, ldb, ldc, M_, N_, K_, A + i * lda + k, B + k * ldb + j, C + i * ldc + j);
        }
      }
    }
}

static inline void do_block5x4(int lda, int ldb, int ldc, int M, int N, int K, double *A, double *B, double *C)
{
  /* For each row i of A */
  int Malgn = (M / 5) * 5;
  int Nalgn = (N / 4) * 4;
  int Kalgn = (K / 4) * 4;
  //printf("%d %d\n", Kalgn, K);
  for (int i = 0; i < Malgn; i += 5)
  {
    /* For each column j of B */
    for (int j = 0; j < Nalgn; j += 4)
    {
      /* Compute C(i,j) */
      for (int k = 0; k < Kalgn; k += 4)
      {
        do_block_SIMD5x4(lda, ldb, ldc, A + i * lda + k, B + k * ldb + j, C + i * ldc + j);
      }
    }
  }
#if 0
  // // Now block 1 is
   do_block_naive(lda, Malgn, Nalgn, K-Kalgn, A+Malgn, B+Kalgn*lda, C);
   do_block_naive(lda, Malgn, N-Nalgn, Kalgn, A, B+Nalgn, C+Nalgn);
   do_block_naive(lda, Malgn, N-Nalgn, K-Kalgn, A+Malgn, B+lda*Kalgn+Nalgn, C+Nalgn);
   do_block_naive(lda, M-Malgn, Nalgn, Kalgn, A+lda*Malgn, B, C+lda*Malgn);
   do_block_naive(lda, M-Malgn, Nalgn, K-Kalgn, A+lda*Malgn+Kalgn, B+lda*Kalgn, C+lda*Malgn);
   do_block_naive(lda, M-Malgn, N-Nalgn, Kalgn, A+lda*Malgn, B+Nalgn, C+lda*Malgn+Nalgn);
   do_block_naive(lda, M-Malgn, N-Nalgn, K-Kalgn, A+lda*Malgn+Kalgn, B+lda*Kalgn+Nalgn, C+lda*Malgn+Nalgn);

#else

  for (int i = 0; i < Malgn; ++i)
  {
    for (int j = 0; j < Nalgn; ++j)
    {
      double cij = C[i * ldc + j];
      for (int k = Kalgn; k < K; ++k)
      {
        cij += A[i * lda + k] * B[k * ldb + j];
        C[i * ldc + j] = cij;
      }
    }
  }

  for (int i = 0; i < Malgn; ++i)
  {
    for (int j = Nalgn; j < N; ++j)
    {
      double cij = C[i * ldc + j];
      for (int k = 0; k < Kalgn; ++k)
      {
        cij += A[i * lda + k] * B[k * ldb + j];
        C[i * ldc + j] = cij;
      }
    }
  }

  for (int i = 0; i < Malgn; ++i)
  {
    for (int j = Nalgn; j < N; ++j)
    {
      double cij = C[i * ldc + j];
      for (int k = Kalgn; k < K; ++k)
      {
        cij += A[i * lda + k] * B[k * ldb + j];
        C[i * ldc + j] = cij;
      }
    }
  }

  for (int i = Malgn; i < M; ++i)
  {
    for (int j = 0; j < Nalgn; ++j)
    {
      double cij = C[i * ldc + j];
      for (int k = 0; k < Kalgn; ++k)
      {
        cij += A[i * lda + k] * B[k * ldb + j];
        C[i * ldc + j] = cij;
      }
    }
  }

  for (int i = Malgn; i < M; ++i)
  {
    for (int j = 0; j < Nalgn; ++j)
    {
      double cij = C[i * ldc + j];
      for (int k = Kalgn; k < K; ++k)
      {
#ifdef TRANSPOSE
        cij += A[i * lda + k] * B[j * lda + k];
#else
        cij += A[i * lda + k] * B[k * ldb + j];
#endif
        C[i * ldc + j] = cij;
      }
    }
  }

  for (int i = Malgn; i < M; ++i)
  {
    for (int j = Nalgn; j < N; ++j)
    {
      double cij = C[i * ldc + j];
      for (int k = 0; k < Kalgn; ++k)
      {
#ifdef TRANSPOSE
        cij += A[i * lda + k] * B[j * lda + k];
#else
        cij += A[i * lda + k] * B[k * ldb + j];
#endif
        C[i * ldc + j] = cij;
      }
    }
  }

  for (int i = Malgn; i < M; ++i)
  {
    for (int j = Nalgn; j < N; ++j)
    {
      double cij = C[i * ldc + j];
      for (int k = Kalgn; k < K; ++k)
      {
#ifdef TRANSPOSE
        cij += A[i * lda + k] * B[j * lda + k];
#else
        cij += A[i * lda + k] * B[k * ldb + j];
#endif
        C[i * ldc + j] = cij;
      }
    }
  }
#endif
}

static inline void do_block8x8(int lda, int ldb, int ldc, int M, int N, int K, double *A, double *B, double *C)
{
  /* For each row i of A */
  for (int i = 0; i < M; i += 8)
  {
    int I_LDA = i * lda;
    int I_LDC = i * ldc;
    for (int k = 0; k < K; k += 8)
    {
      int I_LDA_K = I_LDA + k;
      int K_LDB = k * ldb;
      for (int j = 0; j < N; j += 8)
      {
        /* For each column j of B */
        /* Compute C(i,j) */
        do_block_SIMD8x8(lda, ldb, ldc, A + I_LDA_K, B + K_LDB + j, C + I_LDC + j);
      }
    }
  }
}

static inline void do_blockL1(int lda, int ldb, int ldc, int M, int N, int K, double *A, double *B, double *C)
{
  for (int k = 0; k < K; k += L1_K)
  {
    int K_LDB = k * ldb;
    for (int j = 0; j < N; j += L1_N)
    {
      int K_LDB_J = K_LDB + j;
#ifdef ENABLE_L1_CACHING
      populate_B_CACHED(ldb, B + K_LDB_J, B_L1_CACHED, L1_K, L1_N);
#endif
      for (int i = 0; i < M; i += L1_M)
      {
#ifdef ENABLE_L1_CACHING
        do_block8x8(lda, L1_N, ldc, L1_M, L1_N, L1_K, A + i * lda + k, B_L1_CACHED, C + i * ldc + j);
#else
        do_block8x8(lda, ldb, ldc, L1_M, L1_N, L1_K, A + i * lda + k, B + K_LDB_J, C + i * ldc + j);
#endif
      }
    }
  }
}

// static inline populate_B_ALGN(int ldb, int target_ldb, double* B, int M){
//   int N_ = ldb + target_ldb - (ldb/target_ldb)*target_ldb;
//   for(int i = 0; i < M; i++){
//     int j = 0;
//     for (j = 0; j < ldb; j++)
//     {
//       B_SUB_ALGN[(i+(j/target_ldb))*target_ldb+(j%target_ldb)] = B[ldb*i+j];
//     }
//   }
//   if(ldb < N_){
//     int col = N_-target_ldb;
//     for(int i = 0; i < M; i++){
//       for(int j = 0 ;j < N_-ldb; j++){
//         B_SUB_ALGN[(i+(col/target_ldb))*target_ldb+(col%target_ldb)] = 0;
//       }
//     }
//   }
// }

static inline void do_blockL2(int lda, int ldb, int ldc, int M, int N, int K, double *A, double *B, double *C)
{
  for (int j = 0; j < N; j += L2_N)
  {
    for (int k = 0; k < K; k += L2_K)
    {
      int K_LDB = k * ldb;
      int J = K_LDB + j;
#ifdef ENABLE_L2_CACHING
      populate_B_CACHED(ldb, B + J, B_L2_CACHED, L2_K, L2_N);
#endif
      for (int i = 0; i < M; i += L2_M)
      {
#ifdef ENABLE_L2_CACHING
        do_blockL1(lda, L2_N, ldc, L2_M, L2_N, L2_K, A + i * lda + k, B_L2_CACHED, C + i * ldc + j);
#else
        do_blockL1(lda, ldb, ldc, L2_M, L2_N, L2_K, A + i * lda + k, B + J, C + i * ldc + j);
#endif
      }
    }
  }
}

static double *pad(int lda, int M, int N, double *A)
{
  double *newA = malloc(M * N * sizeof(double));
  int i = 0, j = 0;
  for (i = 0; i < lda; i++)
  {
    for (j = 0; j < lda; j++)
    {
      newA[i * N + j] = A[i * lda + j];
    }
    for (; j < N; j++)
    {
      newA[i * N + j] = 0;
    }
  }
  for (; i < M; i++)
  {
    for (j = 0; j < N; j++)
    {
      newA[i * N + j] = 0;
    }
  }
  return newA;
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */
void square_dgemm(int LD, double *A_, double *B_, double *C)
{
  // B_L2_CACHED = buffer + 64 - ((int)&buffer) % 64;
  if (A_L3_CACHED == NULL)
    A_L3_CACHED = (double *)aligned_alloc(64, 10240 * sizeof(double));
  if (B_L2_CACHED == NULL)
    B_L2_CACHED = (double *)aligned_alloc(64, 10240 * sizeof(double));
  if (B_L1_CACHED == NULL)
    B_L1_CACHED = (double *)aligned_alloc(64, 10240 * sizeof(double));

  double *A = A_;
  double *B = B_;
  int lda = LD;
  int ldb = LD;
  int ldc = LD;
  int M = LD, N = LD, K = LD;
  if ((LD % L3_M != 0) || (LD % L3_K != 0))
  {
    M = ((LD / L3_M) * L3_M) + L3_M;
    K = ((LD / L3_K) * L3_K) + L3_K;
    A = pad(lda, M, K, A_);
    lda = K;
  }

  if ((LD % L3_K != 0) || (LD % L3_N != 0))
  {
    K = ((LD / L3_K) * L3_K) + L3_K;
    N = ((LD / L3_N) * L3_N) + L3_N;
    B = pad(ldb, K, N, B_);
    ldb = N;
  }
  /* For each block-row of A */
  for (int i = 0; i < M; i += L3_M)
  {
    int I_LDC = i * ldc;
    int I_LDA = i * lda;
    for (int k = 0; k < K; k += L3_K)
    {
      int A_I = I_LDA + k;
      int K_LDB = k * ldb;

#ifdef ENABLE_L3_CACHING
      populate_B_CACHED(lda, A + i * lda + k, A_L3_CACHED, L3_M, L3_K);
#endif
      for (int j = 0; j < N; j += L3_N)
      {
#ifdef ENABLE_L3_CACHING
        do_blockL2(L3_K, ldb, ldc, L3_M, L3_N, L3_K, A_L3_CACHED, B + K_LDB + j, C + I_LDC + j);
#else
        /* Perform individual block dgemm */
        do_blockL2(lda, ldb, ldc, L3_M, L3_N, L3_K, A + A_I, B + K_LDB + j, C + I_LDC + j);
#endif
      }
    }
  }
  // free(B_L2_CACHED);
  // B_L1_CACHED = NULL;
  // free(B_L1_CACHED);
  // B_L2_CACHED = NULL;
  if (A != A_)
    free(A);
  if (B != B_)
    free(B);
  return;
}
