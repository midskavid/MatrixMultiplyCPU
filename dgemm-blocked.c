/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */
#include <immintrin.h>
#include <avx2intrin.h>
#include <math.h>

#define min(a,b) (((a)<(b))?(a):(b))


double *B_L2_CACHED = NULL;
double *B_L1_CACHED = NULL;

#if !defined(L1_M)
#define L1_M 32
#endif

#if !defined(L1_N)
#define L1_N 16
#endif

#if !defined(L1_K)
#define L1_K 16
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
#define L3_N 128
#endif

#if !defined(L3_K)
#define L3_K 128
#endif



#if !defined(KERNEL)
#define KERNEL 8
#endif


const char* dgemm_desc = "Simple blocked dgemm.";

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B using SIMD operations
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */


static inline void populate_B_CACHED(int ldb, double *B, double *B_Cached, int M, int N)
{
  for (int i = 0; i < M; i++)
  {
    for (int j = 0; j < N; j++)
    {
      B_Cached[i * N + j] = B[i * ldb + j];
    }
  }
}



static inline void do_block_SIMD8x4(int lda, int ldb, int ldc, double *restrict A, double *restrict B, double *restrict C)
{
  // A = __builtin_assume_aligned (A, 8);
  B = __builtin_assume_aligned (B, 64);
  // C = __builtin_assume_aligned (C, 8);

  register __m256d c00_c01_c02_c03 = _mm256_loadu_pd(C);
  register __m256d c10_c11_c12_c13 = _mm256_loadu_pd(C + ldc);
  register __m256d c20_c21_c22_c23 = _mm256_loadu_pd(C + 2 * ldc);
  register __m256d c30_c31_c32_c33 = _mm256_loadu_pd(C + 3 * ldc);
  register __m256d c40_c41_c42_c43 = _mm256_loadu_pd(C + 4 * ldc);
  register __m256d c50_c51_c52_c53 = _mm256_loadu_pd(C + 5 * ldc);
  register __m256d c60_c61_c62_c63 = _mm256_loadu_pd(C + 6 * ldc);
  register __m256d c70_c71_c72_c73 = _mm256_loadu_pd(C + 7 * ldc);

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

  _mm256_storeu_pd(C, c00_c01_c02_c03);
  _mm256_storeu_pd(C + ldc, c10_c11_c12_c13);
  _mm256_storeu_pd(C + 2 * ldc, c20_c21_c22_c23);
  _mm256_storeu_pd(C + 3 * ldc, c30_c31_c32_c33);
  _mm256_storeu_pd(C + 4 * ldc, c40_c41_c42_c43);
  _mm256_storeu_pd(C + 5 * ldc, c50_c51_c52_c53);
  _mm256_storeu_pd(C + 6 * ldc, c60_c61_c62_c63);
  _mm256_storeu_pd(C + 7 * ldc, c70_c71_c72_c73);
}


static inline void do_block_SIMD5x4(int lda, double* restrict A, double* restrict B, double* restrict C) {
  A = __builtin_assume_aligned (A, 4);
  B = __builtin_assume_aligned (B, 4);
  C = __builtin_assume_aligned (C, 4);

  register __m256d c00_c01_c02_c03 = _mm256_loadu_pd(C);
  register __m256d c10_c11_c12_c13 = _mm256_loadu_pd(C+lda);
  register __m256d c20_c21_c22_c23 = _mm256_loadu_pd(C+2*lda);
  register __m256d c30_c31_c32_c33 = _mm256_loadu_pd(C+3*lda);
  register __m256d c40_c41_c42_c43 = _mm256_loadu_pd(C+4*lda);
#if 0
  for (int kk=0;kk<4;++kk) {
    register __m256d a0x = _mm256_broadcast_sd(A+kk);
    register __m256d a1x = _mm256_broadcast_sd(A+kk+lda);
    register __m256d a2x = _mm256_broadcast_sd(A+kk+2*lda);
    register __m256d a3x = _mm256_broadcast_sd(A+kk+3*lda);
    register __m256d a4x = _mm256_broadcast_sd(A+kk+4*lda);

    register __m256d b = _mm256_loadu_pd(B+kk*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
    c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b, c40_c41_c42_c43);
  }
#else
    register __m256d a0x = _mm256_broadcast_sd(A);
    register __m256d a1x = _mm256_broadcast_sd(A+lda);
    register __m256d a2x = _mm256_broadcast_sd(A+2*lda);
    register __m256d a3x = _mm256_broadcast_sd(A+3*lda);
    register __m256d a4x = _mm256_broadcast_sd(A+4*lda);

    register __m256d b = _mm256_loadu_pd(B);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
    c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b, c40_c41_c42_c43);

    a0x = _mm256_broadcast_sd(A+1);
    a1x = _mm256_broadcast_sd(A+1+lda);
    a2x = _mm256_broadcast_sd(A+1+2*lda);
    a3x = _mm256_broadcast_sd(A+1+3*lda);
    a4x = _mm256_broadcast_sd(A+1+4*lda);

    b = _mm256_loadu_pd(B+1*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
    c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b, c40_c41_c42_c43);

    a0x = _mm256_broadcast_sd(A+2);
    a1x = _mm256_broadcast_sd(A+2+lda);
    a2x = _mm256_broadcast_sd(A+2+2*lda);
    a3x = _mm256_broadcast_sd(A+2+3*lda);
    a4x = _mm256_broadcast_sd(A+2+4*lda);

    b = _mm256_loadu_pd(B+2*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
    c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b, c40_c41_c42_c43);

    a0x = _mm256_broadcast_sd(A+3);
    a1x = _mm256_broadcast_sd(A+3+lda);
    a2x = _mm256_broadcast_sd(A+3+2*lda);
    a3x = _mm256_broadcast_sd(A+3+3*lda);
    a4x = _mm256_broadcast_sd(A+3+4*lda);

    b = _mm256_loadu_pd(B+3*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
    c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b, c40_c41_c42_c43);

#endif
  _mm256_storeu_pd (C, c00_c01_c02_c03);
  _mm256_storeu_pd (C+lda, c10_c11_c12_c13);
  _mm256_storeu_pd (C+2*lda, c20_c21_c22_c23);
  _mm256_storeu_pd (C+3*lda, c30_c31_c32_c33);
  _mm256_storeu_pd (C+4*lda, c40_c41_c42_c43);
}


static inline void do_block_SIMD4x4(int lda, double* restrict A, double* restrict B, double* restrict C) {
  
  A = __builtin_assume_aligned (A, 4);
  B = __builtin_assume_aligned (B, 4);
  C = __builtin_assume_aligned (C, 4);

  register __m256d c00_c01_c02_c03 = _mm256_loadu_pd(C);
  register __m256d c10_c11_c12_c13 = _mm256_loadu_pd(C+lda);
  register __m256d c20_c21_c22_c23 = _mm256_loadu_pd(C+2*lda);
  register __m256d c30_c31_c32_c33 = _mm256_loadu_pd(C+3*lda);

#if 0
  for (int kk=0;kk<4;++kk) {
    register __m256d a0x = _mm256_broadcast_sd(A+kk);
    register __m256d a1x = _mm256_broadcast_sd(A+kk+lda);
    register __m256d a2x = _mm256_broadcast_sd(A+kk+2*lda);
    register __m256d a3x = _mm256_broadcast_sd(A+kk+3*lda);

    register __m256d b = _mm256_loadu_pd(B+kk*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
  }
#else
    register __m256d a0x = _mm256_broadcast_sd(A);
    register __m256d a1x = _mm256_broadcast_sd(A+lda);
    register __m256d a2x = _mm256_broadcast_sd(A+2*lda);
    register __m256d a3x = _mm256_broadcast_sd(A+3*lda);
    

    register __m256d b = _mm256_loadu_pd(B);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
    

    a0x = _mm256_broadcast_sd(A+1);
    a1x = _mm256_broadcast_sd(A+1+lda);
    a2x = _mm256_broadcast_sd(A+1+2*lda);
    a3x = _mm256_broadcast_sd(A+1+3*lda);
    

    b = _mm256_loadu_pd(B+1*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
    

    a0x = _mm256_broadcast_sd(A+2);
    a1x = _mm256_broadcast_sd(A+2+lda);
    a2x = _mm256_broadcast_sd(A+2+2*lda);
    a3x = _mm256_broadcast_sd(A+2+3*lda);
    

    b = _mm256_loadu_pd(B+2*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
    

    a0x = _mm256_broadcast_sd(A+3);
    a1x = _mm256_broadcast_sd(A+3+lda);
    a2x = _mm256_broadcast_sd(A+3+2*lda);
    a3x = _mm256_broadcast_sd(A+3+3*lda);
    

    b = _mm256_loadu_pd(B+3*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
    
#endif
  _mm256_storeu_pd (C, c00_c01_c02_c03);
  _mm256_storeu_pd (C+lda, c10_c11_c12_c13);
  _mm256_storeu_pd (C+2*lda, c20_c21_c22_c23);
  _mm256_storeu_pd (C+3*lda, c30_c31_c32_c33);
}


static inline void do_block_SIMD3x4(int lda, double* A, double* B, double* C) {
  register __m256d c00_c01_c02_c03 = _mm256_loadu_pd(C);
  register __m256d c10_c11_c12_c13 = _mm256_loadu_pd(C+lda);
  register __m256d c20_c21_c22_c23 = _mm256_loadu_pd(C+2*lda);

#if 0
  for (int kk=0;kk<4;++kk) {
    register __m256d a0x = _mm256_broadcast_sd(A+kk);
    register __m256d a1x = _mm256_broadcast_sd(A+kk+lda);
    register __m256d a2x = _mm256_broadcast_sd(A+kk+2*lda);

    register __m256d b = _mm256_loadu_pd(B+kk*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
  }
#else
    register __m256d a0x = _mm256_broadcast_sd(A);
    register __m256d a1x = _mm256_broadcast_sd(A+lda);
    register __m256d a2x = _mm256_broadcast_sd(A+2*lda);
    
    register __m256d b = _mm256_loadu_pd(B);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    

    a0x = _mm256_broadcast_sd(A+1);
    a1x = _mm256_broadcast_sd(A+1+lda);
    a2x = _mm256_broadcast_sd(A+1+2*lda);
    
    

    b = _mm256_loadu_pd(B+1*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    
    

    a0x = _mm256_broadcast_sd(A+2);
    a1x = _mm256_broadcast_sd(A+2+lda);
    a2x = _mm256_broadcast_sd(A+2+2*lda);
    
    

    b = _mm256_loadu_pd(B+2*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    
    

    a0x = _mm256_broadcast_sd(A+3);
    a1x = _mm256_broadcast_sd(A+3+lda);
    a2x = _mm256_broadcast_sd(A+3+2*lda);
    
    

    b = _mm256_loadu_pd(B+3*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    
    
#endif
  _mm256_storeu_pd (C, c00_c01_c02_c03);
  _mm256_storeu_pd (C+lda, c10_c11_c12_c13);
  _mm256_storeu_pd (C+2*lda, c20_c21_c22_c23);
}



static inline void do_block_SIMD2x4(int lda, double* A, double* B, double* C) {
  register __m256d c00_c01_c02_c03 = _mm256_loadu_pd(C);
  register __m256d c10_c11_c12_c13 = _mm256_loadu_pd(C+lda);
#if 0
  for (int kk=0;kk<4;++kk) {
    register __m256d a0x = _mm256_broadcast_sd(A+kk);
    register __m256d a1x = _mm256_broadcast_sd(A+kk+lda);

    register __m256d b = _mm256_loadu_pd(B+kk*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
  }
#else
    register __m256d a0x = _mm256_broadcast_sd(A);
    register __m256d a1x = _mm256_broadcast_sd(A+lda);
    
    
    register __m256d b = _mm256_loadu_pd(B);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    
    

    a0x = _mm256_broadcast_sd(A+1);
    a1x = _mm256_broadcast_sd(A+1+lda);
    
    
    

    b = _mm256_loadu_pd(B+1*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    
    
    

    a0x = _mm256_broadcast_sd(A+2);
    a1x = _mm256_broadcast_sd(A+2+lda);
    
    
    

    b = _mm256_loadu_pd(B+2*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    
    
    

    a0x = _mm256_broadcast_sd(A+3);
    a1x = _mm256_broadcast_sd(A+3+lda);
    
    
    

    b = _mm256_loadu_pd(B+3*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    
    
    
#endif
  _mm256_storeu_pd (C, c00_c01_c02_c03);
  _mm256_storeu_pd (C+lda, c10_c11_c12_c13);
}


static inline void do_block_SIMD2x2(int lda, double* restrict A, double* restrict B, double* restrict C) {
  register __m128d c00_c01 = _mm_loadu_pd(C);
  register __m128d c10_c11 = _mm_loadu_pd(C+lda);

  register __m128d a0x = _mm_set1_pd(*(A));
  register __m128d a1x = _mm_set1_pd(*(A+lda));

  register __m128d b = _mm_loadu_pd(B);

  c00_c01 = _mm_fmadd_pd(a0x, b, c00_c01);
  c10_c11 = _mm_fmadd_pd(a1x, b, c10_c11);

  a0x = _mm_set1_pd(*(A+1));
  a1x = _mm_set1_pd(*(A+1+lda));

  b = _mm_loadu_pd(B+1*lda);

  c00_c01 = _mm_fmadd_pd(a0x, b, c00_c01);
  c10_c11 = _mm_fmadd_pd(a1x, b, c10_c11);


  _mm_storeu_pd (C, c00_c01);
  _mm_storeu_pd (C+lda, c10_c11);
}



static inline void do_block_naive (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j) 
    {
      /* Compute C(i,j) */
      double cij = C[i*lda+j];
      for (int k = 0; k < K; ++k)
#ifdef TRANSPOSE
  cij += A[i*lda+k] * B[j*lda+k];
#else
  cij += A[i*lda+k] * B[k*lda+j];
#endif
      C[i*lda+j] = cij;
    }
}


static inline void do_block_naiveSIMD (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  int Malgn = (M/2)*2;
  int Nalgn = (N/2)*2;
  int Kalgn = (K/2)*2;

  for (int i = 0; i < Malgn; i+= 2)
  {
    /* For each column j of B */ 
    for (int j = 0; j < Nalgn; j+= 2) 
    {
      /* Compute C(i,j) */
      for (int k = 0; k < Kalgn; k+=2)
      {
#ifdef TRANSPOSE
  do_block_SIMD2x2(lda, A + i*lda + k, B + j*lda + k, C + i*lda + j);
#else
  do_block_SIMD2x2(lda, A + i*lda + k, B + k*lda + j, C + i*lda + j);
#endif   
      }
    }
  }

   do_block_naive(lda, Malgn, Nalgn, K-Kalgn, A+Kalgn, B+Kalgn*lda, C);
   do_block_naive(lda, Malgn, N-Nalgn, Kalgn, A, B+Nalgn, C+Nalgn);
   do_block_naive(lda, Malgn, N-Nalgn, K-Kalgn, A+Kalgn, B+lda*Kalgn+Nalgn, C+Nalgn);
   do_block_naive(lda, M-Malgn, Nalgn, Kalgn, A+lda*Malgn, B, C+lda*Malgn);
   do_block_naive(lda, M-Malgn, Nalgn, K-Kalgn, A+lda*Malgn+Kalgn, B+lda*Kalgn, C+lda*Malgn);
   do_block_naive(lda, M-Malgn, N-Nalgn, Kalgn, A+lda*Malgn, B+Nalgn, C+lda*Malgn+Nalgn);
   do_block_naive(lda, M-Malgn, N-Nalgn, K-Kalgn, A+lda*Malgn+Kalgn, B+lda*Kalgn+Nalgn, C+lda*Malgn+Nalgn);


}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static inline void do_block8x4(int lda, int ldb, int ldc, int M, int N, int K, double *A, double *B, double *C)
{
  /* For each row i of A */
  for (int i = 0; i < M; i += 8)
  {
    /* For each column j of B */
    /* Compute C(i,j) */
    for (int k = 0; k < K; k += 4)
    {
      for (int j = 0; j < N; j += 4)
      {
        do_block_SIMD8x4(lda, ldb, ldc, A + i * lda + k, B + k * ldb + j, C + i * ldc + j);
      }
    }
  }
}

static inline void do_blockL1(int lda, int ldb, int ldc, int M, int N, int K, double *A, double *B, double *C)
{
  for (int i = 0; i < M; i += L1_M)
  {
    for (int k = 0; k < K; k += L1_K)
    {
      for (int j = 0; j < N; j += L1_N)
      {
#ifdef ENABLE_L1_CACHING
        populate_B_CACHED(ldb, B + k * ldb + j, B_L1_CACHED, L1_K, L1_N);
        do_block8x4(lda, L1_N, ldc, L1_M, L1_N, L1_K, A + i * lda + k, B_L1_CACHED, C + i * ldc + j);
#else
        do_block8x4(lda, ldb, ldc, L1_M, L1_N, L1_K, A + i * lda + k, B + k * ldb + j, C + i * ldc + j);
#endif

      }
    }
  }
}

static inline void do_blockL2(int lda, int ldb, int ldc, int M, int N, int K, double *A, double *B, double *C)
{
  for (int i = 0; i < M; i += L2_M)
  {
    for (int k = 0; k < K; k += L2_K)
    {
      for (int j = 0; j < N; j += L2_N)
      {
#ifdef ENABLE_L2_CACHING
        populate_B_CACHED(ldb, B + k * ldb + j, B_L2_CACHED, L2_K, L2_N);        
        do_blockL1(lda, L2_N, ldc, L2_M, L2_N, L2_K, A + i * lda + k, B_L2_CACHED, C + i * ldc + j);
#else
        do_blockL1(lda, ldb, ldc, L2_M, L2_N, L2_K, A + i * lda + k, B + k * ldb + j, C + i * ldc + j);
#endif

      }
    }
  }
}


static double* PadAndAlign(int lda, int M, int N, double *A)
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

static void FillC (int lda, double* Cnew, double* C, int M, int N) {
  for (int i = 0; i < lda; i++) {
    for (int j = 0; j < lda; j++) {
      C[i * lda + j] = Cnew[i * N + j];
    }
  }
}


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */  
void square_dgemm(int LD, double *A_, double *B_, double *C_)
{
  // B_L2_CACHED = buffer + 64 - ((int)&buffer) % 64;
  if (!B_L2_CACHED)
    B_L2_CACHED = (double *)aligned_alloc(64, 8200 * sizeof(double));
  if (!B_L1_CACHED)
    B_L1_CACHED = (double *)aligned_alloc(64, 5000 * sizeof(double));

  double* A = A_;
  double* B = B_;
  double* C = C_;

  int lda = LD;
  int ldb = LD;
  int ldc = LD;
  int M = LD, N = LD, K = LD;
  if ((LD % L3_M != 0) || (LD % L3_K != 0))
  {
    M = ((LD / L3_M) * L3_M) + L3_M;
    K = ((LD / L3_K) * L3_K) + L3_K;
    A = PadAndAlign(lda, M, K, A_);
    lda = K;
  }

  if ((LD % L3_K != 0) || (LD % L3_N != 0))
  {
    K = ((LD / L3_K) * L3_K) + L3_K;
    N = ((LD / L3_N) * L3_N) + L3_N;
    B = PadAndAlign(ldb, K, N, B_);
    ldb = N;
  }

  if ((LD % L3_M != 0) || (LD % L3_N != 0))
  {
    M = ((LD / L3_M) * L3_M) + L3_M;
    N = ((LD / L3_N) * L3_N) + L3_N;    
    C = PadAndAlign(ldc, M, N, C_);
    ldc = N;
  }

  /* For each block-row of A */
  for (int i = 0; i < M; i += L3_M)
  {
    for (int k = 0; k < K; k += L3_K)
    {
      for (int j = 0; j < N; j += L3_N)
      {
        /* Perform individual block dgemm */
        do_blockL2(lda, ldb, ldc, L3_M, L3_N, L3_K, A + i * lda + k, B + k * ldb + j, C + i * ldc + j);
      }
    }
  }
  if (C !=C_ )
    FillC(LD, C, C_, M, N);

  if (A != A_)
    free(A);
  if (B != B_)
    free(B);
  if (C != C_)
    free(C);

  return;
}
