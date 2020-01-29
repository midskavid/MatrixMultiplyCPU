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


#if !defined(BLOCK_SIZEL1)
#define BLOCK_SIZEL1 128
#endif

#if !defined(BLOCK_SIZEL2)
#define BLOCK_SIZEL2 256
#endif

#if !defined(BLOCK_SIZEL3)
#define BLOCK_SIZEL3 1024
#endif



#if !defined(KERNEL)
#define KERNEL 8
#endif


const char* dgemm_desc = "Simple blocked dgemm.";

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B using SIMD operations
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */


// static inline void do_block_SIMD8x8(int lda, double* restrict A, double* restrict B, double* restrict C) {
//   A = __builtin_assume_aligned (A, 4);
//   B = __builtin_assume_aligned (B, 4);
//   C = __builtin_assume_aligned (C, 4);

//   register __m256d c00_c01_c02_c03 = _mm256_loadu_pd(C);
//   register __m256d c10_c11_c12_c13 = _mm256_loadu_pd(C+lda);
//   register __m256d c20_c21_c22_c23 = _mm256_loadu_pd(C+2*lda);
//   register __m256d c30_c31_c32_c33 = _mm256_loadu_pd(C+3*lda);
//   register __m256d c40_c41_c42_c43 = _mm256_loadu_pd(C+4*lda);
//   register __m256d c50_c51_c52_c53 = _mm256_loadu_pd(C+5*lda);
//   register __m256d c60_c61_c62_c63 = _mm256_loadu_pd(C+6*lda);
//   register __m256d c70_c71_c72_c73 = _mm256_loadu_pd(C+7*lda);
  
//   for (int kk=0;kk<8;++kk) {
//     register __m256d a0x = _mm256_broadcast_sd(A+kk);
//     register __m256d a1x = _mm256_broadcast_sd(A+kk+lda);
//     register __m256d a2x = _mm256_broadcast_sd(A+kk+2*lda);
//     register __m256d a3x = _mm256_broadcast_sd(A+kk+3*lda);
//     register __m256d a4x = _mm256_broadcast_sd(A+kk+4*lda);
//     register __m256d a5x = _mm256_broadcast_sd(A+kk+5*lda);
//     register __m256d a6x = _mm256_broadcast_sd(A+kk+6*lda);
//     register __m256d a7x = _mm256_broadcast_sd(A+kk+7*lda);



//     register __m256d b1 = _mm256_loadu_pd(B+kk*lda);
//     register __m256d b1 = _mm256_loadu_pd(B+kk*lda);

//     c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
//     c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
//     c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
//     c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
//     c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b, c40_c41_c42_c43);
//     c50_c51_c52_c53 = _mm256_fmadd_pd(a5x, b, c50_c51_c52_c53);
//     c60_c61_c62_c63 = _mm256_fmadd_pd(a6x, b, c60_c61_c62_c63);
//     c70_c71_c72_c73 = _mm256_fmadd_pd(a7x, b, c70_c71_c72_c73);

//   }
//   _mm256_storeu_pd (C, c00_c01_c02_c03);
//   _mm256_storeu_pd (C+lda, c10_c11_c12_c13);
//   _mm256_storeu_pd (C+2*lda, c20_c21_c22_c23);
//   _mm256_storeu_pd (C+3*lda, c30_c31_c32_c33);
//   _mm256_storeu_pd (C+4*lda, c40_c41_c42_c43);
//   _mm256_storeu_pd (C+5*lda, c50_c51_c52_c53);
//   _mm256_storeu_pd (C+6*lda, c60_c61_c62_c63);
//   _mm256_storeu_pd (C+7*lda, c70_c71_c72_c73);

// }



static inline void do_block_SIMD8x4(int lda, double* restrict A, double* restrict B, double* restrict C) {
  // A = __builtin_assume_aligned (A, 8);
  // B = __builtin_assume_aligned (B, 8);
  // C = __builtin_assume_aligned (C, 8);

  register __m256d c00_c01_c02_c03 = _mm256_loadu_pd(C);
  register __m256d c10_c11_c12_c13 = _mm256_loadu_pd(C+lda);
  register __m256d c20_c21_c22_c23 = _mm256_loadu_pd(C+2*lda);
  register __m256d c30_c31_c32_c33 = _mm256_loadu_pd(C+3*lda);
  register __m256d c40_c41_c42_c43 = _mm256_loadu_pd(C+4*lda);
  register __m256d c50_c51_c52_c53 = _mm256_loadu_pd(C+5*lda);
  register __m256d c60_c61_c62_c63 = _mm256_loadu_pd(C+6*lda);
  register __m256d c70_c71_c72_c73 = _mm256_loadu_pd(C+7*lda);
  


#if 1
  for (int kk=0;kk<4;++kk) {
    register __m256d a0x = _mm256_broadcast_sd(A+kk);
    register __m256d a1x = _mm256_broadcast_sd(A+kk+lda);
    register __m256d a2x = _mm256_broadcast_sd(A+kk+2*lda);
    register __m256d a3x = _mm256_broadcast_sd(A+kk+3*lda);
    register __m256d a4x = _mm256_broadcast_sd(A+kk+4*lda);
    register __m256d a5x = _mm256_broadcast_sd(A+kk+5*lda);
    register __m256d a6x = _mm256_broadcast_sd(A+kk+6*lda);
    register __m256d a7x = _mm256_broadcast_sd(A+kk+7*lda);



    register __m256d b = _mm256_loadu_pd(B+kk*lda);
    

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
    register __m256d a1x = _mm256_broadcast_sd(A+lda);
    register __m256d a2x = _mm256_broadcast_sd(A+2*lda);
    register __m256d a3x = _mm256_broadcast_sd(A+3*lda);
    register __m256d a4x = _mm256_broadcast_sd(A+4*lda);
    register __m256d a5x = _mm256_broadcast_sd(A+5*lda);
    register __m256d a6x = _mm256_broadcast_sd(A+6*lda);
    register __m256d a7x = _mm256_broadcast_sd(A+7*lda);



    register __m256d b = _mm256_loadu_pd(B);
    

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
    c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b, c40_c41_c42_c43);
    c50_c51_c52_c53 = _mm256_fmadd_pd(a5x, b, c50_c51_c52_c53);
    c60_c61_c62_c63 = _mm256_fmadd_pd(a6x, b, c60_c61_c62_c63);
    c70_c71_c72_c73 = _mm256_fmadd_pd(a7x, b, c70_c71_c72_c73);

    a0x = _mm256_broadcast_sd(A+1);
    a1x = _mm256_broadcast_sd(A+1+lda);
    a2x = _mm256_broadcast_sd(A+1+2*lda);
    a3x = _mm256_broadcast_sd(A+1+3*lda);
    a4x = _mm256_broadcast_sd(A+1+4*lda);
    a5x = _mm256_broadcast_sd(A+1+5*lda);
    a6x = _mm256_broadcast_sd(A+1+6*lda);
    a7x = _mm256_broadcast_sd(A+1+7*lda);

    b = _mm256_loadu_pd(B+1*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
    c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b, c40_c41_c42_c43);
    c50_c51_c52_c53 = _mm256_fmadd_pd(a5x, b, c50_c51_c52_c53);
    c60_c61_c62_c63 = _mm256_fmadd_pd(a6x, b, c60_c61_c62_c63);
    c70_c71_c72_c73 = _mm256_fmadd_pd(a7x, b, c70_c71_c72_c73);

    a0x = _mm256_broadcast_sd(A+2);
    a1x = _mm256_broadcast_sd(A+2+lda);
    a2x = _mm256_broadcast_sd(A+2+2*lda);
    a3x = _mm256_broadcast_sd(A+2+3*lda);
    a4x = _mm256_broadcast_sd(A+2+4*lda);
    a5x = _mm256_broadcast_sd(A+2+5*lda);
    a6x = _mm256_broadcast_sd(A+2+6*lda);
    a7x = _mm256_broadcast_sd(A+2+7*lda);

    b = _mm256_loadu_pd(B+2*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
    c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b, c40_c41_c42_c43);
    c50_c51_c52_c53 = _mm256_fmadd_pd(a5x, b, c50_c51_c52_c53);
    c60_c61_c62_c63 = _mm256_fmadd_pd(a6x, b, c60_c61_c62_c63);
    c70_c71_c72_c73 = _mm256_fmadd_pd(a7x, b, c70_c71_c72_c73);

    a0x = _mm256_broadcast_sd(A+3);
    a1x = _mm256_broadcast_sd(A+3+lda);
    a2x = _mm256_broadcast_sd(A+3+2*lda);
    a3x = _mm256_broadcast_sd(A+3+3*lda);
    a4x = _mm256_broadcast_sd(A+3+4*lda);
    a5x = _mm256_broadcast_sd(A+3+5*lda);
    a6x = _mm256_broadcast_sd(A+3+6*lda);
    a7x = _mm256_broadcast_sd(A+3+7*lda);

    b = _mm256_loadu_pd(B+3*lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
    c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b, c40_c41_c42_c43);
    c50_c51_c52_c53 = _mm256_fmadd_pd(a5x, b, c50_c51_c52_c53);
    c60_c61_c62_c63 = _mm256_fmadd_pd(a6x, b, c60_c61_c62_c63);
    c70_c71_c72_c73 = _mm256_fmadd_pd(a7x, b, c70_c71_c72_c73);

#endif
  _mm256_storeu_pd (C, c00_c01_c02_c03);
  _mm256_storeu_pd (C+lda, c10_c11_c12_c13);
  _mm256_storeu_pd (C+2*lda, c20_c21_c22_c23);
  _mm256_storeu_pd (C+3*lda, c30_c31_c32_c33);
  _mm256_storeu_pd (C+4*lda, c40_c41_c42_c43);
  _mm256_storeu_pd (C+5*lda, c50_c51_c52_c53);
  _mm256_storeu_pd (C+6*lda, c60_c61_c62_c63);
  _mm256_storeu_pd (C+7*lda, c70_c71_c72_c73);

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
static inline void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < M; i+= 8)
  {
    /* For each column j of B */ 
    for (int j = 0; j < N; j+= 4) 
    {
      /* Compute C(i,j) */
      for (int k = 0; k < K; k+=4)
      {
        do_block_SIMD8x4(lda, A + i*lda + k, B + k*lda + j, C + i*lda + j);
      }
    }
  }
}

static inline void do_blockL1 (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each block-row of A */ 
  for (int i = 0; i < M; i += BLOCK_SIZEL1)
    /* For each block-column of B */
    for (int j = 0; j < N; j += BLOCK_SIZEL1)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < K; k += BLOCK_SIZEL1)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M_ = min (BLOCK_SIZEL1, M-i);
        int N_ = min (BLOCK_SIZEL1, N-j);
        int K_ = min (BLOCK_SIZEL1, K-k);

        /* Perform individual block dgemm */
        do_block(lda, M_, N_, K_, A + i*lda + k, B + k*lda + j, C + i*lda + j);
      }
}

static inline void do_blockL2 (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each block-row of A */ 
  for (int i = 0; i < M; i += BLOCK_SIZEL2)
    /* For each block-column of B */
    for (int j = 0; j < N; j += BLOCK_SIZEL2)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < K; k += BLOCK_SIZEL2)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M_ = min (BLOCK_SIZEL2, M-i);
        int N_ = min (BLOCK_SIZEL2, N-j);
        int K_ = min (BLOCK_SIZEL2, K-k);

        /* Perform individual block dgemm */
        do_blockL1(lda, M_, N_, K_, A + i*lda + k, B + k*lda + j, C + i*lda + j);
      }
}


static double* PadAndAlign (int lda, double* mOld, int newLda){
  // int rowNums = ceil((float)lda/row)*row;
  // int colNums = ceil((float)lda/col)*col;
  int N = newLda*newLda;
  double* mNew __attribute__((aligned(32))) = malloc(N * sizeof(double));
  
  for (int ii=0;ii<N;++ii) {
    int rIdx = ii/newLda;
    int cIdx = ii%newLda;
    if (rIdx<lda&&cIdx<lda) mNew[ii] = mOld[rIdx*lda+cIdx];
    else mNew[ii] = 0.0;
  }
  return mNew;
}

static void FillC (int lda, double* Cnew, double* C, int newLda) {
  int N = newLda*newLda;
  for (int ii=0;ii<N;++ii) {
    int rIdx = ii/newLda;
    int cIdx = ii%newLda;
    if (rIdx<lda&&cIdx<lda) C[rIdx*lda+cIdx] = Cnew[ii];
  }

}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  int newLda = (int)ceilf((float)lda/KERNEL)*KERNEL;
  double* newA = PadAndAlign(lda, A, newLda);
  double* newB = PadAndAlign(lda, B, newLda);
  double* newC = PadAndAlign(lda, C, newLda);

  /* For each block-row of A */ 
  for (int i = 0; i < newLda; i += BLOCK_SIZEL3)
    /* For each block-column of B */
    for (int j = 0; j < newLda; j += BLOCK_SIZEL3)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < newLda; k += BLOCK_SIZEL3)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M = min (BLOCK_SIZEL3, newLda-i);
        int N = min (BLOCK_SIZEL3, newLda-j);
        int K = min (BLOCK_SIZEL3, newLda-k);

        /* Perform individual block dgemm */
        do_blockL2(newLda, M, N, K, newA + i*newLda + k, newB + k*newLda + j, newC + i*newLda + j);
      }

  FillC(lda, newC, C, newLda);
  free(newA);
  free(newB);
  free(newC);
}
