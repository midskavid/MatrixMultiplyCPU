/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */
#include <immintrin.h>
#include <avx2intrin.h>
#define min(a, b) (((a) < (b)) ? (a) : (b))

#if !defined(BLOCK_SIZEL2)
#define BLOCK_SIZEL2 28
#endif

#if !defined(BLOCK_SIZEL3)
#define BLOCK_SIZEL3 395
#endif

const char *dgemm_desc = "Simple blocked dgemm.";

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B using SIMD operations
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static inline void do_block_SIMD(int lda, int M, int N, int K, double *A, double *B, double *C)
{
  register __m256d c00_c01_c02_c03 = _mm256_loadu_pd(C);
  register __m256d c10_c11_c12_c13 = _mm256_loadu_pd(C + lda);

  for (int kk = 0; kk < K; ++kk)
  {
    register __m256d a0x = _mm256_broadcast_sd(A + kk);
    register __m256d a1x = _mm256_broadcast_sd(A + kk + lda);

    register __m256d b = _mm256_loadu_pd(B + kk * lda);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
  }

  _mm256_storeu_pd(C, c00_c01_c02_c03);
  _mm256_storeu_pd(C + lda, c10_c11_c12_c13);
}

static inline void do_block_naive(int lda, int M, int N, int K, double *A, double *B, double *C)
{
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */
    for (int j = 0; j < N; ++j)
    {
      /* Compute C(i,j) */
      double cij = C[i * lda + j];
      for (int k = 0; k < K; ++k)
#ifdef TRANSPOSE
        cij += A[i * lda + k] * B[j * lda + k];
#else
        cij += A[i * lda + k] * B[k * lda + j];
#endif
      C[i * lda + j] = cij;
    }
}

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static inline void do_block(int lda, int M, int N, int K, double *A, double *B, double *C)
{
  /* For each row i of A */
  for (int i = 0; i < M; i += 2)
    /* For each column j of B */
    for (int j = 0; j < N; j += 4)
    {
      /* Compute C(i,j) */
      for (int k = 0; k < K; k += 4)
      {
        int M_ = min(2, M - i);
        int N_ = min(4, N - j);
        int K_ = min(4, K - k);
        if (M_ == 2 && N_ == 4)
        {
#ifdef TRANSPOSE
          do_block_SIMD(lda, M_, N_, K_, A + i * lda + k, B + j * lda + k, C + i * lda + j);
#else
          do_block_SIMD(lda, M_, N_, K_, A + i * lda + k, B + k * lda + j, C + i * lda + j);
#endif
        }
        else
        {
#ifdef TRANSPOSE
          do_block_naive(lda, M_, N_, K_, A + i * lda + k, B + j * lda + k, C + i * lda + j);
#else
          do_block_naive(lda, M_, N_, K_, A + i * lda + k, B + k * lda + j, C + i * lda + j);
#endif
        }
      }
    }
}

static inline void do_blockL2(int lda, int M, int N, int K, double *A, double *B, double *C)
{
  /* For each block-row of A */
  for (int i = 0; i < M; i += BLOCK_SIZEL2)
    /* For each block-column of B */
    for (int j = 0; j < N; j += BLOCK_SIZEL2)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < K; k += BLOCK_SIZEL2)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M_ = min(BLOCK_SIZEL2, M - i);
        int N_ = min(BLOCK_SIZEL2, N - j);
        int K_ = min(BLOCK_SIZEL2, K - k);

        /* Perform individual block dgemm */
#ifdef TRANSPOSE
        do_block(lda, M_, N_, K_, A + i * lda + k, B + j * lda + k, C + i * lda + j);
#else
        do_block(lda, M_, N_, K_, A + i * lda + k, B + k * lda + j, C + i * lda + j);
#endif
      }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double *A, double *B, double *C)
{
#ifdef TRANSPOSE
  for (int i = 0; i < lda; ++i)
    for (int j = i + 1; j < lda; ++j)
    {
      double t = B[i * lda + j];
      B[i * lda + j] = B[j * lda + i];
      B[j * lda + i] = t;
    }
#endif
  /* For each block-row of A */
  for (int i = 0; i < lda; i += BLOCK_SIZEL3)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZEL3)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZEL3)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M = min(BLOCK_SIZEL3, lda - i);
        int N = min(BLOCK_SIZEL3, lda - j);
        int K = min(BLOCK_SIZEL3, lda - k);

        /* Perform individual block dgemm */
#ifdef TRANSPOSE
        do_blockL2(lda, M, N, K, A + i * lda + k, B + j * lda + k, C + i * lda + j);
#else
        do_blockL2(lda, M, N, K, A + i * lda + k, B + k * lda + j, C + i * lda + j);
#endif
      }
#if TRANSPOSE
  for (int i = 0; i < lda; ++i)
    for (int j = i + 1; j < lda; ++j)
    {
      double t = B[i * lda + j];
      B[i * lda + j] = B[j * lda + i];
      B[j * lda + i] = t;
    }
#endif
}
