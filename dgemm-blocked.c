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

// double buffer[16500];
double *B_SUB_ALGN;

double *B_L2_CACHED = NULL;
double *B_L1_CACHED = NULL;

#if !defined(L2_M)
#define L2_M 64
#endif

#if !defined(L2_N)
#define L2_N 64
#endif

#if !defined(L2_K)
#define L2_K 64
#endif

#if !defined(L1_M)
#define L1_M 64
#endif

#if !defined(L1_N)
#define L1_N 64
#endif

#if !defined(L1_K)
#define L1_K 64
#endif

#if !defined(BLOCK_SIZEL3)
#define BLOCK_SIZEL3 64
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

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
  register __m256d c00_c01_c02_c03 = _mm256_loadu_pd(C);
  register __m256d c10_c11_c12_c13 = _mm256_loadu_pd(C + ldc);
  register __m256d c20_c21_c22_c23 = _mm256_loadu_pd(C + 2 * ldc);
  register __m256d c30_c31_c32_c33 = _mm256_loadu_pd(C + 3 * ldc);
  register __m256d c40_c41_c42_c43 = _mm256_loadu_pd(C + 4 * ldc);

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
      for (int j = 0; j < N; j += 4)
      {
        /* Compute C(i,j) */
        int M_ = min(2, M - i);
        int N_ = min(4, N - j);
        int K_ = min(4, K - k);
        if (M_ == 2 && N_ == 4)
        {
#ifdef TRANSPOSE
          do_block_SIMD(lda, M_, N_, K_, A + i * lda + k, B + j * lda + k, C + i * lda + j);
#else
          do_block_SIMD(lda, ldb, ldc, M_, N_, K_, A + i * lda + k, B + k * ldb + j, C + i * ldc + j);
#endif
        }
        else
        {
#ifdef TRANSPOSE
          do_block_naive(lda, M_, N_, K_, A + i * lda + k, B + j * lda + k, C + i * lda + j);
#else
          do_block_naive(lda, ldb, ldc, M_, N_, K_, A + i * lda + k, B + k * lda + j, C + i * lda + j);
#endif
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
#ifdef TRANSPOSE
        do_block_SIMD5x4(lda, A + i * lda + k, B + j * lda + k, C + i * lda + j);
#else
        do_block_SIMD5x4(lda, ldb, ldc, A + i * lda + k, B + k * ldb + j, C + i * ldc + j);
#endif
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
#ifdef TRANSPOSE
        cij += A[i * lda + k] * B[j * lda + k];
#else
        cij += A[i * lda + k] * B[k * ldb + j];
#endif
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
#ifdef TRANSPOSE
        cij += A[i * lda + k] * B[j * lda + k];
#else
        cij += A[i * lda + k] * B[k * ldb + j];
#endif
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
    for (int j = 0; j < Nalgn; ++j)
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

static inline void do_blockL1(int lda, int ldb, int ldc, int M, int N, int K, double *A, double *B, double *C)
{
  for (int i = 0; i < M; i += L1_M)
  {
    for (int k = 0; k < K; k += L1_K)
    {
      for (int j = 0; j < N; j += L1_N)
      {
        populate_B_CACHED(ldb, B + k * ldb + j, B_L1_CACHED, L1_K, L1_N);
#ifdef TRANSPOSE
        do_block(lda, M_, N_, K_, A + i * lda + k, B + j * ldb + k, C + i * ldc + j);
#else
        do_block5x4(lda, L1_N, ldc, L1_M, L1_N, L1_K, A + i * lda + k, B_L1_CACHED, C + i * ldc + j);
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
  for (int k = 0; k < K; k += L2_K)
  {
    for (int j = 0; j < N; j += L2_N)
    {
      populate_B_CACHED(ldb, B + k * ldb + j, B_L2_CACHED, L2_K, L2_N);
      for (int i = 0; i < M; i += L2_M)
      {
#ifdef TRANSPOSE
        do_blockL1(lda, M_, N_, K_, A + i * lda + k, B + j * ldb + k, C + i * ldc + j);
#else
        do_blockL1(lda, L2_N, ldc, L2_M, L2_N, L2_K, A + i * lda + k, B_L2_CACHED, C + i * ldc + j);
#endif
      }
    }
  }
}

static double *pad(int lda, int N, double *A)
{
  double *newA = malloc(N * N * sizeof(double));
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
  for (; i < N; i++)
  {
    newA[i * N + j] = 0;
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
  if (B_L2_CACHED == NULL)
    B_L2_CACHED = (double *)aligned_alloc(64, 8200 * sizeof(double));
  if (B_L1_CACHED == NULL)
    B_L1_CACHED = (double *)aligned_alloc(64, 5000 * sizeof(double));

  double *A = A_;
  double *B = B_;
  int lda = LD;
  int ldb = LD;
  int ldc = LD;
  if (LD % BLOCK_SIZEL3 != 0)
  {
    int L = ((LD / BLOCK_SIZEL3) * BLOCK_SIZEL3) + BLOCK_SIZEL3;
    A = pad(lda, L, A_);
    B = pad(lda, L, B_);
    lda = L;
    ldb = L;
  }
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
  {
    int M = min(BLOCK_SIZEL3, lda - i);
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZEL3)
    {
      int N = min(BLOCK_SIZEL3, lda - j);
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZEL3)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int K = min(BLOCK_SIZEL3, lda - k);

        /* Perform individual block dgemm */
#ifdef TRANSPOSE
        do_blockL2(lda, M, N, K, A + i * lda + k, B + j * lda + k, C + i * lda + j);
#else
        do_blockL2(lda, ldb, ldc, M, N, K, A + i * lda + k, B + k * ldb + j, C + i * ldc + j);
#endif
      }
    }
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
  // free(B_L2_CACHED);
  // free(B_L1_CACHED);
  if (A != A_)
  {
    free(A);
    free(B);
  }
  return;
}
