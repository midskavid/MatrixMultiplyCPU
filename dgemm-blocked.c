#include <stdio.h>
/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

const char *dgemm_desc = "Simple blocked dgemm.";

#if !defined(L1_BLOCK_SIZE)
#define L1_BLOCK_SIZE 150
// #define BLOCK_SIZE 719
#endif

#ifndef L2_BLOCK_SIZE
#define L2_BLOCK_SIZE 32
#endif

#define min(a, b) (((a) < (b)) ? (a) : (b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block(int lda, int M, int N, int K, double *A, double *B, double *C)
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

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double *A, double *B, double *C)
{
  // printf("LDA = %d", lda);
  // printf("size of double is %d", sizeof(double));
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
  for (int i = 0; i < lda; i += L1_BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += L1_BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += L1_BLOCK_SIZE)
      {
        /* Correct block dimensions if L1 block "goes off edge of" the matrix */
        int M_ = min(L1_BLOCK_SIZE, lda - i);
        int N_ = min(L1_BLOCK_SIZE, lda - j);
        int K_ = min(L1_BLOCK_SIZE, lda - k);

        for (int i_block_l2 = 0; i_block_l2 < M_; i_block_l2 += L2_BLOCK_SIZE)
        {
          for (int j_block_l2 = 0; j_block_l2 < N_; j_block_l2 += L2_BLOCK_SIZE)
          {
            for (int k_block_l2 = 0; k_block_l2 < K_; k_block_l2 += L2_BLOCK_SIZE)
            {
              /* Correct block dimensions if L2 block "goes off edge of" the matrix */
              int M = min(L2_BLOCK_SIZE, M_ - i_block_l2);
              int N = min(L2_BLOCK_SIZE, N_ - j_block_l2);
              int K = min(L2_BLOCK_SIZE, K_ - k_block_l2);

              /* Perform individual block dgemm */
#ifdef TRANSPOSE
              do_block(lda, M, N, K,
                       A + i * lda + k + i_block_l2 * lda + k_block_l2,
                       B + j * lda + k + j_block_l2 * lda + k_block_l2,
                       C + i * lda + j + i_block_l2 * lda + j_block_l2);
#else
              do_block(lda, M, N, K,
                       A + i * lda + k + i_block_l2 * lda + k_block_l2,
                       B + k * lda + j + k_block_l2 * lda + j_block_l2,
                        C + i * lda + j + i_block_l2 * lda + j_block_l2);
#endif
            }
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
}
