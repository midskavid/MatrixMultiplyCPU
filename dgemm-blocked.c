#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>
/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

const char *dgemm_desc = "Simple blocked dgemm.";

#if !defined(L1_BLOCK_SIZE)
#define L1_BLOCK_SIZE 300
// #define BLOCK_SIZE 719
#endif

#ifndef L2_BLOCK_SIZE
#define L2_BLOCK_SIZE 37
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

static void do_block_modified(int lda, int M, int N, int K, double *A, double *B_modified, double *B, double *C)
{

  // printf("Printing B\n");
  // for (int i = 0; i < K; i++)
  // {
  //   for (int j = 0; j < N; j++)
  //   {
  //     printf("%f ", B[i * lda + j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n==============\n");
  // printf("Printing B_modified\n");
  // for (int i = 0; i < K; i++)
  // {
  //   for (int j = 0; j < N; j++)
  //   {
  //     printf("%f ", B_modified[i * K + j]);
  //   }
  //   printf("\n");
  // }
  // printf("\n==============\n");

  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */
    for (int j = 0; j < N; ++j)
    {
      /* Compute C(i,j) */
      double cij = C[i * lda + j];
      for (int k = 0; k < K; ++k)
      {
        cij += A[i * lda + k] * B_modified[j * K + k];
        // if (B_modified[j * K + k] != B[k * lda + j])
        // {
        //   printf("Problem with i = %d, j = %d, k = %d!!!\n", i, j, k);
        //   printf("M = %d, N = %d, K = %d\n", M, N, K);
        //   exit(0);
        // }
      }
      C[i * lda + j] = cij;
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double *A, double *B, double *C)
{
  // printf("LDA = %d\n\n", lda);
  // printf("size of double is %d", sizeof(double));
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

#ifdef TRANSPOSE
            double *B_transpose = (double *)malloc(sizeof(double) * L2_BLOCK_SIZE * L2_BLOCK_SIZE);
#endif
            for (int k_block_l2 = 0; k_block_l2 < K_; k_block_l2 += L2_BLOCK_SIZE)
            {
              /* Correct block dimensions if L2 block "goes off edge of" the matrix */
              int M = min(L2_BLOCK_SIZE, M_ - i_block_l2);
              int N = min(L2_BLOCK_SIZE, N_ - j_block_l2);
              int K = min(L2_BLOCK_SIZE, K_ - k_block_l2);

              /* Perform individual block dgemm */
#ifdef TRANSPOSE
              // printf("Printing without Transpose:\n");
              for (int b_i = 0; b_i < L2_BLOCK_SIZE; b_i++)
              {
                for (int b_j = 0; b_j < L2_BLOCK_SIZE; b_j++)
                {
                  if(b_i < K && b_j < N)
                    B_transpose[(b_j * L2_BLOCK_SIZE) + b_i] = B[((k + k_block_l2 + b_i) * lda) + j + j_block_l2 + b_j];
                  else
                    B_transpose[(b_j * L2_BLOCK_SIZE) + b_i] = 0;
                  // printf("%f ", B[((k + k_block_l2 + b_i) * lda) + j + j_block_l2 + b_j]) ;
                }
                // printf("\n");
              }

              // printf("\nPrinting With Transpose:\n");
              // for (int b_i = 0; b_i < K; b_i++)
              // {
              //   for (int b_j = 0; b_j < N; b_j++)
              //   {
              //     printf("%f ", B_transpose[b_i*K + b_j]);
              //   }
              //   printf("\n");
              // }
              // printf("===========================\n");

              do_block_modified(lda, M, N, K,
                                A + i * lda + k + i_block_l2 * lda + k_block_l2,
                                B_transpose, B + k * lda + j + k_block_l2 * lda + j_block_l2, //B + j * lda + k + j_block_l2 * lda + k_block_l2,
                                C + i * lda + j + i_block_l2 * lda + j_block_l2);
#else
              do_block(lda, M, N, K,
                       A + i * lda + k + i_block_l2 * lda + k_block_l2,
                       B + k * lda + j + k_block_l2 * lda + j_block_l2,
                       C + i * lda + j + i_block_l2 * lda + j_block_l2);
#endif
            }
#ifdef TRANSPOSE
            free(B_transpose);
#endif
          }
        }
      }
}
