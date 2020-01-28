/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */
#include <immintrin.h>
#include <avx2intrin.h>
#include <malloc.h>
#define min(a,b) (((a)<(b))?(a):(b))



#if !defined(BLOCK_SIZEL2)
#define BLOCK_SIZEL2 40
#endif

#if !defined(BLOCK_SIZEL3)
#define BLOCK_SIZEL3 400
#endif

int L2_M = 40;
int L2_N = 40;
int L2_K = 40;

double* A_block;
double* B_block;

const char* dgemm_desc = "Simple blocked dgemm.";

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B using SIMD operations
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
// static inline void do_block_SIMD(int lda, int M, int N, int K, double* A, double* B, double* C) {
//   register __m256d c00_c01_c02_c03 = _mm256_loadu_pd(C);
//   register __m256d c10_c11_c12_c13 = _mm256_loadu_pd(C+lda);

//   for (int kk=0;kk<K;++kk) {
//     register __m256d a0x = _mm256_broadcast_sd(A+kk);
//     register __m256d a1x = _mm256_broadcast_sd(A+kk+lda);

//     register __m256d b = _mm256_loadu_pd(B+kk*lda);

//     c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
//     c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
//   }

//   _mm256_storeu_pd (C, c00_c01_c02_c03);
//   _mm256_storeu_pd (C+lda, c10_c11_c12_c13);
// }


/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B using SIMD operations
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static inline void do_block_SIMD(int lda, int ldb, int ldc, int M,\
 int N, int K, double* A, double* B, double* C) {
  register __m256d c00_c01_c02_c03 = _mm256_loadu_pd(C);
  register __m256d c10_c11_c12_c13 = _mm256_loadu_pd(C+ldc);

  for (int kk=0;kk<K;++kk) {
    register __m256d a0x = _mm256_broadcast_sd(A+kk);
    register __m256d a1x = _mm256_broadcast_sd(A+kk+lda);

    register __m256d b = _mm256_loadu_pd(B+kk*ldb);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
  }

  _mm256_storeu_pd (C, c00_c01_c02_c03);
  _mm256_storeu_pd (C+ldc, c10_c11_c12_c13);
}


static inline void do_block_SIMD5x4(int lda, int ldb, int ldc, 
int M, int N, int K, double* A, double* B, double* C) {
  register __m256d c00_c01_c02_c03 = _mm256_loadu_pd(C);
  register __m256d c10_c11_c12_c13 = _mm256_loadu_pd(C+ldc);
  register __m256d c20_c21_c22_c23 = _mm256_loadu_pd(C+2*ldc);
  register __m256d c30_c31_c32_c33 = _mm256_loadu_pd(C+3*ldc);
  register __m256d c40_c41_c42_c43 = _mm256_loadu_pd(C+4*ldc);

  for (int kk=0;kk<4;++kk) {
    register __m256d a0x = _mm256_broadcast_sd(A+kk);
    register __m256d a1x = _mm256_broadcast_sd(A+kk+lda);
    register __m256d a2x = _mm256_broadcast_sd(A+kk+2*lda);
    register __m256d a3x = _mm256_broadcast_sd(A+kk+3*lda);
    register __m256d a4x = _mm256_broadcast_sd(A+kk+4*lda);

    register __m256d b = _mm256_loadu_pd(B+kk*ldb);

    c00_c01_c02_c03 = _mm256_fmadd_pd(a0x, b, c00_c01_c02_c03);
    c10_c11_c12_c13 = _mm256_fmadd_pd(a1x, b, c10_c11_c12_c13);
    c20_c21_c22_c23 = _mm256_fmadd_pd(a2x, b, c20_c21_c22_c23);
    c30_c31_c32_c33 = _mm256_fmadd_pd(a3x, b, c30_c31_c32_c33);
    c40_c41_c42_c43 = _mm256_fmadd_pd(a4x, b, c40_c41_c42_c43);
  }

  _mm256_storeu_pd (C, c00_c01_c02_c03);
  _mm256_storeu_pd (C+ldc, c10_c11_c12_c13);
  _mm256_storeu_pd (C+2*ldc, c20_c21_c22_c23);
  _mm256_storeu_pd (C+3*ldc, c30_c31_c32_c33);
  _mm256_storeu_pd (C+4*ldc, c40_c41_c42_c43);
}

static inline void do_block_naive (int lda, int ldb, int ldc, int M, int N, int K, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < M; ++i)
    /* For each column j of B */ 
    for (int j = 0; j < N; ++j) 
    {
      /* Compute C(i,j) */
      double cij = C[i*ldc+j];
      for (int k = 0; k < K; ++k)
#ifdef TRANSPOSE
  cij += A[i*lda+k] * B[j*lda+k];
#else
  cij += A[i*lda+k] * B[k*ldb+j];
#endif
      C[i*ldc+j] = cij;
    }
}


// static inline void do_block_naive (int lda, int M, int N, int K, double* A, double* B, double* C)
// {
//   /* For each row i of A */
//   for (int i = 0; i < M; ++i)
//     /* For each column j of B */ 
//     for (int j = 0; j < N; ++j) 
//     {
//       /* Compute C(i,j) */
//       double cij = C[i*lda+j];
//       for (int k = 0; k < K; ++k)
// #ifdef TRANSPOSE
//   cij += A[i*lda+k] * B[j*lda+k];
// #else
//   cij += A[i*lda+k] * B[k*lda+j];
// #endif
//       C[i*lda+j] = cij;
//     }
// }

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
// static inline void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
// {
//   /* For each row i of A */
//   for (int i = 0; i < M; i+= 2)
//     /* For each column j of B */ 
//     for (int j = 0; j < N; j+= 4) 
//     {
//       /* Compute C(i,j) */
//       for (int k = 0; k < K; k+=4)
//       {
//         int M_ = min (2, M-i);
//         int N_ = min (4, N-j);
//         int K_ = min (4, K-k);
//         if (M_==2&&N_==4) 
//         {
// #ifdef TRANSPOSE
//   do_block_SIMD(lda, M_, N_, K_, A + i*lda + k, B + j*lda + k, C + i*lda + j);
// #else
//   do_block_SIMD(lda, M_, N_, K_, A + i*lda + k, B + k*lda + j, C + i*lda + j);
// #endif   
//         }
//         else
//         {
// #ifdef TRANSPOSE
//   do_block_naive(lda, M_, N_, K_, A + i*lda + k, B + j*lda + k, C + i*lda + j);
// #else
//   do_block_naive(lda, M_, N_, K_, A + i*lda + k, B + k*lda + j, C + i*lda + j);
// #endif
//         }
//       }
//     }
// }

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static inline void do_block_copied(int lda, int ldb, int ldc, int M, int N,\
 int K, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < M; i+= 5)
    /* For each column j of B */ 
    for (int j = 0; j < N; j+= 4) 
    {
      /* Compute C(i,j) */
      for (int k = 0; k < K; k+=4)
      {
        int M_ = min (5, M-i);
        int N_ = min (4, N-j);
        int K_ = min (4, K-k);
        if (M_==5&&N_==4) 
        {
#ifdef TRANSPOSE
  do_block_SIMD(lda, M_, N_, K_, A + i*lda + k, B + j*lda + k, C + i*lda + j);
#else
  do_block_SIMD5x4(lda, ldb, ldc, M_, N_, K_, A + i*lda + k, B + k*ldb + j, C + i*ldc + j);
#endif   
        }
        else
        {
#ifdef TRANSPOSE
  do_block_naive(lda, M_, N_, K_, A + i*lda + k, B + j*lda + k, C + i*lda + j);
#else
  do_block_naive(lda, ldb, ldc, M_, N_, K_, A + i*lda + k, B + k*ldb + j, C + i*ldc + j);
#endif
        }
      }
    }
}


// static inline void do_blockL2 (int lda, int M, int N, int K, double* A, double* B, double* C)
// {
//   /* For each block-row of A */ 
//   for (int i = 0; i < M; i += BLOCK_SIZEL2)
//     /* For each block-column of B */
//     for (int j = 0; j < N; j += BLOCK_SIZEL2)
//       /* Accumulate block dgemms into block of C */
//       for (int k = 0; k < K; k += BLOCK_SIZEL2)
//       {
//   /* Correct block dimensions if block "goes off edge of" the matrix */
//   int M_ = min (BLOCK_SIZEL2, M-i);
//   int N_ = min (BLOCK_SIZEL2, N-j);
//   int K_ = min (BLOCK_SIZEL2, K-k);

//   /* Perform individual block dgemm */
// #ifdef TRANSPOSE
//   do_block(lda, M_, N_, K_, A + i*lda + k, B + j*lda + k, C + i*lda + j);
// #else
//   do_block(lda, M_, N_, K_, A + i*lda + k, B + k*lda + j, C + i*lda + j);
// #endif
//       }
// }


static inline void populate_sub(int lda, int M, int N, int row, int col, double* A, double* A_sub){
  for(int i = 0; i < M; i++){
    for(int j = 0; j < N; j++){
      A_sub[i*N+j] = A[(row+i)*lda+col+j];
    }
  }
}

// static inline void populate_sub_a(int lda, int M, int N, int row, int col, double* A){
//   for(int i = 0; i < M; i++){
//     for(int j = 0; j < N; j++){
//       A_block[i*N+j] = A[(row+i)*lda+col+j];
//     }
//   }
//   return;
// }

// static inline void populate_sub_b(int lda, int M, int N, int row, int col, double* B){
//   for(int i = 0; i < M; i++){
//     for(int j = 0; j < N; j++){
//       B_block[i*N+j] = B[(row+i)*lda+col+j];
//     }
//   }
// }

// static inline void helper(int LDA, int block_m, int block_n, int block_k, \
//                           int i, int j, int k, )


static inline void helper(int LDA, int i, int j, int k, \
                          int block_m, int block_n, int block_k, \
                          double* A, double* B, double* C){
  populate_sub(LDA, block_m, block_k, i, k, A, A_block);
  populate_sub(LDA, block_k, block_n, k, j, B, B_block);
  do_block_copied(block_k, block_n, LDA,\
      block_m, block_n, block_k, \
      A_block, B_block, C + i*LDA + j);
}

/**
 * Creating a copy of block of B in B_block
 * so as to fit that in L2 completly
 **/

static inline void do_blockL2 (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  int M__ = (M/L2_M)*L2_M;
  int N__ = (N/L2_N)*L2_N;
  int K__ = (K/L2_K)*L2_K;

  int block_m = L2_M;
  int block_n = L2_N;
  int block_k = L2_K;

  int i = 0;
  for (i = 0; i < M__; i += L2_M){
    int j = 0;
    for (j = 0; j < N__; j += L2_N){
      int k = 0;
      for (k = 0; k < K__; k += L2_K){
        helper(lda, i, j, k, block_m, block_n, block_k, A, B, C);
      }
      if(k < K){
        block_k = K-k;
        helper(lda, i, j, k, block_m, block_n, block_k, A, B, C);
      }
    }
    if(j < N){
      block_n = N-j;
      block_k = L2_K;
      int k = 0;
      for (k = 0; k < K__; k += L2_K){
        helper(lda, i, j, k, block_m, block_n, block_k, A, B, C);
        }
        block_k = K-k;
        helper(lda, i, j, k, block_m, block_n, block_k, A, B, C);
    }
    
  }
  if(i < M){
    block_m = M-i;
    block_n = L2_N;
    block_k = L2_K;
    int j = 0;
    for (j = 0; j < N__; j += L2_N){
      int k = 0;
        for (k = 0; k < K__; k += L2_K){
          helper(lda, i, j, k, block_m, block_n, block_k, A, B, C);
        }
        if(k < K){
          block_k = K-k;
          helper(lda, i, j, k, block_m, block_n, block_k, A, B, C);
        }
        
      }
      if(j < N){
      block_n = N-j;
      block_k = L2_K;
      int k = 0;
      for (k = 0; k < K__; k += L2_K){
        helper(lda, i, j, k, block_m, block_n, block_k, A, B, C);
        }

        if(k < K){
          block_k = K-k;
          helper(lda, i, j, k, block_m, block_n, block_k, A, B, C);
        }
      }
  }


}


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
  A_block = (double*) malloc (L2_M * L2_K * sizeof(double));
  B_block = (double*) malloc (L2_K * L2_N * sizeof(double));
#ifdef TRANSPOSE
  for (int i = 0; i < lda; ++i)
    for (int j = i+1; j < lda; ++j) {
        double t = B[i*lda+j];
        B[i*lda+j] = B[j*lda+i];
        B[j*lda+i] = t;
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
	int M = min (BLOCK_SIZEL3, lda-i);
	int N = min (BLOCK_SIZEL3, lda-j);
	int K = min (BLOCK_SIZEL3, lda-k);

	/* Perform individual block dgemm */
#ifdef TRANSPOSE
	do_blockL2(lda, M, N, K, A + i*lda + k, B + j*lda + k, C + i*lda + j);
#else
	do_blockL2(lda, M, N, K, A + i*lda + k, B + k*lda + j, C + i*lda + j);
#endif
      }
#if TRANSPOSE
  for (int i = 0; i < lda; ++i)
    for (int j = i+1; j < lda; ++j) {
        double t = B[i*lda+j];
        B[i*lda+j] = B[j*lda+i];
        B[j*lda+i] = t;
    }
#endif
}
