/*
 *  A simple blocked implementation of matrix multiply
 *  Provided by Jim Demmel at UC Berkeley
 *  Modified by Scott B. Baden at UC San Diego to
 *    Enable user to select one problem size only via the -n option
 *    Support CBLAS interface
 */

const char* dgemm_desc = "Simple blocked dgemm.";

#if !defined(BLOCK_SIZEL1)
#define BLOCK_SIZEL1 37
#endif

#if !defined(BLOCK_SIZEL2)
#define BLOCK_SIZEL2 120
#endif

#if !defined(BLOCK_SIZEL3)
#define BLOCK_SIZEL3 750
#endif


#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
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

static void do_blockL1 (int lda, int M, int N, int K, double* A, double* B, double* C)
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
#ifdef TRANSPOSE
  do_block(lda, M_, N_, K_, A + i*lda + k, B + j*lda + k, C + i*lda + j);
#else
  do_block(lda, M_, N_, K_, A + i*lda + k, B + k*lda + j, C + i*lda + j);
#endif
      }
}

static void do_blockL2 (int lda, int M, int N, int K, double* A, double* B, double* C)
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
#ifdef TRANSPOSE
  do_blockL1(lda, M_, N_, K_, A + i*lda + k, B + j*lda + k, C + i*lda + j);
#else
  do_blockL1(lda, M_, N_, K_, A + i*lda + k, B + k*lda + j, C + i*lda + j);
#endif
      }
}


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in row-major order
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double* C)
{
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
