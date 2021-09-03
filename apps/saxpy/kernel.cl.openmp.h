static void saxpy(float* Z, float A, float* X, float* Y, BRISBANE_OPENMP_KERNEL_ARGS) {
  size_t i;
#pragma omp parallel for shared(Z, A, X, Y) private(i)
  BRISBANE_OPENMP_KERNEL_BEGIN(i)
  Z[i] = A * X[i] + Y[i];
  BRISBANE_OPENMP_KERNEL_END
}

