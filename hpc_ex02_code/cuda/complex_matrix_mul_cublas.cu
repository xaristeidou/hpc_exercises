#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define N 1024
#define BLOCK_SIZE 16

//Combination kernels

// E = AC - BD
__global__ void compute_E(const double *AC, const double *BD, double *E, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        E[idx] = AC[idx] - BD[idx];
}

// F = AD + BC
__global__ void compute_F(const double *AD, const double *BC, double *F, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        F[idx] = AD[idx] + BC[idx];
}

// Host reference
void complex_matrix_mul_host(const double *A, const double *B,
                             const double *C, const double *D,
                             double *E, double *F, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum_ac = 0.0, sum_bd = 0.0;
            double sum_ad = 0.0, sum_bc = 0.0;
            for (int k = 0; k < n; k++) {
                sum_ac += A[i * n + k] * C[k * n + j];
                sum_bd += B[i * n + k] * D[k * n + j];
                sum_ad += A[i * n + k] * D[k * n + j];
                sum_bc += B[i * n + k] * C[k * n + j];
            }
            E[i * n + j] = sum_ac - sum_bd;
            F[i * n + j] = sum_ad + sum_bc;
        }
    }
}

// initialize random values
void init_random(double *mat, int size) {
    for (int i = 0; i < size; i++)
        mat[i] = (double)rand() / RAND_MAX;
}

int main(void) {
    int n = N;
    size_t bytes = (size_t)n * n * sizeof(double);
    int total = n * n;

    // Host allocation
    double *h_A = (double *)malloc(bytes);
    double *h_B = (double *)malloc(bytes);
    double *h_C = (double *)malloc(bytes);
    double *h_D = (double *)malloc(bytes);
    double *h_E = (double *)malloc(bytes);
    double *h_F = (double *)malloc(bytes);

    srand(42);
    init_random(h_A, total);
    init_random(h_B, total);
    init_random(h_C, total);
    init_random(h_D, total);

    // Device allocation
    double *d_A, *d_B, *d_C, *d_D;
    double *d_AC, *d_BD, *d_AD, *d_BC;
    double *d_E, *d_F;

    cudaMalloc(&d_A,  bytes);
    cudaMalloc(&d_B,  bytes);
    cudaMalloc(&d_C,  bytes);
    cudaMalloc(&d_D,  bytes);
    cudaMalloc(&d_AC, bytes);
    cudaMalloc(&d_BD, bytes);
    cudaMalloc(&d_AD, bytes);
    cudaMalloc(&d_BC, bytes);
    cudaMalloc(&d_E,  bytes);
    cudaMalloc(&d_F,  bytes);

    // Copy inputs Host -> Device 
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D, bytes, cudaMemcpyHostToDevice);

    // cuBLAS setup 
    cublasHandle_t handle;
    cublasCreate(&handle);

    /*
     * cuBLAS uses column-major order, but our matrices are row-major.
     * For row-major A (m×k) * C (k×n), we compute:
     *   C^T * A^T = (A*C)^T   which is (A*C) stored in row-major.
     *
     * So we call cublasDgemm with:
     *   transa = CUBLAS_OP_N, transb = CUBLAS_OP_N
     *   m = n (cols of result), n = n (rows of result), k = n
     *   A_cublas = d_C, B_cublas = d_A   (swapped)
     */
    double alpha = 1.0;
    double beta  = 0.0;

    //  GPU timing 
    cudaEvent_t start_dev, stop_dev;
    cudaEventCreate(&start_dev);
    cudaEventCreate(&stop_dev);
    cudaEventRecord(start_dev);

    /* Compute AC */
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_C, n, d_A, n, &beta,  d_AC, n);

    /* Compute BD */
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_D, n, d_B, n, &beta,  d_BD, n);
    
    /* Compute AD */
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_D, n, d_A, n, &beta,  d_AD, n);

    /* Compute BC */
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_C, n, d_B, n, &beta,  d_BC, n);

    /* Combine: E = AC - BD,  F = AD + BC */
    int grid = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;
    compute_E<<<grid, BLOCK_SIZE>>>(d_AC, d_BD, d_E, total);
    compute_F<<<grid, BLOCK_SIZE>>>(d_AD, d_BC, d_F, total);

    cudaEventRecord(stop_dev);
    cudaEventSynchronize(stop_dev);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start_dev, stop_dev);
    cudaEventDestroy(start_dev);
    cudaEventDestroy(stop_dev);

    //  Copy results Device -> Host 
    cudaMemcpy(h_E, d_E, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_F, d_F, bytes, cudaMemcpyDeviceToHost);

    //  Host computation (timed) 
    double *h_E_ref = (double *)malloc(bytes);
    double *h_F_ref = (double *)malloc(bytes);

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    complex_matrix_mul_host(h_A, h_B, h_C, h_D, h_E_ref, h_F_ref, n);
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double cpu_ms = (ts_end.tv_sec - ts_start.tv_sec) * 1e3 +
                    (ts_end.tv_nsec - ts_start.tv_nsec) / 1e6;

    //  Verification 
    double max_err_E = 0.0, max_err_F = 0.0;
    for (int i = 0; i < total; i++) {
        double err_e = fabs(h_E[i] - h_E_ref[i]);
        double err_f = fabs(h_F[i] - h_F_ref[i]);
        if (err_e > max_err_E)
        {
            max_err_E = err_e;
        }
        if (err_f > max_err_F)
        {
            max_err_F = err_f;
        }
    }

    printf("Complex matrix multiplication – cuBLAS (%d x %d)\n", n, n);
    printf("Max absolute error E: %e\n", max_err_E);
    printf("Max absolute error F: %e\n", max_err_F);

    if (max_err_E < 1e-6 && max_err_F < 1e-6)
    {
        printf("Error in calculations is small.\n");
    }
    else
    {
        printf("Error in calculations is big.\n");
    }

    /* FLOP count:
     *   4 real matrix multiplications (AC, BD, AD, BC): 4 x 2N^3 = 8N^3
     *   2 element-wise operations (E=AC-BD, F=AD+BC):   2N^2
     *   Total: 8N^3 + 2N^2
     */
    double flops = 8.0 * (double)n * n * n + 2.0 * (double)n * n;

    printf("\nFLOP count: %.3e\n", flops);
    printf("\nDevice (GPU, cuBLAS) time: %.3f ms\n", gpu_ms);
    printf("Device throughput: %.3f GFLOP/s\n", flops / (gpu_ms * 1e6));
    printf("\nHost (CPU) time: %.3f ms\n", cpu_ms);
    printf("Host throughput: %.3f GFLOP/s\n", flops / (cpu_ms * 1e6));
    printf("\nSpeedup: %.2fx\n", cpu_ms / gpu_ms);

    //  Cleanup 
    cublasDestroy(handle);

    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);
    free(h_E);
    free(h_F);
    free(h_E_ref);
    free(h_F_ref);

    cudaFree(d_A);  
    cudaFree(d_B);
    cudaFree(d_C);  
    cudaFree(d_D);
    cudaFree(d_AC); 
    cudaFree(d_BD);
    cudaFree(d_AD); 
    cudaFree(d_BC);
    cudaFree(d_E);  
    cudaFree(d_F);

    return 0;
}
