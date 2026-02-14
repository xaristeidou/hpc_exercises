#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 1024
#define BLOCK_SIZE 16

// CUDA kernel: computes E = A*C - B*D  and  F = A*D + B*C
// All matrices are N x N, stored in row-major order.
__global__ void complex_matrix_mul(const double *A, const double *B,
                                   const double *C, const double *D,
                                   double *E, double *F, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        double sum_ac = 0.0;
        double sum_bd = 0.0;
        double sum_ad = 0.0;
        double sum_bc = 0.0;

        for (int k = 0; k < n; k++) {
            double a = A[row * n + k];
            double b = B[row * n + k];
            double c = C[k * n + col];
            double d = D[k * n + col];

            sum_ac += a * c;
            sum_bd += b * d;
            sum_ad += a * d;
            sum_bc += b * c;
        }

        E[row * n + col] = sum_ac - sum_bd;
        F[row * n + col] = sum_ad + sum_bc;
    }
}

// Host reference computation for verification
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

// Fill matrix with random values in [0, 1)
void init_random(double *mat, int size) {
    for (int i = 0; i < size; i++)
        mat[i] = (double)rand() / RAND_MAX;
}

int main(void) {
    size_t bytes = N * N * sizeof(double);

    // Host allocation and initialization
    double *h_A = (double *)malloc(bytes);
    double *h_B = (double *)malloc(bytes);
    double *h_C = (double *)malloc(bytes);
    double *h_D = (double *)malloc(bytes);
    double *h_E = (double *)malloc(bytes);
    double *h_F = (double *)malloc(bytes);

    srand(42);
    init_random(h_A, N * N);
    init_random(h_B, N * N);
    init_random(h_C, N * N);
    init_random(h_D, N * N);

    // Device allocation
    double *d_A, *d_B, *d_C, *d_D, *d_E, *d_F;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMalloc(&d_D, bytes);
    cudaMalloc(&d_E, bytes);
    cudaMalloc(&d_F, bytes);

    // Copy inputs Host -> Device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_D, h_D, bytes, cudaMemcpyHostToDevice);

    // Launch kernel (timed)
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start_dev, stop_dev;
    cudaEventCreate(&start_dev);
    cudaEventCreate(&stop_dev);

    cudaEventRecord(start_dev);
    complex_matrix_mul<<<blocks, threads>>>(d_A, d_B, d_C, d_D, d_E, d_F, N);
    cudaEventRecord(stop_dev);
    cudaGetLastError();
    cudaEventSynchronize(stop_dev);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start_dev, stop_dev);
    cudaEventDestroy(start_dev);
    cudaEventDestroy(stop_dev);

    // Copy results Device -> Host
    cudaMemcpy(h_E, d_E, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_F, d_F, bytes, cudaMemcpyDeviceToHost);

    // Host computation (timed)
    double *h_E_ref = (double *)malloc(bytes);
    double *h_F_ref = (double *)malloc(bytes);

    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);
    complex_matrix_mul_host(h_A, h_B, h_C, h_D, h_E_ref, h_F_ref, N);
    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double cpu_ms = (ts_end.tv_sec - ts_start.tv_sec) * 1e3 +
                    (ts_end.tv_nsec - ts_start.tv_nsec) / 1e6;

    double max_err_E = 0.0, max_err_F = 0.0;
    for (int i = 0; i < N * N; i++) {
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

    printf("Complex matrix multiplication (%d x %d)\n", N, N);
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
    double flops = 8.0 * (double)N * N * N + 2.0 * (double)N * N;

    printf("\nFLOP count: %.3e\n", flops);
    printf("\nDevice (GPU) time: %.3f ms\n", gpu_ms);
    printf("Device throughput:  %.3f GFLOP/s\n", flops / (gpu_ms * 1e6));
    printf("\nHost   (CPU) time: %.3f ms\n", cpu_ms);
    printf("Host   throughput:  %.3f GFLOP/s\n", flops / (cpu_ms * 1e6));
    printf("\nSpeedup: %.2fx\n", cpu_ms / gpu_ms);

    // Cleanup
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
    cudaFree(d_E);
    cudaFree(d_F);

    return 0;
}
