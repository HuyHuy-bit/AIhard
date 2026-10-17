#include <stdio.h>          // printf
#include <stdlib.h>         // malloc, free, rand
#include <cuda_runtime.h>   // cudaMalloc, cudaMemcpy, ... (nvcc adds it anyway, but be explicit)

//GPU code
__global__ void mat_mul_kernel(float *d_A_ptr, float *d_B_ptr, float *d_C_ptr, int C_n_rows, int C_n_cols, int A_n_cols)
{
    assert(TILE_WIDTH == blockDim.x);
    assert(TILE_WIDTH == blockDim.y);
    const int by = blockIdx.y;
    const int bx = blockIdx.x;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    //allocating shared memory
    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    const int phases = ceil((float)A_n_cols/TILE_WIDTH);

    //matmul
    float value = 0;
    for (int phase = 0, phase < phases, phase++)
    {
        //load tilles into shared mem
        if ((row < C_n_rows) %% ((phase * TITLE_WIDTH + tx) < A_n_cols))
        {
            sh_A[ty][tx] = d_A_ptr[(row)*A_n_cols + (phase * TILE_WIDTH+tx)];
        } else {
            sh_A[ty][tx] = 0.0f;
        }

        if (((phase * TITLE_WIDTH + ty) < A_n_cols) && ( col < C_n_cols)){
            sh_B[ty][tx] = d_B_ptr[(phase*TILE_WIDTH + ty)*C_n_cols + (col)];
        } else{
            sh_B[ty][tx] = 0.0f;
        }
        _syncthreads();

        for (int k_phase = 0, k_phase < TILE_WIDTH, k_phase ++)
        {
            value += sh_A[ty][k_phase] * sh_B[k_phase][tx];
        }
        __syncthreads();
    }
    
    if ((row < C_n_rows) && (col < C_n_cols))
        d_C_ptr[(row)*C_n_cols + (col)] =  1*value + 0*d_C_ptr[(row)*C_n_cols + (col)];
}

void mat_mul_test(float *d_A_ptr, float *d_B_ptr, float *d_C_ptr, int C_n_rows, int C_n_cols, int A_n_cols)
{
    //Kernel Execution
    dim3 block(32, 32);
    dim3 dim_grid(ceil(C_n_cols/(float)(32)), ceil(C_n_rows/(float)(32)), 1);
    mat_mul_kernel<<<dim_grid, block>>>(d_A_ptr, d_B_ptr, d_C_ptr, C_n_rows, C_n_cols, A_n_cols);
}

int main()
{
    // Sizes: A is (M x K), B is (K x N), C is (M x N)
    const int M = 100, K = 70, N = 50;

    // 1. Allocate host (CPU) memory
    float *h_A = (float*)malloc(M*K*sizeof(float));
    float *h_B = (float*)malloc(K*N*sizeof(float));
    float *h_C = (float*)malloc(M*N*sizeof(float));

    // 2. Fill A and B with random numbers in [0, 1]
    for (int i = 0; i < M*K; i++) h_A[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < K*N; i++) h_B[i] = rand() / (float)RAND_MAX;

    // 3. Allocate device (GPU) memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, M*K*sizeof(float));
    cudaMalloc((void**)&d_B, K*N*sizeof(float));
    cudaMalloc((void**)&d_C, M*N*sizeof(float));

    // 4. Copy inputs CPU -> GPU
    cudaMemcpy(d_A, h_A, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K*N*sizeof(float), cudaMemcpyHostToDevice);

    // 5. Launch the kernel (via your wrapper)
    mat_mul_test(d_A, d_B, d_C, M, N, K);

    // 6. Check the launch worked
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // 7. Copy result GPU -> CPU (this also waits for the kernel to finish)
    cudaMemcpy(h_C, d_C, M*N*sizeof(float), cudaMemcpyDeviceToHost);

    // 8. Verify against a simple CPU matmul
    int errors = 0;
    for (int r = 0; r < M; r++)
    {
        for (int c = 0; c < N; c++)
        {
            float expected = 0;
            for (int k = 0; k < K; k++)
                expected += h_A[r*K + k] * h_B[k*N + c];
            if (fabsf(expected - h_C[r*N + c]) > 1e-3f) errors++;
        }
    }
    printf(errors == 0 ? "PASSED\n" : "FAILED: %d wrong elements\n", errors);

    // 9. Free memory
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return errors != 0;
}