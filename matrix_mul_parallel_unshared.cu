#include <cstdio>
#include <cstdlib>
#include <cassert>

#define TILE_WIDTH 16

__global__ void gpu_matrix_mult(int *A, int *B, int *C, int row, int width, int col) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if(c < row && c < col) {
        for(int i = 0; i < width ; ++i) {
            sum += A[r * width + i] * B[i * col + c];
        }
        C[r * col + c] = sum;
    }    
}

void initialize_matrix(int *mtx, int m, int n) {
    for(int i = 0; i< m*n ; ++i) {
        mtx[i] = rand()%10;   
    }
}
int main() {
    
    int row = 1 << 13;
    int width = 1 << 12;
    int col = 1 << 11;

    int *hA, *hB, *hC;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMallocHost(&hA, row * width * sizeof(int));
    cudaMallocHost(&hB, width * col * sizeof(int));
    cudaMallocHost(&hC, row * col * sizeof(int));

    initialize_matrix(hA, row, width);
    initialize_matrix(hB, width, col);

    int *deviceA, *deviceB, *deviceC;
    cudaMalloc((void**)&deviceA, sizeof(int) * row * width);
    cudaMalloc((void**)&deviceB, sizeof(int) * col * width);
    cudaMalloc((void**)&deviceC, sizeof(int) * row * col);
    
    cudaMemcpy(deviceA, hA, sizeof(int) * row * width, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceB, hB, sizeof(int) * width * col, cudaMemcpyHostToDevice);
    
    int threadX = 32;
    int threadY = 16;
    int grid_rows = (row + threadY -1)/threadY;
    int grid_cols = (col + threadX -1)/threadX;

    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(threadX, threadY);

    cudaEventRecord(start, 0);
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, row, width, col);
    cudaMemcpy(hC, deviceC, sizeof(int) * row * col, cudaMemcpyDeviceToHost);
    
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float gpu_elapsed_time_ms;
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);

    printf("Time Elapsed on GPU Matrix Multiplication: %f\n", gpu_elapsed_time_ms);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    cudaFreeHost(hA);
    cudaFreeHost(hB);
    cudaFreeHost(hC);
    return 0;
}
