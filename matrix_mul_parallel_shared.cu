#include <cstdio>
#include <cstdlib>
#include <cassert>

#define TILE_WIDTH 8

__global__ void gpu_matrix_mult(int *A, int *B, int *C, int row, int width, int col) {
    
    __shared__ int sA[TILE_WIDTH][TILE_WIDTH];
    __shared__ int sB[TILE_WIDTH][TILE_WIDTH];

    int r = blockDim.y * blockIdx.y + threadIdx.y + blockDim.y * blockIdx.z;
    int c = blockDim.x * blockIdx.x + threadIdx.x + blockDim.y * blockIdx.z;
    int Cval = 0;

    sA[threadIdx.y][threadIdx.x] = 0;
    sB[threadIdx.x][threadIdx.y] = 0;

    for (int tileNum = 0; tileNum < width/TILE_WIDTH; tileNum++){
        if(r < row && (tileNum) * TILE_WIDTH + threadIdx.x < width) {
            sA[threadIdx.y][threadIdx.x] = A[r * width + (tileNum) * TILE_WIDTH + threadIdx.x];
        }
        __syncthreads();
        if(c < col && (tileNum) * TILE_WIDTH + threadIdx.y < width) {
            sB[threadIdx.y][threadIdx.x] = B[((tileNum) * TILE_WIDTH + threadIdx.y) * col + c];
        }
        __syncthreads();

        for(int j = 0; j < TILE_WIDTH ; ++j) {
            Cval += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        } 
        __syncthreads();
    }

    if(r < row && c < col){
        C[r * col + c] = Cval;
    }

}

void verify_results(int *A, int *B, int* C, int row, int width, int col) {
    for(int i = 0; i< row; ++i){
        for(int j = 0; j< col; ++j) {
            int value = 0;
            for(int k = 0; k < width ; ++k) {
                value += A[i * width + k] * B[k * col + j];
            }
            printf("%d:%d\n", value, C[i * col + j]);
        }
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
    
    int threads = 8;
    int blockwidth = 2;
    int grid_rows = (row + threads-1)/(threads);
    int grid_cols = (col + threads-1)/(threads);

    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(threads, threads, blockwidth);

    cudaEventRecord(start, 0);
    gpu_matrix_mult<<<dimGrid, dimBlock>>>(deviceA, deviceB, deviceC, row, width, col);
    
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    
    cudaMemcpy(hC, deviceC, sizeof(int) * row * col, cudaMemcpyDeviceToHost);
    
    float gpu_elapsed_time_ms;
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    // verify_results(hA, hB, hC, row, width, col);
    printf("Time Elapsed on GPU Matrix Multiplication: %f\n", gpu_elapsed_time_ms);

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
    cudaFreeHost(hA);
    cudaFreeHost(hB);
    cudaFreeHost(hC);
    return 0;
}
