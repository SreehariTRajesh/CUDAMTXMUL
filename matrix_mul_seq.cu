#include <cstdio>
#include <cstdlib>


void initialize_matrix(int *mtx, int m, int n) {
    for(int i = 0; i< m*n ; ++i) {
        mtx[i] = rand()%10;   
    }
}

void cpu_matrix_mul(int *A, int *B, int* C, int row, int width, int col) {
    for(int i = 0; i< row; ++i){
        for(int j = 0; j< col; ++j) {
            int value = 0;
            for(int k = 0; k < width ; ++k) {
                value += A[i * width + k] * B[k * col + j];
            }
            C[i * col + j] = value;
        }
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


    cudaEventRecord(start, 0);
    cpu_matrix_mul(hA, hB, hC, row, width, col);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float gpu_elapsed_time_ms;
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    
    printf("Time Elapsed on GPU Matrix Multiplication: %f\n", gpu_elapsed_time_ms);

    cudaFreeHost(hA);
    cudaFreeHost(hB);
    cudaFreeHost(hC);
    return 0;
}
