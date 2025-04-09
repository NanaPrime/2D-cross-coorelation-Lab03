#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define INPUT_SIZE 256
#define KERNEL_SIZE 8
#define OUTPUT_SIZE (INPUT_SIZE - KERNEL_SIZE + 1)

__constant__ float d_kernel[KERNEL_SIZE * KERNEL_SIZE];

__global__ void crossCorrelate(float* input, float* output) {
    int out_row = threadIdx.y;
    int out_col = threadIdx.x;

    if (out_row < OUTPUT_SIZE && out_col < OUTPUT_SIZE) {
        float sum = 0.0f;
        for (int m = 0; m < KERNEL_SIZE; m++) {
            for (int n = 0; n < KERNEL_SIZE; n++) {
                int in_row = out_row + m;
                int in_col = out_col + n;
                sum += input[in_row * INPUT_SIZE + in_col] *
                       d_kernel[m * KERNEL_SIZE + n];
            }
        }
        output[out_row * OUTPUT_SIZE + out_col] = sum;
    }
}

int main() {
    float h_input[INPUT_SIZE][INPUT_SIZE], h_kernel[KERNEL_SIZE][KERNEL_SIZE], h_output[OUTPUT_SIZE][OUTPUT_SIZE];
    float *d_input, *d_output;

    // Generate random float values between -1 and 1 for input and kernel
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            h_input[i][j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }
    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            h_kernel[i][j] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }
    }

    // Allocate device memory
    cudaMalloc(&d_input, INPUT_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_output, OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, h_input, INPUT_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_kernel, h_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));

    // CUDA timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    dim3 threadsPerBlock(OUTPUT_SIZE, OUTPUT_SIZE);  // 9x9 for 16x16 input and 8x8 kernel
    crossCorrelate<<<1, threadsPerBlock>>>(d_input, d_output);
    cudaDeviceSynchronize();

    // End CUDA timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("CUDA Execution Time: %.4f ms\n", milliseconds);

    // Copy result back
    cudaMemcpy(h_output, d_output, OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Print first few results
    printf("Cross-Correlation Output (First 4x4 Elements):\n");
    for (int i = 0; i < 4 && i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < 4 && j < OUTPUT_SIZE; j++) {
            printf("%6.2f ", h_output[i][j]);
        }
        printf("\n");
    }

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
