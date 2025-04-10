#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 256
#define KERNEL_SIZE 8
#define OUTPUT_SIZE (INPUT_SIZE - KERNEL_SIZE + 1)

__constant__ float d_kernel[KERNEL_SIZE * KERNEL_SIZE];

// Function to initialize matrix with random values between -1 and 1
void initializeMatrix(float* matrix, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;  // Random float between -1 and 1
    }
}

__global__ void crossCorrelateKernel(float* input, float* output) {
    // Kernel implementation goes here
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x < OUTPUT_SIZE && out_y < OUTPUT_SIZE) {
        float sum = 0.0f;
        for (int m = 0; m < KERNEL_SIZE; m++) {
            for (int n = 0; n < KERNEL_SIZE; n++) {
                sum += input[(out_y + m) * INPUT_SIZE + (out_x + n)] * d_kernel[m * KERNEL_SIZE + n];
            }
        }
        output[out_y * OUTPUT_SIZE + out_x] = sum;
    }
}

int main() {
    // Host matrices
    float* h_input = (float*)malloc(INPUT_SIZE * INPUT_SIZE * sizeof(float));
    float* h_kernel = (float*)malloc(KERNEL_SIZE * KERNEL_SIZE * sizeof(float));
    float* h_output = (float*)malloc(OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float));

    // Initialize random seed
    srand((unsigned int)time(NULL));

    // Initialize matrices on the host
    initializeMatrix(h_input, INPUT_SIZE, INPUT_SIZE);
    initializeMatrix(h_kernel, KERNEL_SIZE, KERNEL_SIZE);

    // Device memory allocation
    float *d_input, *d_output;
    cudaMalloc(&d_input, INPUT_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_output, OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float));

    // Copy the input matrix to device memory
    cudaMemcpy(d_input, h_input, INPUT_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    
    // Copy the kernel matrix to constant memory (this assumes you have the kernel set up properly)
    cudaMemcpyToSymbol(d_kernel, h_kernel, KERNEL_SIZE * KERNEL_SIZE * sizeof(float));

    // Launch the kernel (configure grid and block sizes as needed)
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((OUTPUT_SIZE + 15) / 16, (OUTPUT_SIZE + 15) / 16);
    crossCorrelateKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output);

    // Copy the result back to the host
    cudaMemcpy(h_output, d_output, OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Print or verify the output
    // (Add your verification code here)

    // Cleanup
    free(h_input);
    free(h_kernel);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
