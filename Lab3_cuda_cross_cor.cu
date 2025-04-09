// cuda_cross_correlation.cu
#include <iostream>
#include <cstdlib>
#include <ctime>

#define INPUT_SIZE 256
#define KERNEL_SIZE 8
#define OUTPUT_SIZE (INPUT_SIZE - KERNEL_SIZE + 1)

__constant__ float d_kernel[KERNEL_SIZE * KERNEL_SIZE];

__global__ void crossCorrelateKernel(float* input, float* output, int width) {
    // Calculate output indices
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x >= OUTPUT_SIZE || out_y >= OUTPUT_SIZE) return;

    float sum = 0.0f;
    for (int m = 0; m < KERNEL_SIZE; ++m) {
        for (int n = 0; n < KERNEL_SIZE; ++n) {
            int in_x = out_x + n;
            int in_y = out_y + m;
            sum += input[in_y * width + in_x] * d_kernel[m * KERNEL_SIZE + n];
        }
    }

    output[out_y * OUTPUT_SIZE + out_x] = sum;
}

void generateRandomMatrix(float* matrix, int size) {
    for (int i = 0; i < size; ++i)
        matrix[i] = static_cast<float>(rand()) / RAND_MAX * 2 - 1;
}

int main() {
    srand(time(0));

    size_t inputBytes = INPUT_SIZE * INPUT_SIZE * sizeof(float);
    size_t outputBytes = OUTPUT_SIZE * OUTPUT_SIZE * sizeof(float);
    size_t kernelBytes = KERNEL_SIZE * KERNEL_SIZE * sizeof(float);

    float *h_input = new float[INPUT_SIZE * INPUT_SIZE];
    float *h_output = new float[OUTPUT_SIZE * OUTPUT_SIZE];
    float *h_kernel = new float[KERNEL_SIZE * KERNEL_SIZE];

    generateRandomMatrix(h_input, INPUT_SIZE * INPUT_SIZE);
    generateRandomMatrix(h_kernel, KERNEL_SIZE * KERNEL_SIZE);

    float *d_input, *d_output;
    cudaMalloc(&d_input, inputBytes);
    cudaMalloc(&d_output, outputBytes);

    cudaMemcpy(d_input, h_input, inputBytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(d_kernel, h_kernel, kernelBytes);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((OUTPUT_SIZE + 15) / 16, (OUTPUT_SIZE + 15) / 16);

    // 🎯 CUDA timing starts here
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // 🚀 Launch the kernel
    crossCorrelateKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, INPUT_SIZE);
    cudaDeviceSynchronize();  // Wait for kernel to complete

    // 🛑 CUDA timing ends here
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "CUDA cross-correlation complete." << std::endl;
    std::cout << "Execution Time: " << milliseconds << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy result back to host
    cudaMemcpy(h_output, d_output, outputBytes, cudaMemcpyDeviceToHost);

    // Cleanup
    delete[] h_input;
    delete[] h_output;
    delete[] h_kernel;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
