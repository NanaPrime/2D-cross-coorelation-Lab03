# 2D-cross-coorelation-Lab03

This is for AI Hardware and Programming Class

Lab 3: 2D Cross-Correlation (Convolution) Using CUDA on Jetson Nano
This lab requires you to implement a CUDA-based 2D cross-correlation operation between an input matrix and a filter kernel. The implementation must use statically allocated shared memory and no padding. The output matrix should be computed such that the element at position (i,j) is:

out[i][j] = Σ Σ in[i+m][j+n] * kernel[m][n]
                m n
where in is the input signal, kernel is the filter kernel, and (m,n) range over the kernel dimensions
You will need to prepare two separate implementations:
A sequential (CPU) version of the cross-correlation as a baseline.
A parallel (CUDA) version using shared memory for optimization.
You will provide your own input signal and filter kernel. The input matrix is of size 256 x 256, and the kernel is of size 8x8. For simplicity, both the input and kernel values are floating-point within the range of (-1, 1). The values of the input and kernel can be generated randomly. The size of the output array will depend on the sizes of the input signal and the kernel. (Hint: Think about how the kernel size affects the output size.)
Sequential Implementation (CPU): Write a sequential version of the cross-correlation in C/C++. This will serve as a baseline for correctness and performance comparison. Ensure the implementation is correct and matches the expected output.
Parallel Implementation (CUDA): Write a CUDA kernel to perform the 2D cross-correlation. No padding should be used, meaning the output signal will be smaller than the input signal.


Submission Guidelines
Code:
Submit two separate (files) implementation codes:
A sequential (CPU) version of the cross-correlation.
A parallel (CUDA) version of the cross-correlation.
Both the CUDA kernel and host code are needed for the parallel implementation.
Include comments in the code to explain key steps.
Verification:
Verify that both the sequential and parallel implementations produce the same correct results on Jetson Nano.
Performance Analysis:
Analyze the performance of your implementation, including execution time (Please refer to the command in the text file shared in the handouts).


Compare the time consumption of the CUDA implementation with the sequential (CPU) implementation as a baseline.
Write a report:
Write a report to describe the above procedure, including the code, verification, and performance analysis.
You can refer to the following example about 2D matrix multiplication to learn the dim3 API for designing your cross-correlation CUDA code

- Oracle 
