#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//#include <iostream>
//if run on VS Code for Check, make sure to include the tasks.json

#define INPUT_SIZE 256
#define KERNEL_SIZE 8
#define OUTPUT_SIZE (INPUT_SIZE - KERNEL_SIZE + 1)

float input[INPUT_SIZE][INPUT_SIZE];
float kernel[KERNEL_SIZE][KERNEL_SIZE];
float output[OUTPUT_SIZE][OUTPUT_SIZE];

// Fill with random floats between -1 and 1
void initializeMatrix(float* mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
    }
}

// Perform 2D cross-correlation
void crossCorrelateCPU() {
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            float sum = 0.0f;
            for (int m = 0; m < KERNEL_SIZE; m++) {
                for (int n = 0; n < KERNEL_SIZE; n++) {
                    sum += input[i + m][j + n] * kernel[m][n];
                }
            }
            output[i][j] = sum;
        }
    }
}

int main() {
    srand((unsigned int)time(NULL));

    // Initialize input and kernel
    initializeMatrix(&input[0][0], INPUT_SIZE, INPUT_SIZE);
    initializeMatrix(&kernel[0][0], KERNEL_SIZE, KERNEL_SIZE);

    // Start timer
    clock_t start = clock();

    // Run cross-correlation
    crossCorrelateCPU();

    // Stop timer
    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    printf("CPU Execution Time: %.2f ms\n", time_spent);

    // Print top 4x4 part of result
    printf("Output (Top 4x4):\n");
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%6.2f ", output[i][j]);
        }
        printf("\n");
    }

    return 0;
}
