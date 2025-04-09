// sequential_cross_correlation.cpp
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>

#define INPUT_SIZE 256
#define KERNEL_SIZE 8
#define OUTPUT_SIZE (INPUT_SIZE - KERNEL_SIZE + 1)

using namespace std;

void generateRandomMatrix(vector<vector<float>> &matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            matrix[i][j] = static_cast<float>(rand()) / RAND_MAX * 2 - 1;
}

void crossCorrelate(
    const vector<vector<float>> &input,
    const vector<vector<float>> &kernel,
    vector<vector<float>> &output
) {
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        for (int j = 0; j < OUTPUT_SIZE; ++j) {
            float sum = 0.0f;
            for (int m = 0; m < KERNEL_SIZE; ++m) {
                for (int n = 0; n < KERNEL_SIZE; ++n) {
                    sum += input[i + m][j + n] * kernel[m][n];
                }
            }
            output[i][j] = sum;
        }
    }
}


int main() {
    srand(time(0));
    vector<vector<float>> input(INPUT_SIZE, vector<float>(INPUT_SIZE));
    vector<vector<float>> kernel(KERNEL_SIZE, vector<float>(KERNEL_SIZE));
    vector<vector<float>> output(OUTPUT_SIZE, vector<float>(OUTPUT_SIZE, 0.0f));

    generateRandomMatrix(input, INPUT_SIZE, INPUT_SIZE);
    generateRandomMatrix(kernel, KERNEL_SIZE, KERNEL_SIZE);

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    crossCorrelate(input, kernel, output);

    // End timing
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "Sequential cross-correlation complete." << std::endl;
    std::cout << "Execution Time: " << elapsed.count() << " ms" << std::endl;

    return 0;
}
