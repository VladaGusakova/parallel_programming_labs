#include <iostream>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

void read_matrix(const string& filename, vector<int>& matrix, int N) {
    ifstream in(filename);
    if (!in) throw runtime_error("Cannot open " + filename);
    for (int i = 0; i < N * N; ++i) {
        in >> matrix[i];
    }
}

void write_matrix(const string& filename, const vector<int>& matrix, int N) {
    ofstream out(filename);
    if (!out) throw runtime_error("Cannot open " + filename);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            out << matrix[i * N + j];
            if (j < N - 1) out << " ";
        }
        out << "\n";
    }
}

__global__ void matrixMulKernel(const int* A, const int* B, int* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        int sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 7) {
        cerr << "Usage: " << argv[0] << " <N> <file_A> <file_B> <file_out> <block_x> <block_y>\n";
        return 1;
    }

    int N = stoi(argv[1]);
    string file_a = argv[2];
    string file_b = argv[3];
    string file_out = argv[4];
    int block_x = stoi(argv[5]);
    int block_y = stoi(argv[6]);

    vector<int> h_A(N * N);
    vector<int> h_B(N * N);
    vector<int> h_C(N * N, 0);

    read_matrix(file_a, h_A, N);
    read_matrix(file_b, h_B, N);

    int* d_A, * d_B, * d_C;
    size_t bytes = N * N * sizeof(int);

    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(block_x, block_y);
    dim3 numBlocks((N + block_x - 1) / block_x, (N + block_y - 1) / block_y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matrixMulKernel <<<numBlocks, threadsPerBlock>>> (d_A, d_B, d_C, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_C.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    write_matrix(file_out, h_C, N);

    cout << milliseconds << endl;

    return 0;
}
