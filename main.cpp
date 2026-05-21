#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <stdexcept>
#include <string>
#include <omp.h>

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

int main(int argc, char* argv[]) {
    if (argc != 6) {
        cerr << "Usage: " << argv[0] << " <N> <file_A> <file_B> <file_out> <num_threads>\n";
        return 1;
    }

    int N = stoi(argv[1]);
    string file_a = argv[2];
    string file_b = argv[3];
    string file_out = argv[4];
    int num_threads = stoi(argv[5]);

    vector<int> A(N * N);
    vector<int> B(N * N);
    vector<int> C(N * N, 0);

    read_matrix(file_a, A, N);
    read_matrix(file_b, B, N);

    omp_set_num_threads(num_threads);

    auto start = chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            int sum = 0;
            for (int k = 0; k < N; ++k) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end - start).count();

    write_matrix(file_out, C, N);

    cout << duration << endl;

    return 0;
}