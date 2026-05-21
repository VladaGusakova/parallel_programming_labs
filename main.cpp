#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <stdexcept>
#include <string>
#include <mpi.h>

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
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 5) {
        if (rank == 0) cerr << "Usage: " << argv[0] << " <N> <file_A> <file_B> <file_out>\n";
        MPI_Finalize();
        return 1;
    }

    int N = stoi(argv[1]);
    string file_a = argv[2];
    string file_b = argv[3];
    string file_out = argv[4];

    vector<int> A;
    vector<int> B(N * N);
    vector<int> C;

    if (rank == 0) {
        A.resize(N * N);
        C.resize(N * N, 0);
        read_matrix(file_a, A, N);
        read_matrix(file_b, B, N);
    }

    MPI_Bcast(B.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> sendcounts(size);
    vector<int> displs(size);
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        int rows = N / size + (i < N % size ? 1 : 0);
        sendcounts[i] = rows * N;
        displs[i] = sum;
        sum += sendcounts[i];
    }

    vector<int> local_A(sendcounts[rank]);
    vector<int> local_C(sendcounts[rank], 0);

    MPI_Scatterv(rank == 0 ? A.data() : nullptr, sendcounts.data(), displs.data(), MPI_INT,
        local_A.data(), sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

    double start_time;
    if (rank == 0) start_time = MPI_Wtime();

    int local_rows = sendcounts[rank] / N;
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < N; ++j) {
            int sum_val = 0;
            for (int k = 0; k < N; ++k) {
                sum_val += local_A[i * N + k] * B[k * N + j];
            }
            local_C[i * N + j] = sum_val;
        }
    }

    MPI_Gatherv(local_C.data(), sendcounts[rank], MPI_INT,
        rank == 0 ? C.data() : nullptr, sendcounts.data(), displs.data(), MPI_INT,
        0, MPI_COMM_WORLD);

    if (rank == 0) {
        double end_time = MPI_Wtime();
        auto duration = (long long)((end_time - start_time) * 1000.0);
        write_matrix(file_out, C, N);
        cout << duration << endl;
    }

    MPI_Finalize();
    return 0;
}
