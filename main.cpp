#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <stdexcept>
#include <mpi.h>

using namespace std;

void generate_matrix(vector<double>& M, int N) {
    for (int i = 0; i < N * N; ++i) {
        M[i] = static_cast<double>(rand() % 10);
    }
}

void write_matrix(const string& filename, const vector<double>& matrix, int N) {
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

    if (argc < 2) {
        if (rank == 0)
            cerr << "Usage: " << argv[0] << " <N> [output_file]\n";
        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);
    if (N <= 0) {
        if (rank == 0)
            cerr << "Invalid matrix size\n";
        MPI_Finalize();
        return 1;
    }

    string out_file = (argc >= 3) ? argv[2] : "result.txt";

    vector<double> A;
    vector<double> B(N * N);
    vector<double> C;

    if (rank == 0) {
        srand(time(nullptr));
        A.resize(N * N);
        C.resize(N * N, 0.0);
        generate_matrix(A, N);
        generate_matrix(B, N);
    }

    MPI_Bcast(B.data(), N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    vector<int> sendcounts(size);
    vector<int> displs(size);
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        int rows = N / size + (i < N % size ? 1 : 0);
        sendcounts[i] = rows * N;
        displs[i] = sum;
        sum += sendcounts[i];
    }

    vector<double> local_A(sendcounts[rank]);
    vector<double> local_C(sendcounts[rank], 0.0);

    MPI_Scatterv(
        rank == 0 ? A.data() : nullptr,
        sendcounts.data(),
        displs.data(),
        MPI_DOUBLE,
        local_A.data(),
        sendcounts[rank],
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    int local_rows = sendcounts[rank] / N;
    for (int i = 0; i < local_rows; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum_val = 0.0;
            for (int k = 0; k < N; ++k) {
                sum_val += local_A[i * N + k] * B[k * N + j];
            }
            local_C[i * N + j] = sum_val;
        }
    }

    MPI_Gatherv(
        local_C.data(),
        sendcounts[rank],
        MPI_DOUBLE,
        rank == 0 ? C.data() : nullptr,
        sendcounts.data(),
        displs.data(),
        MPI_DOUBLE,
        0,
        MPI_COMM_WORLD
    );

    double end_time = MPI_Wtime();

    if (rank == 0) {
        write_matrix(out_file, C, N);

        double elapsed = end_time - start_time;
        long long operations = 2LL * N * N * N;
        double gflops = (operations / elapsed) / 1e9;

        cout << "Matrix size: " << N << "x" << N << endl;
        cout << "Processes: " << size << endl;
        cout << "Time: " << elapsed << " sec" << endl;
        cout << "Performance: " << gflops << " GFLOPS" << endl;
    }

    MPI_Finalize();
    return 0;
}