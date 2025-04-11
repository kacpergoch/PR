#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
using namespace std;

void sequential_matrix_multiply(int** matrix_result,const int n, int** matrix_A, int** matrix_B) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix_result[i][j] = 0;
            for (int k = 0; k < n; k++) {
                matrix_result[i][j] += matrix_A[i][k] * matrix_B[k][j];
            }
        }
    }
}

void parallel1_matrix_multiply(int** matrix_result,const int n, int** matrix_A, int** matrix_B,
                                const int num_threads = omp_get_max_threads()) {
    int i, j, k;
    omp_set_num_threads(num_threads);

#pragma omp parallel for shared(matrix_A, matrix_B, matrix_result, i) private(j, k) schedule(runtime)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            matrix_result[i][j] = 0;
            for (k = 0; k < n; k++) {
                matrix_result[i][j] += matrix_A[i][k] * matrix_B[k][j];
            }
        }
    }
}

void parallel2_matrix_multiply(int** matrix_result,const int n, int** matrix_A, int** matrix_B) {
    int i, j, k;
    for (i = 0; i < n; i++) {
#pragma omp parallel for shared(matrix_A, matrix_B, matrix_result, i, j) private(k)
        for (j = 0; j < n; j++) {
            matrix_result[i][j] = 0;
            for (k = 0; k < n; k++) {
                matrix_result[i][j] += matrix_A[i][k] * matrix_B[k][j];
            }
        }
    }
}

void parallel3_matrix_multiply(int** matrix_result,const int n, int** matrix_A, int** matrix_B) {
    int i, j, k;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            matrix_result[i][j] = 0;
#pragma omp parallel for shared(matrix_A, matrix_B, matrix_result, i, j, k)
            for (k = 0; k < n; k++) {
                matrix_result[i][j] += matrix_A[i][k] * matrix_B[k][j];
            }
        }
    }
}

bool read_matrix_from_file(const string& filename, int** &matrix, int &rows, int &cols) {
    ifstream file_stream(filename);
    if (!file_stream.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return false;
    }

    file_stream >> rows >> cols;

    matrix = new int*[rows];
    for (int i = 0; i < rows; i++) {
        matrix[i] = new int[cols];
        for (int j = 0; j < cols; j++) {
            file_stream >> matrix[i][j];
        }
    }

    file_stream.close();
    return true;
}

int** allocate_matrix(const int rows, const int cols) {
    int** matrix = new int*[rows];
    for (int i = 0; i < rows; i++) {
        matrix[i] = new int[cols];
    }
    return matrix;
}

void free_matrix(int** matrix, const int rows) {
    for (int i = 0; i < rows; i++) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

void test_configuration(int** A, int** B, const int rows_A, const int cols_B,
                       const int threads, const string& schedule = "") {
    int** C = allocate_matrix(rows_A, cols_B);
    double total_time = 0;

    if (!schedule.empty()) {
        omp_set_schedule(schedule == "static" ? omp_sched_static :
                         schedule == "dynamic" ? omp_sched_dynamic :
                         omp_sched_guided, 0);
    }

    const auto start = chrono::high_resolution_clock::now();
    parallel1_matrix_multiply(C, rows_A, A, B, threads);
    const auto end = chrono::high_resolution_clock::now();

    chrono::duration<double> elapsed = end - start;
    total_time = elapsed.count();

    cout << "| " << rows_A << "x" << cols_B << "\t\t| " << threads << "\t\t| "
         << (schedule.empty() ? "N/A" : schedule) << "\t| "
         << total_time << "\t|" << endl;

    free_matrix(C, rows_A);
}

void print_table_header() {
    cout << "|    Matrix Size   |  Num Threads  |  Schedule  | Time (s)     |\n";
    cout << "|------------------|---------------|------------|--------------|\n";
}

void benchmark_matrix_multiplication(const string& file_A, const string& file_B) {
    int** A = nullptr;
    int** B = nullptr;
    int** C_seq = nullptr;
    int** C_var1 = nullptr;
    int** C_var2 = nullptr;
    int** C_var3 = nullptr;
    int rows_A, cols_A, rows_B, cols_B;

    bool read_A = false, read_B = false;
#pragma omp parallel sections num_threads(2)
    {
#pragma omp section
        {
            read_A = read_matrix_from_file(file_A, A, rows_A, cols_A);
        }
#pragma omp section
{
    read_B = read_matrix_from_file(file_B, B, rows_B, cols_B);
}
    }
    if (!read_A || !read_B) return;

    if (cols_A != rows_B) {
        cerr << "Error: Matrices cannot be multiplied (cols_A != rows_B)\n";
        free_matrix(A, rows_A);
        free_matrix(B, rows_B);
        return;
    }

    C_seq = allocate_matrix(rows_A, cols_B);
    C_var1 = allocate_matrix(rows_A, cols_B);
    C_var2 = allocate_matrix(rows_A, cols_B);
    C_var3 = allocate_matrix(rows_A, cols_B);

    cout << "Multiplying " << rows_A << "x" << cols_A << " with " << rows_B << "x" << cols_B << "...\n";


    for (int i = 0; i < 4; i++) {
        auto start = chrono::high_resolution_clock::now();
        switch (i) {
            case 0: sequential_matrix_multiply(C_seq, rows_A, A, B); break;
            case 1: parallel1_matrix_multiply(C_var1, rows_A, A, B); break;
            case 2: parallel2_matrix_multiply(C_var2, rows_A, A, B); break;
            case 3: parallel3_matrix_multiply(C_var3, rows_A, A, B); break;
            default: break;
        }
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> time = end - start;
        cout << "Method " << i << " time: " << time.count() << "s\n";
    }


    free_matrix(A, rows_A);
    free_matrix(B, rows_B);
    free_matrix(C_seq, rows_A);
    free_matrix(C_var1, rows_A);
    free_matrix(C_var2, rows_A);
    free_matrix(C_var3, rows_A);
}

void benchmark_thread_number(const string& file_A, const string& file_B) {
    int** A = nullptr;
    int** B = nullptr;
    int rows_A, cols_A, rows_B, cols_B;

    bool read_A = false, read_B = false;
#pragma omp parallel sections num_threads(2)
    {
#pragma omp section
        {
            read_A = read_matrix_from_file(file_A, A, rows_A, cols_A);
        }
#pragma omp section
{
    read_B = read_matrix_from_file(file_B, B, rows_B, cols_B);
}
    }
    if (!read_A || !read_B) return;

    if (cols_A != rows_B) {
        cerr << "Error: Matrices cannot be multiplied (cols_A != rows_B)\n";
        free_matrix(A, rows_A);
        free_matrix(B, rows_B);
        return;
    }

    print_table_header();

    const int threads_list[] = {1, 2, 4, 8, 16};
    const string schedules[] = {"static", "dynamic", "guided"};
    for (int t : threads_list) {
        test_configuration(A, B, rows_A, cols_B, t);
    }


    free_matrix(A, rows_A);
    free_matrix(B, rows_B);
}

void benchmark_schedule(const string& file_A, const string& file_B) {
    int** A = nullptr;
    int** B = nullptr;
    int rows_A, cols_A, rows_B, cols_B;

    bool read_A = false, read_B = false;
#pragma omp parallel sections num_threads(2)
    {
#pragma omp section
        {
            read_A = read_matrix_from_file(file_A, A, rows_A, cols_A);
        }
#pragma omp section
{
    read_B = read_matrix_from_file(file_B, B, rows_B, cols_B);
}
    }
    if (!read_A || !read_B) return;

    if (cols_A != rows_B) {
        cerr << "Error: Matrices cannot be multiplied (cols_A != rows_B)\n";
        free_matrix(A, rows_A);
        free_matrix(B, rows_B);
        return;
    }

    print_table_header();


    const string schedules[] = {"static", "dynamic", "guided"};
    for (int t : {8, 16}) {
        for (const auto& sched : schedules) {
            test_configuration(A, B, rows_A, cols_B, t, sched);
        }
    }


    free_matrix(A, rows_A);
    free_matrix(B, rows_B);
}

int main() {
    const string base_path = "matrices/";
    const string sizes_test1[] = {"10_10", "100_100", "500_500", "1000_1000", "2000_2000"};
    const string sizes_test2[] = {"100_100", "500_500", "1000_1000", "2000_2000"};
    const string sizes_test3[] = {"1000_1000", "2000_2000"};
    for (const string& size : sizes_test1) {
        benchmark_matrix_multiplication(base_path + std::string(size) + "_A.txt", base_path + std::string(size) + "_B.txt");
        cout << "-----------------------------------\n";
    }


    for (const auto& size : sizes_test2) {
        cout << "\nBenchmark for size: " << size << endl;
        benchmark_thread_number(base_path + size + "_A.txt", base_path + size + "_B.txt");
        cout << "-----------------------------------------\n";
    }

    for (const auto& size : sizes_test3) {
        cout << "\nAdditional benchmark for size: " << size << endl;
        benchmark_schedule(base_path + size + "_A.txt", base_path + size + "_B.txt");
        cout << "-----------------------------------------\n";
    }
    return 0;
}