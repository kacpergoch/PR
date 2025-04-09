#include <chrono>
#include <iostream>
#include <fstream>
#include <string>
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

void parallel1_matrix_multiply(int** matrix_result,const int n, int** matrix_A, int** matrix_B) {
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix_result[i][j] = 0;
            for (int k = 0; k < n; k++) {
                matrix_result[i][j] += matrix_A[i][k] * matrix_B[k][j];
            }
        }
    }
}

void parallel2_matrix_multiply(int** matrix_result,const int n, int** matrix_A, int** matrix_B) {
    for (int i = 0; i < n; i++) {
#pragma omp parallel for
        for (int j = 0; j < n; j++) {
            matrix_result[i][j] = 0;
            for (int k = 0; k < n; k++) {
                matrix_result[i][j] += matrix_A[i][k] * matrix_B[k][j];
            }
        }
    }
}

void parallel3_matrix_multiply(int** matrix_result,const int n, int** matrix_A, int** matrix_B) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix_result[i][j] = 0;
#pragma omp parallel for
            for (int k = 0; k < n; k++) {
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


void test_matrix_multiplication(const string& file_A, const string& file_B) {
    int** A = nullptr;
    int** B = nullptr;
    int** C_seq = nullptr;
    int** C_var1 = nullptr;
    int** C_var2 = nullptr;
    int** C_var3 = nullptr;
    int rows_A, cols_A, rows_B, cols_B;

    if (!read_matrix_from_file(file_A, A, rows_A, cols_A) ||
        !read_matrix_from_file(file_B, B, rows_B, cols_B)) {
        return; // Error already printed
        }

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
            case 0:
                sequential_matrix_multiply(C_seq, rows_A, A, B);
                break;
            case 1:
                parallel1_matrix_multiply(C_seq, rows_A, A, B);
                break;
            case 2:
                parallel2_matrix_multiply(C_seq, rows_A, A, B);
                break;
            case 3:
                parallel3_matrix_multiply(C_seq, rows_A, A, B);
                break;
            default:
                break;
        }
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> time = end - start;
    }


    free_matrix(A, rows_A);
    free_matrix(B, rows_B);
    free_matrix(C_seq, rows_A);
    free_matrix(C_var1, rows_A);
    free_matrix(C_var2, rows_A);
    free_matrix(C_var3, rows_A);
}

int main() {
    const string base_path = "matrices/";
    string sizes[] = {"10_10", "100_100", "500_500", "1000_1000", "2000_2000"};

    for (const string& size : sizes) {
        test_matrix_multiplication(base_path + std::string(size) + "_A.txt", base_path + std::string(size) + "_B.txt");
        cout << "-----------------------------------\n";
    }

    return 0;
}