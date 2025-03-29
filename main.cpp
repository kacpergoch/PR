#include <iostream>
#include <fstream>
#include <string>
using namespace std;

void sequential_matrix_multiply(int** matrix_result,const int n, const int** matrix_A, const int** matrix_B) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            matrix_result[i][j] = 0; // Initialize the result element to zero
            for (int k = 0; k < n; k++) {
                matrix_result[i][j] += matrix_A[i][k] * matrix_B[k][j];
            }
        }
    }
}

void read_matrix_from_file(const string filename, int** matrix) {
    ifstream file_stream;
    file_stream.open(filename);
    if (!file_stream.is_open()) {
        return;
    }

    int n = 0, m = 0, value = 0;
    int** temp_matrix = new int*[n];
    file_stream >> n >> m;
    for (int i = 0; i < n; i++) {
        temp_matrix[i] = new int[m];
        for (int j = 0; j < m; j++) {
            file_stream >> value;
            temp_matrix[i][j] = svalue;
        }
    }
    matrix = temp_matrix;
}

int main() {
    int** matrix = nullptr;
    read_matrix_from_file("matrices/10_10.txt", matrix);
    delete[] matrix;
    return 0;
}