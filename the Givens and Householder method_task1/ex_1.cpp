#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cassert>
#include <random>

using namespace std;
using namespace chrono;

// Функция для вычисления нормы вектора
double vector_norm(const vector<double>& v) {
    double max_val = 0.0;
    for (double val : v) {
        max_val = max(max_val, abs(val));
    }
    return max_val;
}

double vector_norm_2(const vector<double>& v) {
    double sum_sq = 0.0;
    for (double val : v) {
        sum_sq += val * val;
    }
    return sqrt(sum_sq);
}

// Функция для вычисления матричной нормы (подчиненной максимум-норме)
double matrix_norm(const vector<vector<double>>& A) {
    int n = A.size();
    double max_norm = 0.0;
    for (int i = 0; i < n; ++i) {
        double row_sum = 0.0;
        for (int j = 0; j < n; ++j) {
            row_sum += abs(A[i][j]);
        }
        max_norm = max(max_norm, row_sum);
    }
    return max_norm;
}

// Функция для умножения матриц
vector<vector<double>> matrix_multiply(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    int m = B[0].size();
    int p = B.size();
    vector<vector<double>> C(n, vector<double>(m, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            for (int k = 0; k < p; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}


// Функция для вычисления разности матриц
vector<vector<double>> matrix_subtract(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C[i][j] = A[i][j] - B[i][j];
        }
    }
    return C;
}

// Функция для транспонирования матрицы
vector<vector<double>> transpose(const vector<vector<double>>& A) {
    int n = A.size();
    int m = A[0].size();
    vector<vector<double>> At(m, vector<double>(n));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            At[j][i] = A[i][j];
        }
    }
    return At;
}

// Функция для установки диагональных элементов R положительными
void ensure_positive_diagonal(vector<vector<double>>& Q, vector<vector<double>>& R) {
    int n = min(Q.size(), R.size());
    for (int i = 0; i < n; ++i) {
        if (R[i][i] < 0) {
            // Инвертируем знак столбца Q
            for (size_t j = 0; j < Q.size(); ++j) {
                Q[j][i] = -Q[j][i];
            }
            // Инвертируем знак строки R
            for (size_t j = i; j < R[0].size(); ++j) {
                R[i][j] = -R[i][j];
            }
        }
    }
}

// Функция для установки небольших значений в ноль
void zero_small_values(vector<vector<double>>& R, double threshold = 1e-10) {
    int n = R.size();
    int m = R[0].size();
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            if (i > j && abs(R[i][j]) < threshold)
                R[i][j] = 0.0;
}




// 1.QR-разложение методом Гивенса (in-place)
pair<vector<vector<double>>, vector<vector<double>>> qr_givens(vector<vector<double>> A) {
    int n = A.size();
    int m = A[0].size();
    assert(m <= n); // Обеспечиваем, что матрица не "широкая"

    vector<vector<double>> R = A;
    vector<vector<double>> Q(n, vector<double>(n, 0.0));

    // Инициализация Q как единичной матрицы
    for (int i = 0; i < n; ++i)
        Q[i][i] = 1.0;

    auto start = high_resolution_clock::now();

    // Применяем повороты Гивенса для каждой поддиагональной позиции
    for (int j = 0; j < min(n, m); ++j) {
        for (int i = j + 1; i < n; ++i) {
            double a = R[j][j];
            double b = R[i][j];
            double c, s;

            if (b == 0.0) {
                c = 1.0;
                s = 0.0;
            } else {
                double r = hypot(a, b);
                c = a / r;
                s = -b / r;
            }

            // Обновляем матрицу R
            for (int k = j; k < m; ++k) {
                double temp = c * R[j][k] - s * R[i][k];
                R[i][k] = s * R[j][k] + c * R[i][k];
                R[j][k] = temp;
            }

            // Обновляем матрицу Q
            for (int k = 0; k < n; ++k) {
                double temp = c * Q[k][j] - s * Q[k][i];
                Q[k][i] = s * Q[k][j] + c * Q[k][i];
                Q[k][j] = temp;
            }
        }
    }

    ensure_positive_diagonal(Q, R);
    zero_small_values(R);

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << "Время выполнения QR разложения Гивенса: " << duration.count() << " ms" << endl;

    return make_pair(Q, R);
}

// 2. QR-разложение методом Хаусхолдера
pair<vector<vector<double>>, vector<vector<double>>> qr_householder(const vector<vector<double>>& A) {
    int n = A.size();
    vector<vector<double>> Q(n, vector<double>(n, 0.0));
    vector<vector<double>> R = A;

    for (int i = 0; i < n; ++i) {
        Q[i][i] = 1.0;
    }

    auto start = high_resolution_clock::now();

    for (int k = 0; k < n - 1; ++k) {
        vector<double> x(n - k);
        for (int i = k; i < n; ++i) {
            x[i - k] = R[i][k];
        }

        double x_norm = 0.0;
        for (double val : x) {
            x_norm += val * val;
        }
         x_norm = sqrt(x_norm);

        vector<double> v = x;
        v[0] += (x[0] >= 0 ? 1 : -1) * x_norm;

        double v_norm_sq = 0.0;
        for (double val : v) {
            v_norm_sq += val * val;
        }

        vector<vector<double>> H(n, vector<double>(n, 0.0));
          for (int i = 0; i < n; ++i) {
            H[i][i] = 1.0;
        }

        for (int i = k; i < n; ++i) {
            for (int j = k; j < n; ++j) {
                H[i][j] -= 2.0 * v[i - k] * v[j - k] / v_norm_sq;
            }
        }

        R = matrix_multiply(H,R);
        Q = matrix_multiply(Q, transpose(H));
    }

    ensure_positive_diagonal(Q, R);
    zero_small_values(R);
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(end - start);

    cout << "Время выполнения QR разложения Хаусхолдера: " << duration.count() << " ms" << endl;
    return make_pair(Q, R);
}


// Функция для чтения CSV файла в матрицу
vector<vector<double>> read_csv(const string& filename) {
    vector<vector<double>> matrix;
    ifstream file(filename);
    string line, cell;

    if (!file.is_open()) {
        cerr << "Не удалось открыть файл: " << filename << endl;
        exit(EXIT_FAILURE);
    }

    while (getline(file, line)) {
        vector<double> row;
        stringstream ss(line);
        while (getline(ss, cell, ',')) {
            row.push_back(stod(cell));
        }
        matrix.push_back(row);
    }

    file.close();
    return matrix;
}


// Функция для вывода матрицы
void print_matrix(const vector<vector<double>>& A) {
    for (const auto& row : A) {
        for (const auto& val : row) {
            cout << setw(10) << val << " ";
        }
        cout << endl;
    }
}

// // Функция для тестирования QR-разложения на небольших матрицах
// void test_qr_decompositions() {
//     // Определение тестовых матриц 3x3
//     vector<vector<double>> test_matrices = {
//         {12, -51, 4},
//         {6, 167, -68},
//         {-4, 24, -41}
//     };

//     vector<vector<double>> test_matrix1 = {
//         {1, 2, 3},
//         {4, 5, 6},
//         {7, 8, 10}
//     };

//     vector<vector<double>> test_matrix2 = {
//         {2, -1, 0},
//         {-1, 2, -1},
//         {0, -1, 2}
//     };

//     vector<vector<vector<double>>> matrices = {test_matrices, test_matrix1, test_matrix2};

//     for (size_t idx = 0; idx < matrices.size(); ++idx) {
//         cout << "\nТестовая матрица " << idx+1 << ":" << endl;
//         print_matrix(matrices[idx]);

//         // Метод Гивенса
//         cout << "\nQR-разложение методом Гивенса:" << endl;
//         pair<vector<vector<double>>, vector<vector<double>>> qr_givens_result = qr_givens(matrices[idx]);
//         cout << "Матрица Q:" << endl;
//         print_matrix(qr_givens_result.first);
//         cout << "Матрица R:" << endl;
//         print_matrix(qr_givens_result.second);

//         // Метод Хаусхолдера
//         cout << "\nQR-разложение методом Хаусхолдера:" << endl;
//         pair<vector<vector<double>>, vector<vector<double>>> qr_householder_result = qr_householder(matrices[idx]);
//         cout << "Матрица Q:" << endl;
//         print_matrix(qr_householder_result.first);
//         cout << "Матрица R:" << endl;
//         print_matrix(qr_householder_result.second);

//         // Проверка нормы разности
//         vector<vector<double>> givens_diff = matrix_subtract(matrices[idx], matrix_multiply(qr_givens_result.first, qr_givens_result.second));
//         double givens_norm = matrix_norm(givens_diff);
//         cout << "Норма разности для Гивенса: " << givens_norm << endl;

//         vector<vector<double>> householder_diff = matrix_subtract(matrices[idx], matrix_multiply(qr_householder_result.first, qr_householder_result.second));
//         double householder_norm = matrix_norm(householder_diff);
//         cout << "Норма разности для Хаусхолдера: " << householder_norm << endl;
//     }
// }

// Функция для решения системы уравнений Rx = Q^T * b методом обратной подстановки
vector<double> back_substitution(const vector<vector<double>>& R, const vector<double>& b) {
    int n = R.size();
    vector<double> x(n);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = b[i];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= R[i][j] * x[j];
        }
        x[i] /= R[i][i];
    }
    return x;
}

bool compare_vectors(const vector<double>& v1, const vector<double>& v2, double tol = 1e-6) {
    if (v1.size() != v2.size()) return false;
    for (size_t i = 0; i < v1.size(); ++i) {
        if (abs(v1[i] - v2[i]) > tol) return false;
    }
    return true;
}

// void test_back_substitution() {
//     // Тестовый пример 1
//     vector<vector<double>> R1 = {
//         {2, -1, 0},
//         {0, 3, 1},
//         {0, 0, 4}
//     };
//     vector<double> b1 = {1, 4, 16};
//     vector<double> expected_x1 = {0.5, 0, 4}; // Исправлено
//     vector<double> computed_x1 = back_substitution(R1, b1);
//     assert(compare_vectors(computed_x1, expected_x1));
//     cout << "Тест 1 пройден успешно." << endl;

//     // Тестовый пример 2
//     vector<vector<double>> R2 = {
//         {1, 2},
//         {0, 3}
//     };
//     vector<double> b2 = {5, 6};
//     vector<double> expected_x2 = {1, 2};
//     vector<double> computed_x2 = back_substitution(R2, b2);
//     assert(compare_vectors(computed_x2, expected_x2));
//     cout << "Тест 2 пройден успешно." << endl;

//     // Тестовый пример 3
//     vector<vector<double>> R3 = {
//         {4, 2},
//         {0, 5}
//     };
//     vector<double> b3 = {10, 15};
//     vector<double> expected_x3 = {1, 3};
//     vector<double> computed_x3 = back_substitution(R3, b3);
//     assert(compare_vectors(computed_x3, expected_x3));
//     cout << "Тест 3 пройден успешно." << endl;
// }   

int main(int argc, char* argv[]) {
    //ввод
    if (argc != 2) {
        cout << "Использование: " << argv[0] << " <имя_файла.csv>" << endl;
        return EXIT_FAILURE;
    }
    string filename = argv[1];
    cout << "Чтение файла: " << filename << endl;
    vector<vector<double>> A = read_csv(filename);

    int n = A.size();
    if (n == 0 || A[0].size() != n) {
        cerr << "Ошибка: некорректный размер матрицы в файле." << endl;
        return 1;
    }

    //test_back_substitution();

    // // // Вывод матрицы A
    // // cout << "Матрица A:" << endl;
    // // print_matrix(A);
    // // cout << endl;

    // vector<vector<double>> E(n, vector<double>(n, 0.0));
    // for (int i = 0; i < n; ++i) {
    //     E[i][i] = 1.0;
    // }

    // Создаем матрицу EA = E + A
    vector<vector<double>> EA = A;
    for (int i = 0; i < n; ++i) {
        EA[i][i] += 1.0;
    }
    
    // Вычисляем QR-разложение для EA
    pair<vector<vector<double>>, vector<vector<double>>> qr_result = qr_givens(EA);

    
    // // // Вычисление QR-разложения методами Гивенса и Хаусхолдера
    // pair<vector<vector<double>>, vector<vector<double>>> qr_givens_result = qr_givens(A);
    // pair<vector<vector<double>>, vector<vector<double>>> qr_householder_result = qr_householder(A);

    // // Вычисление нормы разности для Гивенса
    // vector<vector<double>> givens_diff = matrix_subtract(A, matrix_multiply(qr_givens_result.first, qr_givens_result.second));
    // double givens_norm = matrix_norm(givens_diff);
    // cout << "Норма разности для Гивенса: " << givens_norm << endl;

    // // // Вычисление нормы разности для Хаусхолдера
    // vector<vector<double>> householder_diff = matrix_subtract(A, matrix_multiply(qr_householder_result.first, qr_householder_result.second));
    // double householder_norm = matrix_norm(householder_diff);
    // cout << "Норма разности для Хаусхолдера: " << householder_norm << endl;

    vector<vector<double>>& Q = qr_result.first;
    vector<vector<double>>& R = qr_result.second;

    // Генерация случайного вектора x и вычисление f = Ax
    n = EA.size();
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1.0, 1.0);

    //x_gen
    vector<double> x_true(n);
    for (int i = 0; i < n; ++i) {
        x_true[i] = dis(gen);
    }
    
    vector<double> f(n, 0.0);
    for(int i=0; i<n; ++i){
        for(int j=0; j < n; ++j){
             f[i] += EA[i][j]* x_true[j];
        }
    }

    // Решение системы Ax = f с помощью QR-разложения
    auto start_solve = high_resolution_clock::now();  

    vector<double> QTf(n, 0.0);
    vector<vector<double>> QT = transpose(Q);
    for(int i =0; i < n; ++i){
         for(int j=0; j < n; ++j){
             QTf[i] += QT[i][j]* f[j];
         }
    }
    vector<double> x = back_substitution(R, QTf);

    auto end_solve = high_resolution_clock::now();
    auto duration_solve = duration_cast<milliseconds>(end_solve - start_solve);


    // Оценка точности
    //r = f - Ax
    vector<double> r(n, 0.0);
    for(int i=0; i<n; ++i) {
        for(int j=0; j < n; ++j) {
             r[i] -= EA[i][j]* x[j];
        }
    }
    for(int i=0; i <n; ++i) {
         r[i] += f[i];
    }
    double r_norm = vector_norm(r);


    vector<double> delta(n);
    for(int i=0; i<n; ++i) {
        delta[i] = x[i] - x_true[i];

    }
    double delta_norm = vector_norm_2(delta);


    // Определение ширины столбцов
    const int width = 15;

    // Вывод заголовков столбцов
    cout << left << setw(width) << "x_true:" << "x:" << endl;
    cout << string(width * 2, '-') << endl; // Разделительная линия

    // Определение минимального размера векторов для безопасного доступа
    size_t min_size = min(x_true.size(), x.size());

    // Вывод элементов векторов построчно
    for (size_t i = 0; i < min_size; ++i) {
        cout << left << setw(width) << x_true[i] << x[i] << endl;
    }
    // Вывод результатов
    //cout << "Время выполнения QR разложения: " << duration_qr.count() << " ms" << endl;
    cout << "Время решения системы: " << duration_solve.count() << " ms" << endl;
    cout << "Норма невязки r: " << r_norm << endl;
    cout << "Норма погрешности delta: " << delta_norm << endl;

    return 0;
}