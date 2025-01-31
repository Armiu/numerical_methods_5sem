#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cassert>
#include <random>
// #include "/home/arina/Desktop/chm/TASK2/matplotlibcpp.h"

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


// 1.QR-разложение методом Гивенса
pair<vector<vector<double>>, vector<vector<double>>> qr_givens(vector<vector<double>> A) {
    int n = A.size();
    int m = A[0].size();
    assert(m <= n); // Обеспечиваем, что матрица не "широкая"

    vector<vector<double>> R = A;
    vector<vector<double>> Q(n, vector<double>(n, 0.0));

    // Инициализация Q как единичной матрицы
    for (int i = 0; i < n; ++i)
        Q[i][i] = 1.0;

    // auto start = high_resolution_clock::now();

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

    // auto end = high_resolution_clock::now();
    // auto duration = duration_cast<milliseconds>(end - start);

    // cout << "Время выполнения QR разложения Гивенса: " << duration.count() << " ms" << endl;

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


void gershgorin(const vector<vector<double>>& A) {
    int n = A.size();
    if (n == 0 || A[0].size() != n) {
        cerr << "Ошибка: некорректный размер матрицы." << endl;
        return;
    }

    vector<double> centers(n);
    vector<double> radii(n);
    double min_eigenvalue = numeric_limits<double>::max(); // Инициализируем максимальным значением double
    double max_eigenvalue = numeric_limits<double>::lowest(); // Инициализируем минимальным значением double

    for (int i = 0; i < n; ++i) {
        centers[i] = A[i][i];
        radii[i] = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                radii[i] += abs(A[i][j]);
            }
        }

        // Вывод кругов и интервалов Гершгорина
        // cout << "Круг " << i + 1 << ": |λ - " << fixed << setprecision(10) << centers[i] << "| ≤ " << radii[i] << endl;
        // cout << "Интервал " << i + 1 << ": [" << fixed << setprecision(10) << centers[i] - radii[i] << ", " << centers[i] + radii[i] << "]" << endl;

        min_eigenvalue = min(min_eigenvalue, centers[i] - radii[i]);
        max_eigenvalue = max(max_eigenvalue, centers[i] + radii[i]);
    }

    cout << "\nОценка спектра: [" << fixed << setprecision(10) << min_eigenvalue << ", " << max_eigenvalue << "]" << endl;
}


// Структура для хранения границ спектра
struct Spectrum {
    double min_eigenvalue;
    double max_eigenvalue;
};

// Функция для вычисления скалярного произведения векторов
double dot_product(const vector<double>& v1, const vector<double>& v2) {
    double result = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}

// Функция для вычисления Евклидовой нормы вектора
double euclidean_norm(const vector<double>& v) {
    return sqrt(dot_product(v, v));
}

// Функция для умножения матрицы на вектор
vector<double> matrix_vector_multiply(const vector<vector<double>>& A, const vector<double>& x) {
    int n = A.size();
    vector<double> result(n, 0.0);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            result[i] += A[i][j] * x[j];
        }
    }
    return result;
}


// Функция для генерации упорядочивания Чебышева
vector<int> generate_chebyshev_ordering(int n) {
    if (n == 1) {
        return {0};
    }

    int m = n / 2;
    vector<int> prev_ordering = generate_chebyshev_ordering(m);
    vector<int> ordering;
    for (int i = 0; i < m; ++i) {
        ordering.push_back(2 * prev_ordering[i]);
        ordering.push_back(n - 1 - prev_ordering[i]);
    }
    return ordering;
}

// Метод Чебышева для решения СЛАУ
vector<double> chebyshev_method(const vector<vector<double>>& A, const vector<double>& f, double gamma1, double gamma2, int n_iterations, const vector<int>& ordering) {
    int n = A.size();
    vector<double> x(n, 0.0);
    double tau0 = 2.0 / (gamma1 + gamma2);
    double rho0 = (1.0 - gamma1 / gamma2) / (1.0 + gamma1 / gamma2);

    vector<double> r = f;  // r = (E+A)x - f
    for (int k = 0; k < n_iterations; ++k) {
        int mu_index = ordering[k];
        double mu_k = -cos((2.0 * mu_index + 1.0) * M_PI / (2.0 * n_iterations));
        double tau_k = tau0 / (1.0 + rho0 * mu_k);

        vector<double> Ax = matrix_vector_multiply(A, x);
        for (int i = 0; i < n; ++i) {
            r[i] = f[i] - x[i] - Ax[i]; // Обновляем невязку r
            x[i] += tau_k * r[i];
        }
    }
    return x;
}

int main(int argc, char* argv[]) {
    // ввод
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

    // Создаем матрицу EA = E + A
    vector<vector<double>> EA = A;
    for (int i = 0; i < n; ++i) {
        EA[i][i] += 1.0;
    }
    vector<vector<double>> E(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        E[i][i] = 1.0;
    }
    
    // Вычисляем QR-разложение для EA
    pair<vector<vector<double>>, vector<vector<double>>> qr_result = qr_givens(EA);
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
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
             f[i] += EA[i][j] * x_true[j];
        }
    }

    // Решение системы Ax = f с помощью QR-разложения
    vector<double> QTf(n, 0.0);
    vector<vector<double>> QT = transpose(Q);
    for(int i = 0; i < n; ++i){
         for(int j = 0; j < n; ++j){
             QTf[i] += QT[i][j] * f[j];
         }
    }
    vector<double> x = back_substitution(R, QTf);

    // Оценка точности
    // r = f - Ax
    vector<double> r(n, 0.0);
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
             r[i] -= EA[i][j] * x[j];
        }
    }
    for(int i = 0; i < n; ++i) {
         r[i] += f[i];
    }
    double r_norm = vector_norm(r);

    vector<double> delta(n);
    for(int i = 0; i < n; ++i) {
        delta[i] = x[i] - x_true[i];
    }
    double delta_norm = euclidean_norm(delta);

    const int width = 15;

    cout << left << setw(width) << "x_true:" << "x:" << endl;
    cout << string(width * 2, '-') << endl; // Разделительная линия

    size_t min_size = min(x_true.size(), x.size());

    // Вывод элементов векторов построчно
    for (size_t i = 0; i < min_size; ++i) {
        cout << left << setw(width) << x_true[i] << x[i] << endl;
    }
    // Вывод результатов
    cout << "Среднеквадратичная норма погрешности решения прямым методом - delta: " << delta_norm << endl;
    Spectrum spectrum_estimation;
    cout << "Оценка спектра матрицы (E + A) методом Гершгорина:" << endl;
    gershgorin(EA);
    spectrum_estimation.min_eigenvalue = 1; // Min eigenvalue from Gershgorin
    spectrum_estimation.max_eigenvalue = 152.6; // Max eigenvalue from Gershgorin


     // Метод Чебышева
    int n_iterations = 2; 
    vector<double> x_cheb;
    double error_chebyshev = 0.0;
    vector<int> cheb_iterations;
    vector<double> cheb_errors;
    vector<double> cheb_relative_errors;
    vector<double> x_cheb_64;

    while (true) {
        vector<int> ordering = generate_chebyshev_ordering(n_iterations);
        x_cheb = chebyshev_method(A, f, spectrum_estimation.min_eigenvalue, spectrum_estimation.max_eigenvalue, n_iterations, ordering);
        // Оценка погрешности (евклидова норма)
        vector<double> diff(n);
        for(int i = 0; i < n; ++i)
            diff[i] = x_cheb[i] - x_true[i];

        error_chebyshev = euclidean_norm(diff);
        cheb_iterations.push_back(n_iterations);
        cheb_errors.push_back(error_chebyshev);

        double relative_error = error_chebyshev / euclidean_norm(x_true);
        cheb_relative_errors.push_back(relative_error);

        if(n_iterations == 64){
            x_cheb_64 = x_cheb;
        }
        if (error_chebyshev <= delta_norm) {
            break;  // Достигнута требуемая точность
        }
        n_iterations *= 2;

    }

    cout << "Оптимальное количество итераций метода Чебышева:" << endl;
    cout << "n = " << n_iterations << endl;
    cout << "\nПогрешности метода Чебышева на каждой итерации:" << endl;
    cout << left << setw(25) << "n_iterations" 
         << "Среднеквадратическая ошибка" 
         << "        Относительная ошибка" << endl;
    cout << string(80, '-') << endl;
    // Установка формата вывода в экспоненциальный
    cout << scientific << setprecision(6);
    for(size_t i = 0; i < cheb_iterations.size(); ++i){
        cout << left << setw(25) << cheb_iterations[i] 
             << cheb_errors[i] 
             << "                       " 
             << cheb_relative_errors[i] << endl;
    }
    // Сброс формата вывода на стандартный
    cout.unsetf(ios::floatfield);
    cout << fixed;

    if(!x_cheb_64.empty()){
        cout << "\nСравнение x_cheb и x_true при n = 64:" << endl;
        cout << left << setw(15) << "x_true:" << "x_cheb:" << endl;
        cout << string(30, '-') << endl;
        for(size_t i=0; i < min(x_true.size(), x_cheb_64.size()); ++i){
            cout << left << setw(15) << x_true[i] << x_cheb_64[i] << endl;
        }
    }
    return 0;
}