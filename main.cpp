#include <iostream>
#include <fstream>

#include <cmath>
#include <ctime>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>


namespace ublas = boost::numeric::ublas;

//#define DEBUG

void eye(ublas::matrix<float>& m) {
    for(unsigned int i = 0; i < m.size1(); i++)
        for(unsigned int j = 0; j < m.size2(); j++)
            m(i, j) = (i == j)?1:0;
}

float sign(float val) {
    return val >= 0?1:-1;
}

float norm(ublas::vector<float>& x) {
    float x_norm = 0.0;
    for(unsigned int i = 0; i < x.size(); i++) x_norm += std::pow(x(i), 2);
    x_norm = std::sqrt(x_norm);
    return x_norm;
}

void normalize(ublas::vector<float>& x) {
    float x_norm = norm(x);
    for(unsigned int i = 0; i < x.size(); i++) {
        x(i) /= x_norm;
    }
}

void householder(ublas::matrix<float>& A,
                 ublas::matrix<float>& QQ,
                    unsigned int row_start,
                    unsigned int col_start,
                    bool column) {
    unsigned int size = column?A.size1():A.size2();
    unsigned int start = column?row_start:col_start;
    ublas::vector<float> x(size - start);
    for(unsigned int i = start ; i < size; i++)
        if(column)
            x(i - start) = A(i, col_start);
        else
            x(i - start) = A(row_start, i);

    float x_norm = norm(x);
    float alpha = -sign(x(0)) * x_norm;

#ifdef DEBUG
    std::cout << "||x|| = " << x_norm << "\n";
    std::cout << "alpha = " << alpha << "\n";
#endif

    ublas::vector<float> v = x;

    v(0) += alpha;
    normalize(v);

#ifdef DEBUG
    std::cout << "v = " << v << "\n";
#endif


    ublas::matrix<float> Q(size, size);
    eye(Q);

    for(unsigned int i = start; i < Q.size1(); i++) {
        for(unsigned int j = start; j < Q.size2(); j++) {
            Q(i, j) = Q(i, j) - 2 * v(i - start) * v(j - start);
        }
    }

#ifdef DEBUG
    std::cout << "Q  = " << Q << "\n";
#endif

    if(column) {
        A = ublas::prod(Q, A);
        QQ = ublas::prod(QQ, Q);
    } else {
        A = ublas::prod(A, Q);
        QQ = ublas::prod(Q, QQ);
    }
}

void bidiag(ublas::matrix<float>& A,
            ublas::matrix<float>& QQL,
            ublas::matrix<float>& QQR) {
    unsigned int row_num = A.size1();

    QQL.resize(row_num, row_num);
    QQR.resize(row_num, row_num);

    eye(QQL);
    eye(QQR);

    for(unsigned int i = 0; i < row_num - 1; i++) {
        householder(A, QQL, i, i, true);
        if(i < row_num - 2) householder(A, QQR, i, i + 1, false);

#ifdef DEBUG
        std::cout << "QQL = " << QQL << "\n";
        std::cout << "QQR = " << QQR << "\n";
        std::cout << "AAA = " << A << "\n";
        std::cout << "*****************\n";
#endif
    }
}

void random_fill(ublas::matrix<float>& A, unsigned int size) {
    A.resize(size, size);

    for(unsigned int i = 0; i < A.size1(); i++) {
        for(unsigned int j = 0; j < A.size2(); j++) {
            A(i, j) = (float)rand() / RAND_MAX;
        }
    }
}

int main() {
    srand(time(0));

    ublas::matrix<float> in;

    /*
    std::fstream f;
    f.open("data/wiki.example", std::fstream::in);
    f >> in;
    f.close();
    */

    random_fill(in, 128);

    std::cout << in << "\n";

    ublas::matrix<float> QQL;
    ublas::matrix<float> QQR;

    bidiag(in, QQL, QQR);

    ublas::matrix<float> result = ublas::prod(in, QQR);
    result = ublas::prod(QQL, result);
    std::cout << result << "\n";

	return 0;
}
