#include <iostream>
#include <fstream>

#include <cmath>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

namespace ublas = boost::numeric::ublas;

#define DEBUG

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

void house_column(ublas::matrix<float>& A,
               unsigned int row_start,
               unsigned int col_start) {

    ublas::vector<float> x(A.size1() - row_start);
    for(unsigned int i = row_start ; i < A.size1(); i++) x(i - row_start) = A(i, col_start);

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

#ifdef DEBUG
    ublas::matrix<float> Q(A.size1(), A.size1());

    for(unsigned int i = 0; i < Q.size1(); i++)
        for(unsigned int j = 0; j < Q.size2(); j++)
            Q(i, j) = (i == j)?1:0;

    for(unsigned int i = row_start; i < Q.size1(); i++) {
        for(unsigned int j = row_start; j < Q.size2(); j++) {
            Q(i, j) = Q(i, j) - 2 * v(i - row_start) * v(j - row_start);
        }
    }

    std::cout << "Q = " << Q << "\n";
#endif
    A = ublas::prod(Q, A);
}

int main() {
	std::cout << "uBLAS\n";
    std::fstream f;
    ublas::matrix<float> in;
    ublas::vector<float> v;

    f.open("data/wiki.qr.example", std::fstream::in);
    f >> in;

    f.close();

    std::cout << in << "\n";

    unsigned int row_num = in.size2();

    for(unsigned int i = 0; i < row_num - 1; i++) {
        house_column(in, i, i);
#ifdef DEBUG
        std::cout << "A = " << in << "\n";
#endif
    }

    std::cout << in << "\n";

	return 0;
}
