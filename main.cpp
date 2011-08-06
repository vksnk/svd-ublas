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

void house_column(ublas::matrix<float>& A,
               unsigned int row_start,
               unsigned int col_start) {
    ublas::vector<float> v(A.size1() - row_start);

    //calc norm of x
    float x_norm = 0.0;
    for(unsigned int i = row_start; i < A.size1(); i++) x_norm += std::pow(A(i, col_start), 2);
    x_norm = std::sqrt(x_norm);

    float alpha = -sign(A(row_start, col_start)) * x_norm;
#ifdef DEBUG
    std::cout << "||x|| = " << x_norm << "\n";
    std::cout << "alpha = " << alpha << "\n";
#endif

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
    }

    std::cout << in << "\n";

	return 0;
}
