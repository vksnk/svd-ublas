#include <iostream>
#include <fstream>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

namespace ublas = boost::numeric::ublas;

int main() {
	std::cout << "uBLAS\n";
    std::fstream f;
    ublas::matrix<float> in;
    ublas::vector<float> v;

    f.open("wiki.example", std::fstream::in);
    f >> in;
    f >> v;
    f.close();

    ublas::vector<float> p = ublas::prod(v, in);

    std::cout << in << "\n";
    std::cout << v << "\n";
    std::cout << p << "\n";

	return 0;
}
