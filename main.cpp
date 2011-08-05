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

    f.open("data/wiki.example", std::fstream::in);
    f >> in;

    f.close();


    std::cout << in << "\n";

	return 0;
}
