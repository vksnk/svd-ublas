#include <iostream>
#include <fstream>

#include <cmath>
#include <ctime>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

namespace ublas = boost::numeric::ublas;

void pretty_print(const char* tag, ublas::matrix<float>& m)
{
    printf("%s\n", tag);
    for(std::size_t i = 0; i < m.size1(); i++)
    {
        printf("\t");
        for(std::size_t j = 0; j < m.size2(); j++)
        {
            printf("%5.2f ", (fabs(m(i, j)) < 0.00001f?0.0f:m(i, j)));
        }
        printf("\n");
    }
}

void random_fill(ublas::matrix < float >&A, unsigned int size1, unsigned int size2)
{
    A.resize(size1, size2);

    for (unsigned int i = 0; i < A.size1(); i++) {
        for (unsigned int j = 0; j < A.size2(); j++) {
            A(i, j) = (float)rand() / RAND_MAX;
        }
    }
}

float matrix_compare(ublas::matrix < float >&res, ublas::matrix < float >&ref)
{
    float diff = 0.0;

    for (unsigned int i = 0; i < res.size1(); i++) {
        for (unsigned int j = 0; j < res.size2(); j++) {
            diff = std::max(diff, std::abs(res(i, j) - ref(i, j)));
        }
    }

    return diff;
}

void eye(ublas::matrix < float >&m)
{
    for (unsigned int i = 0; i < m.size1(); i++)
        for (unsigned int j = 0; j < m.size2(); j++)
            m(i, j) = (i == j) ? 1.0f : 0.0f;
}

float sign(float val)
{
    return val >= 0.0f ? 1.0f : -1.0f;
}

float norm(ublas::vector < float >&x)
{
    float x_norm = 0.0;
    for (unsigned int i = 0; i < x.size(); i++)
        x_norm += std::pow(x(i), 2);
    x_norm = std::sqrt(x_norm);
    return x_norm;
}

void normalize(ublas::vector < float >&x)
{
    float x_norm = norm(x);
    for (unsigned int i = 0; i < x.size(); i++) {
        x(i) /= x_norm;
    }
}

float pythag(float a, float b)
{
    float absa = fabs(a);
    float absb = fabs(b);

    if (absa > absb) {
        return absa * sqrt(1.0f + pow(absb / absa, 2));
    } else {
        return absb * sqrt(1.0f + pow(absa / absb, 2));
    }
}
