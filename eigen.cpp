#include "common.hpp"

void
householder(ublas::matrix<float>& A,
        ublas::matrix <float>& QL,
        ublas::matrix <float>& QR,
        unsigned int row_start, 
        unsigned int col_start)
{
    unsigned int size = A.size1();
    unsigned int start = row_start;

    if (start >= size)
        return; 

    ublas::vector < float > x(size);
    for (unsigned int i = 0; i < size; i++) {
        if (i < start) {
            x(i) = 0;
        } else {
            x(i) = A(i, col_start);
        }
    }

    float x_norm = norm(x);
    float alpha = sign(x(start)) * x_norm;

    // std::cout << column << " " <<  start << " x = " << x(start) << "\n";
#ifdef DEBUG
    std::cout << "||x|| = " << x_norm << "\n";
    std::cout << "alpha = " << alpha << "\n";
#endif

    ublas::vector<float> v = x;

    v(start) += alpha;
    normalize(v);

#ifdef DEBUG
    std::cout << "v = " << v << "\n";
#endif

    ublas::matrix < float > Q;

#ifdef DEBUG
    Q.resize(size, size);
    eye(Q);

    for (unsigned int i = start; i < Q.size1(); i++) {
        for (unsigned int j = start; j < Q.size2(); j++) {
            Q(i, j) = Q(i, j) - 2 * v(i) * v(j);
        }
    }

    std::cout << "Q  = " << Q << "\n";
#endif

    for (unsigned int i = 0; i < A.size2(); i++) {
        float sum_Av = 0.0f;

        for (unsigned int j = 0; j < A.size1(); j++)
            sum_Av = sum_Av + (v(j) * A(j, i));
        for (unsigned int j = 0; j < A.size1(); j++)
            A(j, i) = A(j, i) - 2 * v(j) * sum_Av;
    }

    for (unsigned int i = 0; i < A.size1(); i++) {
        float sum_Qv = 0.0f;

        for (unsigned int j = 0; j < A.size1(); j++)
            sum_Qv = sum_Qv + (v(j) * QL(i, j));
        for (unsigned int j = 0; j < A.size1(); j++)
            QL(i, j) = QL(i, j) - 2 * v(j) * sum_Qv;
    }

    for (unsigned int i = 0; i < A.size1(); i++) {
        float sum_Av = 0.0f;

        for (unsigned int j = 0; j < A.size2(); j++)
            sum_Av = sum_Av + (v(j) * A(i, j));
        for (unsigned int j = 0; j < A.size2(); j++)
            A(i, j) = A(i, j) - 2 * v(j) * sum_Av;
    }

    for (unsigned int i = 0; i < A.size2(); i++) {
        float sum_Qv = 0.0f;

        for (unsigned int j = 0; j < A.size2(); j++)
            sum_Qv = sum_Qv + (v(j) * QR(i, j));
        for (unsigned int j = 0; j < A.size2(); j++)
            QR(i, j) = QR(i, j) - 2 * v(j) * sum_Qv;
    }


}

void eigen(ublas::matrix < float >&A,
            ublas::matrix < float >&QQL,
            ublas::matrix < float >&QQW, 
            ublas::matrix < float >&QQR)
{
    unsigned int row_num = A.size1();
    unsigned int col_num = A.size2();

    QQL.resize(row_num, row_num);
    QQW.resize(row_num, col_num);
    QQR.resize(col_num, col_num);

    eye(QQL);
    eye(QQR);

    unsigned int to = std::min(row_num, col_num);

    for (unsigned int i = 0; i < to; i++) {
        householder(A, QQL, QQR, i + 1, i);
    }
}

bool check_tridiag(ublas::matrix<float>& A)
{
    const float EPS = 0.0001f;

    for (unsigned int i = 0; i < A.size1(); i++) {
        for (unsigned int j = 0; j < A.size2(); j++) {
            if ((std::abs(A(i, j)) > EPS) && ((i - 1) != j) && (i != j) && ((i + 1) != j))
            {
                std::cout << "Failed at " << i << " " << j << " " << A(i, j) << "\n";
                return false;
            }
        }
    }
    return true;
}

bool check_hessenberg(ublas::matrix<float>& A)
{
    const float EPS = 0.0001f;

    for (int i = 0; i < A.size1(); i++) {
        for (int j = 0; j < A.size2(); j++) {
            if ((std::abs(A(i, j)) > EPS) && (i > (j + 1)))
            {
                std::cout << "Failed at " << i << " " << j << " " << A(i, j) << "\n";
                return false;
            }
        }
    }
    return true;
}

int main()
{
    ublas::matrix < float > in;
/*
    std::fstream f;
    f.open("data/wiki.example", std::fstream::in);
    f >> in;
    f.close();
*/
    srand((unsigned int)time(0));
    random_fill(in, 12, 12);

    ublas::matrix < float > ref = in;

    pretty_print("Input:", in);

    ublas::matrix < float > QQL;
    ublas::matrix < float > QQW;
    ublas::matrix < float > QQR;

    eigen(in, QQL, QQW, QQR);

    ublas::matrix < float > result;

    result = ublas::prod(in, trans(QQR));
    result = ublas::prod(QQL, result);

    pretty_print("Tridiag:", in);

    std::cout << "DIFF = " << matrix_compare(result, ref) << "\n";
    std::cout << "Is Hessenberg ? " << check_hessenberg(in) << "\n";
    std::cout << "Is tridiag ? " << check_tridiag(in) << "\n";
    return 0;
}