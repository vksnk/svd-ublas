#include "common.hpp"

void
householder(ublas::matrix<float>& A,
        ublas::matrix <float>& QQ,
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

    ublas::vector<float> v = x;

    v(start) += alpha;
    normalize(v);

    ublas::matrix < float > Q;

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
            sum_Qv = sum_Qv + (v(j) * QQ(i, j));
        for (unsigned int j = 0; j < A.size1(); j++)
            QQ(i, j) = QQ(i, j) - 2 * v(j) * sum_Qv;
    }

    for (unsigned int i = 0; i < A.size1(); i++) {
        float sum_Av = 0.0f;

        for (unsigned int j = 0; j < A.size2(); j++)
            sum_Av = sum_Av + (v(j) * A(i, j));
        for (unsigned int j = 0; j < A.size2(); j++)
            A(i, j) = A(i, j) - 2 * v(j) * sum_Av;
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

void tred2(ublas::matrix<float>& V,
            ublas::vector<float>& d,
            ublas::vector<float>& e
            )
{
    int n = V.size1();
    // pretty_print("V", V);

    // Symmetric Householder reduction to tridiagonal form.
    // This is derived from the Algol procedures tred2 by Bowdler, Martin, Reinsch, and Wilkinson, 
    // Handbook for Auto. Comp., Vol.ii-Linear Algebra, and the corresponding Fortran subroutine in EISPACK.
    for (int j = 0; j < n; j++)
        d(j) = V(n - 1, j);

    // std::cout << d << "\n";

    // Householder reduction to tridiagonal form.
    for (int i = n - 1; i > 0; i--)
    {
        // Scale to avoid under/overflow.
        float scale = 0;
        float h = 0;
        for (int k = 0; k < i; k++)
            scale = scale + fabs(d(k));

        if (scale == 0)
        {
            e(i) = d(i - 1);
            for (int j = 0; j < i; j++)
            {
                d(j) = V(i - 1, j);
                V(i, j) = 0;
                V(j, i) = 0;
            }
        }
        else
        {
            // Generate Householder vector.
            for (int k = 0; k < i; k++)
            {
                d(k) /= scale;
                h += d(k) * d(k);
            }

            float f = d(i - 1);
            float g = (float)sqrt(h);
            if (f > 0) g = -g;

            e(i) = scale * g;
            h = h - f * g;
            d(i - 1) = f - g;
            for (int j = 0; j < i; j++)
                e(j) = 0;

            // Apply similarity transformation to remaining columns.
            for (int j = 0; j < i; j++)
            {
                f = d(j);
                V(j, i) = f;
                g = e(j) + V(j, j) * f;
                for (int k = j + 1; k <= i - 1; k++)
                {
                    g += V(k, j) * d(k);
                    e(k) += V(k, j) * f;
                }
                e(j) = g;
            }

            f = 0;
            for (int j = 0; j < i; j++)
            {
                e(j) /= h;
                f += e(j) * d(j);
            }

            float hh = f / (h + h);
            for (int j = 0; j < i; j++)
                e(j) -= hh * d(j);

            for (int j = 0; j < i; j++)
            {
                f = d(j);
                g = e(j);
                for (int k = j; k <= i - 1; k++)
                    V(k, j) -= (f * e(k) + g * d(k));

                d(j) = V(i - 1, j);
                V(i, j) = 0;
            }
        }
        d(i) = h;

        //pretty_print("V", V);
    }

    // Accumulate transformations.
    for (int i = 0; i < n - 1; i++)
    {
        V(n - 1, i) = V(i, i);
        V(i, i) = 1;
        float h = d(i + 1);
        if (h != 0)
        {
            for (int k = 0; k <= i; k++)
                d(k) = V(k, i + 1) / h;

            for (int j = 0; j <= i; j++)
            {
                float g = 0;
                for (int k = 0; k <= i; k++)
                    g += V(k, i + 1) * V(k, j);
                for (int k = 0; k <= i; k++)
                    V(k, j) -= g * d(k);
            }
        }

        for (int k = 0; k <= i; k++)
            V(k, i + 1) = 0;
    }

    for (int j = 0; j < n; j++)
    {
        d(j) = V(n - 1, j);
        V(n - 1, j) = 0;
    }

    V(n - 1, n - 1) = 1;
    e(0) = 0;

    // pretty_print("V", V);

    // ublas::matrix<float> t3(n, n);
    
    // for(int i = 0; i < n; i++)
    // {
    //     for(int j = 0; j < n; j++)
    //         t3(i, j) = 0;

    //     t3(i, i) = d(i);
    //     if(i)
    //     {
    //         t3(i - 1, i) = e(i);
    //         t3(i, i - 1) = e(i);
    //     }
    // }

    // pretty_print("", t3);

    // t3 = ublas::prod(t3, trans(V));
    // t3 = ublas::prod(V, t3);
    // pretty_print("", t3);
    // std::cout << d << "\n";
    // std::cout << e << "\n";
}

void tql2(ublas::matrix<float>& V,
            ublas::vector<float>& d,
            ublas::vector<float>& e)
{
    int n = V.size1();
    // Symmetric tridiagonal QL algorithm.
    // This is derived from the Algol procedures tql2, by Bowdler, Martin, Reinsch, and Wilkinson, 
    // Handbook for Auto. Comp., Vol.ii-Linear Algebra, and the corresponding Fortran subroutine in EISPACK.
    for (int i = 1; i < n; i++)
        e(i - 1) = e(i);

    e(n - 1) = 0;

    float f = 0;
    float tst1 = 0;
    float eps = 2 * EPS;

    for (int l = 0; l < n; l++)
    {
        // Find small subdiagonal element.
        tst1 = std::max<float>(tst1, fabs(d(l)) + fabs(e(l)));
        int m = l;
        while (m < n)
        {
            if (fabs(e(m)) <= eps * tst1)
                break;
            m++;
        }

        // If m == l, d(l) is an eigenvalue, otherwise, iterate.
        if (m > l)
        {
            int iter = 0;
            do
            {
                iter = iter + 1;  // (Could check iteration count here.)

                // Compute implicit shift
                float g = d(l);
                float p = (d(l + 1) - g) / (2 * e(l));
                float r = pythag(p, 1);
                if (p < 0)
                {
                    r = -r;
                }

                d(l) = e(l) / (p + r);
                d(l + 1) = e(l) * (p + r);
                float dl1 = d(l + 1);
                float h = g - d(l);
                for (int i = l + 2; i < n; i++)
                {
                    d(i) -= h;
                }

                f = f + h;

                // Implicit QL transformation.
                p = d(m);
                float c = 1;
                float c2 = c;
                float c3 = c;
                float el1 = e(l + 1);
                float s = 0;
                float s2 = 0;
                for (int i = m - 1; i >= l; i--)
                {
                    c3 = c2;
                    c2 = c;
                    s2 = s;
                    g = c * e(i);
                    h = c * p;
                    r = pythag(p, e(i));
                    e(i + 1) = s * r;
                    s = e(i) / r;
                    c = p / r;
                    p = c * d(i) - s * g;
                    d(i + 1) = h + s * (c * g + s * d(i));

                    // Accumulate transformation.
                    for (int k = 0; k < n; k++)
                    {
                        h = V(k, i + 1);
                        V(k, i + 1) = s * V(k, i) + c * h;
                        V(k, i) = c * V(k, i) - s * h;
                    }
                }

                p = -s * s2 * c3 * el1 * e(l) / dl1;
                e(l) = s * p;
                d(l) = c * p;

                // Check for convergence.
            }
            while (fabs(e(l)) > eps * tst1);
        }
        d(l) = d(l) + f;
        e(l) = 0;
    }

    // Sort eigenvalues and corresponding vectors.
    for (int i = 0; i < n - 1; i++)
    {
        int k = i;
        float p = d(i);
        for (int j = i + 1; j < n; j++)
        {
            if (d(j) > p)
            {
                k = j;
                p = d(j);
            }
        }

        if (k != i)
        {
            d(k) = d(i);
            d(i) = p;
            for (int j = 0; j < n; j++)
            {
                p = V(j, i);
                V(j, i) = V(j, k);
                V(j, k) = p;
            }
        }
    }
}

void orthes(ublas::matrix<float>& H,
            ublas::matrix<float>& V,
            ublas::vector<float>& ort,
            ublas::vector<float>& d,
            ublas::vector<float>& e)
{
    // Nonsymmetric reduction to Hessenberg form.
    // This is derived from the Algol procedures orthes and ortran, by Martin and Wilkinson, 
    // Handbook for Auto. Comp., Vol.ii-Linear Algebra, and the corresponding Fortran subroutines in EISPACK.
    int n = V.size1();

    int low = 0;
    int high = n - 1;

    for (int m = low + 1; m <= high - 1; m++)
    {
        // Scale column.

        float scale = 0;
        for (int i = m; i <= high; i++)
            scale = scale + fabs(H(i, m - 1));

        if (scale != 0)
        {
            // Compute Householder transformation.
            float h = 0;
            for (int i = high; i >= m; i--)
            {
                ort(i) = H(i, m - 1) / scale;
                h += ort(i) * ort(i);
            }

            float g = sqrt(h);
            if (ort(m) > 0) g = -g;

            h = h - ort(m) * g;
            ort(m) = ort(m) - g;

            // Apply Householder similarity transformation
            // H = (I - u * u' / h) * H * (I - u * u') / h)
            for (int j = m; j < n; j++)
            {
                float f = 0;
                for (int i = high; i >= m; i--)
                    f += ort(i) * H(i, j);

                f = f / h;
                for (int i = m; i <= high; i++)
                    H(i, j) -= f * ort(i);
            }

            for (int i = 0; i <= high; i++)
            {
                float f = 0;
                for (int j = high; j >= m; j--)
                    f += ort(j) * H(i, j);

                f = f / h;
                for (int j = m; j <= high; j++)
                    H(i, j) -= f * ort(j);
            }

            ort(m) = scale * ort(m);
            H(m, m - 1) = scale * g;
        }
    }

    // Accumulate transformations (Algol's ortran).
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            V(i, j) = (i == j ? 1 : 0);

    for (int m = high - 1; m >= low + 1; m--)
    {
        if (H(m, m - 1) != 0)
        {
            for (int i = m + 1; i <= high; i++)
                ort(i) = H(i, m - 1);

            for (int j = m; j <= high; j++)
            {
                float g = 0;
                for (int i = m; i <= high; i++)
                    g += ort(i) * V(i, j);

                // float division avoids possible underflow.
                g = (g / ort(m)) / H(m, m - 1);
                for (int i = m; i <= high; i++)
                    V(i, j) += g * ort(i);
            }
        }
    }
}

void eis_eigsym(ublas::matrix<float>& V)
{
    int n = V.size1();

    ublas::vector<float> d(n);
    ublas::vector<float> e(n);

    tred2(V, d, e);
    tql2(V, d, e);

    pretty_print("Eigen", V);
    std::cout << d << "\n";

    ublas::matrix<float> t3(n, n);
    
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
            t3(i, j) = 0;

        t3(i, i) = d(i);
    }

    t3 = ublas::prod(t3, trans(V));
    t3 = ublas::prod(V, t3);

    pretty_print("Check", t3);
}

void eis_eig(ublas::matrix<float>& H)
{
    int n = H.size1();

    ublas::matrix<float> V(n, n);
    ublas::vector<float> ort(n);
    ublas::vector<float> d(n);
    ublas::vector<float> e(n);

    orthes(H, V, ort, d, e);
    // tql2(H, d, e);

    pretty_print("H", H);
    pretty_print("V", V);
    std::cout << d << "\n";

    // ublas::matrix<float> t3(n, n);
    
    // for(int i = 0; i < n; i++)
    // {
    //     for(int j = 0; j < n; j++)
    //         t3(i, j) = 0;

    //     t3(i, i) = d(i);
    // }

    // t3 = ublas::prod(t3, trans(H));
    // t3 = ublas::prod(H, t3);

    // pretty_print("Check", t3);
}

void eigsym(ublas::matrix < float >&A,
            ublas::matrix < float >&QQ,
            ublas::matrix < float >&QW)
{
    std::size_t n = A.size1();

    QQ.resize(n, n);
    QW.resize(n, n);

    eye(QQ);

    for(std::size_t i = 0; i < n; i++)
    {
        householder(A, QQ, i + 1, i);
    }

    ublas::vector<float> d(n);
    ublas::vector<float> e(n);

    for(std::size_t i = 0; i < n; i++)
    {
        d(i) = A(i, i);
        if(i)
           e(i) = A(i - 1, i);
    }

    tql2(QQ, d, e);

    std::cout << d << "\n";
    std::cout << e << "\n";

   
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
            QW(i, j) = 0;

        QW(i, i) = d(i);
    }
}

void eig(ublas::matrix < float >&A,
            ublas::matrix < float >&QQ,
            ublas::matrix < float >&QW)
{
    std::size_t n = A.size1();

    QQ.resize(n, n);
    QW.resize(n, n);

    eye(QQ);

    for(std::size_t i = 0; i < n; i++)
    {
        householder(A, QQ, i + 1, i);
    }

    pretty_print("Tridiag", A);
    pretty_print("QQ", QQ);

    // ublas::vector<float> d(n);
    // ublas::vector<float> e(n);

    // for(std::size_t i = 0; i < n; i++)
    // {
    //     d(i) = A(i, i);
    //     if(i)
    //        e(i) = A(i - 1, i);
    // }

    // tql2(QQ, d, e);

    // std::cout << d << "\n";
    // std::cout << e << "\n";

   
    // for(int i = 0; i < n; i++)
    // {
    //     for(int j = 0; j < n; j++)
    //         QW(i, j) = 0;

    //     QW(i, i) = d(i);
    // }
}

int main()
{
    ublas::matrix < float > in;

    // std::fstream f;
    // f.open("data/wiki.example", std::fstream::in);
    // f >> in;
    // f.close();

    // srand((unsigned int)time(0));
    random_fill(in, 5, 5);

    ublas::matrix<float> ref = in;

    pretty_print("Input:", in);

    ublas::matrix < float > QQ;
    ublas::matrix < float > QW;

    // eigsym(in, QQ, QW);
    eig(in, QQ, QW);

    ublas::matrix < float > result;

    result = ublas::prod(QW, trans(QQ));
    result = ublas::prod(QQ, result);

    // pretty_print("Tridiag:", in);
    // pretty_print("QL:", QQ);

    std::cout << "DIFF = " << matrix_compare(result, ref) << "\n";
    std::cout << "Is Hessenberg ? " << check_hessenberg(in) << "\n";
    std::cout << "Is tridiag ? " << check_tridiag(in) << "\n";

    eis_eig(ref);
    return 0;
}