/*Dominant eigen vector calculation*/
#include <iostream>
#include <armadillo>
#include <time.h>

using namespace arma;
using namespace std;

int main(int argc, char *argv[])
{
    const int n = atoi(argv[1]);
    const int m = atoi(argv[2]);
    clock_t begin, end;
    mat A = randu<mat>(m, n);
    vec x = randu<vec>(n);
    begin = clock();
    mat D = cov(A);
    end = clock();
    cout << double(end - begin) / CLOCKS_PER_SEC <<endl;
    //D.print(std::cout);
    return 0;
}
