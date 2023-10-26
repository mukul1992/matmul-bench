#include "matmul_print.H"

using namespace amrex;

void printMatrix(FArrayBox &fab)
{
    Array4<Real> const &mat = fab.array();
    const Box &bx = fab.box();
    int imax = bx.bigEnd(0);
    int jmax = bx.bigEnd(1);
    Print() << "Matrix dimensions: " << imax+1 << " x " << jmax+1 << "\n";
    for(int i=0; i<=imax; i++) {
        for(int j=0; j<=jmax; j++) {
            Print() << mat(i,j,0) << " ";
        }
        Print() << "\n";
    }
    Print() << "\n";
}