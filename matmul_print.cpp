#include "matmul_print.H"

using namespace amrex;

void printMatrix(MultiFab& mf, IntVect mat_ndim)
{
    MFIter mfi(mf); ++mfi;
    Array4<Real> const& mat = mf.array(mfi);

    int imax = mat_ndim[0]-1;
    int jmax = mat_ndim[1]-1;
    for(int i=0; i<=imax; i++) {
        for(int j=0; j<=jmax; j++) {
            Print() << mat(i,j,0) << " ";
        }
        Print() << "\n";
    }
    Print() << "\n";
}


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