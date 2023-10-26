#include "matmul_init.H"

using namespace amrex;

void initMatrix(MultiFab& mf)
{
    for (MFIter mfi(mf); mfi.isValid(); ++mfi) {
        const Box& bx = mfi.validbox();
        Array4<Real> const& mat = mf.array(mfi);
        amrex::ParallelForRNG(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int , RandomEngine const& engine)
        {
            mat(i,j,0) = amrex::Random(engine);
        });
    }
}

void initMatrix(FArrayBox &fab)
{
    Array4<Real> const &mat = fab.array();
    const Box &bx = fab.box();
    amrex::ParallelForRNG(bx,
    [=] AMREX_GPU_DEVICE(int i, int j, int, RandomEngine const &engine)
    {
        mat(i, j, 0) = amrex::Random(engine);
    });
}