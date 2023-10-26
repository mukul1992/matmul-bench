#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_MultiFab.H>
#include <AMReX_PlotFileUtil.H>

#include "matmul_init.H"
#include "matmul_print.H"

using namespace amrex;

void add (FArrayBox const& fab_a, FArrayBox const& fab_b, MultiFab& mfc)
{
    Array4<Real const> const &a = fab_a.const_array();
    Array4<Real const> const &b = fab_b.const_array();

    for (MFIter mfi(mfc); mfi.isValid(); ++mfi) {
        const Box &bx = mfi.validbox();
        Print() << "*** the iter box is: " << bx << "\n";
        //Array4<Real const> const &a = mfa.const_array();
        //Array4<Real const> const &b = mfb.const_array();
        Array4<Real> const& c = mfc.array(mfi);
        BL_PROFILE("matmul"); // for NVIDIA Nsight Compute
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int ) noexcept
        {
            c(i, j, 0) = a(i, j, 0) + b(i, j, 0);

        });
    }
}

void matmul (FArrayBox const& fab_a, FArrayBox const& fab_b, MultiFab& mfc,
                    int inner_dim)
{
    Array4<Real const> const &a = fab_a.const_array();
    Array4<Real const> const &b = fab_b.const_array();

    for (MFIter mfi(mfc); mfi.isValid(); ++mfi) {
        const Box &bx = mfi.validbox();
        //Print() << "*** the iter box is: " << bx << "\n";
        //Array4<Real const> const &a = mfa.const_array();
        //Array4<Real const> const &b = mfb.const_array();
        Array4<Real> const& c = mfc.array(mfi);
        BL_PROFILE("matmul"); // for NVIDIA Nsight Compute
        amrex::ParallelFor(bx, [=] AMREX_GPU_DEVICE(int i, int j, int ) noexcept
        {
            Real sum{0.0};
            for (int inner=0; inner<=inner_dim; inner++) {
                sum += a(i, inner, 0) * b(inner, j, 0);
            }
            c(i, j, 0) = sum;
        });
    }
}

int main(int argc, char* argv[])
{
    Real t;
    amrex::Initialize(argc,argv);
    // need to abort if number of processes > 1
    {
        BL_PROFILE("main()");
        Print() << "Matrix multiplication benchmark" << "\n";

        //int max_grid_size = 256;
        Vector<int> grid_size(AMREX_SPACEDIM);
        Vector<int> matrixA_ndim(AMREX_SPACEDIM);
        Vector<int> matrixB_ndim(AMREX_SPACEDIM);
        //IntVect grid_size{256,128};
        //IntVect matrixA_ndim{256,256};
        //IntVect matrixB_ndim{256,256};
        {
            ParmParse pp;
            //pp.query("max_grid_size", max_grid_size);
            pp.queryarr("grid_size", grid_size, 0, AMREX_SPACEDIM);

            pp.queryarr("matrixA_ndim", matrixA_ndim, 0, AMREX_SPACEDIM);
            pp.queryarr("matrixB_ndim", matrixB_ndim, 0, AMREX_SPACEDIM);

        }
        // need to abort if dimensions are not compliant

        IntVect matrixC_ndim{matrixA_ndim[0], matrixB_ndim[1]};

        Box domainA(IntVect(0), IntVect(matrixA_ndim)-1);
        Box domainB(IntVect(0), IntVect(matrixB_ndim)-1);
        Box domainC(IntVect(0), matrixC_ndim-1);

        Print() << "matrix A dimensions: " << domainA << "\n";
        Print() << "matrix B dimensions: " << domainB << "\n";
        Print() << "matrix C dimensions: " << domainC << "\n";

        FArrayBox fab_A(domainA, 1);
        FArrayBox fab_B(domainB, 1);
        //FArrayBox fab_C(domainC, 1);

        BoxArray ba_C(domainC);
        //ba_C.maxSize(max_grid_size);
        ba_C.maxSize(IntVect(grid_size));
        DistributionMapping dm_C{ba_C};
        MultiFab mf_C(ba_C,dm_C,1,0);
        Print() << "matrix C grids: \n" << ba_C << "\n\n";

        initMatrix(fab_A);
        initMatrix(fab_B);

        matmul(fab_A, fab_B, mf_C, matrixA_ndim[1]-1);

        Print() << "matrix A \n";
        printMatrix(fab_A);
        Print() << "matrix B \n";
        printMatrix(fab_B);
        Print() << "matrix C \n";
        RealBox real_box({-1.0,-1.0},{1.0,1.0});
        Geometry geom(domainC, real_box, 0, {0,0});
        WriteSingleLevelPlotfile ("plt_matC", mf_C, {"val"}, geom, 0.0, 0);

    }

    amrex::Finalize();
}
