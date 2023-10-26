# Matrix Multiplication with AMReX

This code performs matrix multiplication $C = A \cdot B$ using AMReX data structures and the `ParallelFor` kernel launch routine.

AMReX data classes and routines are typically designed for storing field data (and particle data) for block-structured mesh based simulations and not for storing matrices and performing matrix operations. Hence this code should not be used to evaluate the general effectiveness of AMReX.

## Inputs 

The code requires the dimensions of matrices $A$ and $B$ as inputs.

It also requires the grid size of the output matrix $C$ over which each `MFIter` loop operates and launches a kernel in a separate stream. For example, for `grid_size = 256 256`, the ouput matrix $C$ is partitioned into grids of maximum size 256 each along the rows and columns. Each `MFIter` loop operates on a single grid and launches the GPU kernel via `ParallelFor` to compute the result at each index of this grid. 

Note: This grid size should not be confused with the "tile size" that is often used to assign work to individual thread blocks in GPU matrix multiplication kernels. In this code, the specific kernel launch configuration is left to the default AMReX implementation. 

Setting the verbosity `amr.v = 2` prints out the dimensions of these grids.
A sample `inputs` file is included. 
