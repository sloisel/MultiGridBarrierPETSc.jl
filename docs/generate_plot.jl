#!/usr/bin/env julia
#
# Generate plot for documentation
# Run with: mpiexec -n 4 julia --project docs/generate_plot.jl
#

using MPI
using SafePETSc
using MultiGridBarrierPETSc
using MultiGridBarrier
using PyPlot
SafePETSc.Init()

# Solve with PETSc (L=3 for reasonable size/speed)
sol_petsc = fem2d_petsc_solve(Float64; L=3, p=1.0, verbose=false)

# Convert to native for plotting
sol_native = sol_petsc_to_native(sol_petsc)

# Only rank 0 creates the plot
rank = MPI.Comm_rank(MPI.COMM_WORLD)
if rank == 0
    figure(figsize=(8, 6))
    plot(sol_native)
    title("2D p-Laplace Solution (L=3, p=1.0)")
    tight_layout()
    savefig("docs/src/fem2d_petsc.svg")
    close()
    println("Plot saved to docs/src/fem2d_petsc.svg")
end
