#!/usr/bin/env julia
#
# Generate plot for documentation
# Run with: julia --project -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project docs/generate_plot.jl`)'
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
sol_native = petsc_to_native(sol_petsc)

# Only rank 0 creates the plot
rank = MPI.Comm_rank(MPI.COMM_WORLD)
if rank == 0
    fig = figure(figsize=(8, 6))
    plot(sol_native)
    suptitle("2D p-Laplace Solution (L=3, p=1.0)")
    # Hide ticks on outer 2D axes, keep the frame
    for ax in fig.axes
        if !hasproperty(ax, :zaxis)  # Not a 3D axis
            ax.set_xticks([])
            ax.set_yticks([])
        end
    end
    tight_layout()
    savefig("docs/src/fem2d_petsc.svg")
    close()
    println("Plot saved to docs/src/fem2d_petsc.svg")
end
