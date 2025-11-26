#!/usr/bin/env julia
#
# Basic Solve Example for MultiGridBarrierPETSc.jl
#
# This example demonstrates the simplest workflow:
# 1. Initialize MultiGridBarrierPETSc
# 2. Solve with PETSc distributed types
# 3. Convert solution to native types
# 4. Display results
#
# Run with: julia -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) examples/basic_solve.jl`)'
#

using MultiGridBarrierPETSc
MultiGridBarrierPETSc.Init()  # Must initialize MPI and PETSc first

using SafePETSc  # For io0() and MPI
using MultiGridBarrier
using LinearAlgebra

println(io0(), "="^70)
println(io0(), "Basic Solve Example - MultiGridBarrierPETSc.jl")
println(io0(), "="^70)

# Get MPI information
rank = MPI.Comm_rank(MPI.COMM_WORLD)
nranks = MPI.Comm_size(MPI.COMM_WORLD)
println(io0(), "Running on $nranks MPI ranks\n")

# Problem parameters
L = 2          # Refinement levels (L=2 is fast for demonstration)
p = 1.0        # Barrier power parameter
maxh = 0.3     # Maximum mesh size

println(io0(), "Problem Parameters:")
println(io0(), "  Refinement levels (L): $L")
println(io0(), "  Barrier parameter (p): $p")
println(io0(), "  Max mesh size (maxh):  $maxh")
println(io0(), "")

# Solve with PETSc distributed types (collective operation)
println(io0(), "Solving with PETSc distributed types...")
sol_petsc = fem2d_petsc_solve(Float64; L=L, maxh=maxh, p=p, verbose=true)

# Convert solution to native Julia types (collective operation)
println(io0(), "\nConverting solution to native types...")
sol_native = petsc_to_native(sol_petsc)

# Display results (only on rank 0)
println(io0(), "")
println(io0(), "="^70)
println(io0(), "Solution Summary:")
println(io0(), "="^70)

# Solution dimensions
z_size = size(sol_native.z)
println(io0(), "Solution matrix size: $(z_size[1]) × $(z_size[2])")

# Convergence information
n_newton_steps = sum(sol_native.SOL_main.its)
println(io0(), "Total Newton steps: $n_newton_steps")
println(io0(), "Elapsed time: $(sol_native.SOL_main.t_elapsed) seconds")

# Solution statistics
z_norm = norm(sol_native.z)
z_min = minimum(sol_native.z)
z_max = maximum(sol_native.z)
println(io0(), "Solution norm: $z_norm")
println(io0(), "Solution range: [$z_min, $z_max]")

println(io0(), "")
println(io0(), "="^70)
println(io0(), "Example completed successfully!")
println(io0(), "="^70)

# Optional: Save solution for visualization
if rank == 0
    println("\nTo visualize this solution, use:")
    println("  using MultiGridBarrier, PyPlot")
    println("  plot(sol_native)")
    println("  savefig(\"basic_solve_solution.png\")")
end
