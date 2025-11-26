#!/usr/bin/env julia
#
# Round-Trip Conversion Example for MultiGridBarrierPETSc.jl
#
# This example demonstrates:
# 1. Creating a native geometry
# 2. Converting to PETSc distributed types
# 3. Solving with PETSc types
# 4. Converting everything back to native types
# 5. Verifying accuracy of conversions
#
# Run with: julia -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) examples/roundtrip_conversion.jl`)'
#

using MultiGridBarrierPETSc
MultiGridBarrierPETSc.Init()  # Must initialize MPI and PETSc first

using SafePETSc  # For io0() and MPI
using MultiGridBarrier
using LinearAlgebra
using SparseArrays

println(io0(), "="^70)
println(io0(), "Round-Trip Conversion Example - MultiGridBarrierPETSc.jl")
println(io0(), "="^70)

# Get MPI information
rank = MPI.Comm_rank(MPI.COMM_WORLD)
nranks = MPI.Comm_size(MPI.COMM_WORLD)
println(io0(), "Running on $nranks MPI ranks\n")

# Step 1: Create native geometry (all ranks create identical copy)
println(io0(), "Step 1: Creating native geometry...")
g_native = fem2d(; maxh=0.3, L=2)

# Get dimensions
n_points = size(g_native.x, 2)
n_weights = length(g_native.w)
println(io0(), "  Number of points: $n_points")
println(io0(), "  Number of weights: $n_weights")
println(io0(), "  Operators: $(keys(g_native.operators))")
println(io0(), "")

# Step 2: Convert to PETSc distributed types (collective)
println(io0(), "Step 2: Converting to PETSc distributed types...")
g_petsc = native_to_petsc(g_native)
println(io0(), "  Conversion complete")
println(io0(), "  Type of x: $(typeof(g_petsc.x))")
println(io0(), "  Type of w: $(typeof(g_petsc.w))")
println(io0(), "  Type of operators: $(typeof(g_petsc.operators[:id]))")
println(io0(), "")

# Step 3: Solve with PETSc types (collective)
println(io0(), "Step 3: Solving with PETSc distributed types...")
sol_petsc = amgb(g_petsc; p=1.5, verbose=true)
println(io0(), "  Solution obtained")
println(io0(), "")

# Step 4: Convert geometry back to native (collective)
println(io0(), "Step 4: Converting geometry back to native types...")
g_back = petsc_to_native(g_petsc)
println(io0(), "  Geometry conversion complete")
println(io0(), "")

# Step 5: Convert solution to native (collective)
println(io0(), "Step 5: Converting solution to native types...")
sol_native = petsc_to_native(sol_petsc)
println(io0(), "  Solution conversion complete")
println(io0(), "")

# Step 6: Verify accuracy of round-trip conversion
println(io0(), "="^70)
println(io0(), "Verification: Checking Round-Trip Accuracy")
println(io0(), "="^70)

# Check geometry coordinates (x)
x_diff = norm(g_native.x - g_back.x)
println(io0(), "Coordinates (x) difference norm: $x_diff")

# Check weights (w)
w_diff = norm(g_native.w - g_back.w)
println(io0(), "Weights (w) difference norm:     $w_diff")

# Check operators
println(io0(), "\nOperator differences:")
for key in sort(collect(keys(g_native.operators)))
    op_native = g_native.operators[key]
    op_back = g_back.operators[key]
    op_diff = norm(op_native - op_back)
    println(io0(), "  $key: $op_diff")
end

# Check subspaces
println(io0(), "\nSubspace matrix differences:")
for key in sort(collect(keys(g_native.subspaces)))
    for (i, (s_native, s_back)) in enumerate(zip(g_native.subspaces[key], g_back.subspaces[key]))
        s_diff = norm(s_native - s_back)
        println(io0(), "  $key[$i]: $s_diff")
    end
end

# Check refine matrices
println(io0(), "\nRefine matrix differences:")
for (i, (r_native, r_back)) in enumerate(zip(g_native.refine, g_back.refine))
    r_diff = norm(r_native - r_back)
    println(io0(), "  refine[$i]: $r_diff")
end

# Check coarsen matrices
println(io0(), "\nCoarsen matrix differences:")
for (i, (c_native, c_back)) in enumerate(zip(g_native.coarsen, g_back.coarsen))
    c_diff = norm(c_native - c_back)
    println(io0(), "  coarsen[$i]: $c_diff")
end

# Overall assessment
println(io0(), "")
println(io0(), "="^70)

# Define tolerance
tol = 1e-10
max_diff = maximum([
    x_diff, w_diff,
    maximum(norm(g_native.operators[k] - g_back.operators[k]) for k in keys(g_native.operators)),
    maximum(norm(g_native.refine[i] - g_back.refine[i]) for i in 1:length(g_native.refine)),
    maximum(norm(g_native.coarsen[i] - g_back.coarsen[i]) for i in 1:length(g_native.coarsen))
])

if max_diff < tol
    println(io0(), "✓ Round-trip conversion PASSED!")
    println(io0(), "  Maximum difference: $max_diff (< $tol)")
else
    println(io0(), "✗ Round-trip conversion FAILED!")
    println(io0(), "  Maximum difference: $max_diff (≥ $tol)")
end

println(io0(), "="^70)
println(io0(), "\nSolution Statistics:")
println(io0(), "  Total Newton steps: $(sum(sol_native.SOL_main.its))")
println(io0(), "  Elapsed time: $(sol_native.SOL_main.t_elapsed) seconds")
println(io0(), "  Solution norm: $(norm(sol_native.z))")
println(io0(), "")
println(io0(), "="^70)
println(io0(), "Example completed successfully!")
println(io0(), "="^70)
