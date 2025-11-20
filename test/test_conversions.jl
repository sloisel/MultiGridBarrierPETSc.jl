using Test
using MPI

# Use MultiGridBarrierPETSc initializer (MPI then PETSc)
using MultiGridBarrierPETSc
MultiGridBarrierPETSc.Init()

# Now load dependencies for tests
using SafePETSc
using MultiGridBarrier
using LinearAlgebra
using SparseArrays
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Set DEBUG to false to enable PETSc solves with MUMPS direct solver
SafePETSc.DEBUG[] = false

if rank == 0
    println("[DEBUG] Conversion tests starting")
    flush(stdout)
end

# Keep output tidy and aggregate at the end
ts = @testset QuietTestSet "Conversion tests" begin

# ============================================================================
# Test geometry_petsc_to_native
# ============================================================================
if rank == 0
    println("[DEBUG] Testing geometry_petsc_to_native")
    flush(stdout)
end

# Create a native geometry and test round-trip conversion
g_native_orig = fem2d(; maxh=0.3)
g_petsc = geometry_native_to_petsc(g_native_orig)
g_native_back = geometry_petsc_to_native(g_petsc)

# Test round-trip preserves values
@test g_native_back.x ≈ g_native_orig.x
@test g_native_back.w ≈ g_native_orig.w

# Test operators
for key in keys(g_native_orig.operators)
    @test g_native_back.operators[key] ≈ g_native_orig.operators[key]
end

# Test subspaces
for key in keys(g_native_orig.subspaces)
    for i in 1:length(g_native_orig.subspaces[key])
        @test g_native_back.subspaces[key][i] ≈ g_native_orig.subspaces[key][i]
    end
end

# Test refine and coarsen
for i in 1:length(g_native_orig.refine)
    @test g_native_back.refine[i] ≈ g_native_orig.refine[i]
end
for i in 1:length(g_native_orig.coarsen)
    @test g_native_back.coarsen[i] ≈ g_native_orig.coarsen[i]
end

# Test types are correct
@test typeof(g_native_back.x) == Matrix{Float64}
@test typeof(g_native_back.w) == Vector{Float64}
for key in keys(g_native_back.operators)
    @test typeof(g_native_back.operators[key]) == SparseMatrixCSC{Float64, Int}
end

# Test dimensions preserved
@test size(g_native_back.x) == size(g_native_orig.x)
@test length(g_native_back.w) == length(g_native_orig.w)
@test length(g_native_back.operators) == length(g_native_orig.operators)

# Test discretization preserved
@test g_native_back.discretization === g_native_orig.discretization

# Test numerical accuracy
@test isapprox(g_native_back.x, g_native_orig.x, rtol=1e-14)
@test isapprox(g_native_back.w, g_native_orig.w, rtol=1e-14)
@test all(isfinite.(g_native_back.x))
@test all(isfinite.(g_native_back.w))

# ============================================================================
# Test sol_petsc_to_native
# ============================================================================
if rank == 0
    println("[DEBUG] Testing sol_petsc_to_native")
    flush(stdout)
end

# Create a PETSc solution
g_petsc_solve = fem2d_petsc(Float64; maxh=0.3)
sol_petsc = amgb(g_petsc_solve; p=2.0, verbose=false)
sol_native = sol_petsc_to_native(sol_petsc)

# Test z conversion
@test sol_native.z ≈ SafePETSc.J(sol_petsc.z)

# Test log preserved
@test sol_native.log == sol_petsc.log

# Test geometry conversion
@test sol_native.geometry.x ≈ SafePETSc.J(sol_petsc.geometry.x)
@test sol_native.geometry.w ≈ SafePETSc.J(sol_petsc.geometry.w)

# Test types are native
@test typeof(sol_native.z) <: Union{Matrix{Float64}, Vector{Float64}}
@test typeof(sol_native.geometry.x) == Matrix{Float64}
@test typeof(sol_native.geometry.w) == Vector{Float64}
for key in keys(sol_native.geometry.operators)
    @test typeof(sol_native.geometry.operators[key]) == SparseMatrixCSC{Float64, Int}
end

# Test SOL_main NamedTuple conversion
@test sol_native.SOL_main !== nothing
@test haskey(sol_native.SOL_main, :z)
@test haskey(sol_native.SOL_main, :c)
@test haskey(sol_native.SOL_main, :its)
@test haskey(sol_native.SOL_main, :ts)

# Test no PETSc types in NamedTuples
for (name, value) in pairs(sol_native.SOL_main)
    if isa(value, AbstractArray)
        @test !(typeof(value) <: Mat)
        @test !(typeof(value) <: Vec)
    end
end

if sol_native.SOL_feasibility !== nothing
    for (name, value) in pairs(sol_native.SOL_feasibility)
        if isa(value, AbstractArray)
            @test !(typeof(value) <: Mat)
            @test !(typeof(value) <: Vec)
        end
    end
end

# Test consistency with direct geometry conversion
g_native_direct = geometry_petsc_to_native(sol_petsc.geometry)
@test sol_native.geometry.x ≈ g_native_direct.x
@test sol_native.geometry.w ≈ g_native_direct.w

# Test numerical validity
@test all(isfinite.(sol_native.z))
@test all(isfinite.(sol_native.SOL_main.z))
@test all(isfinite.(sol_native.SOL_main.c))
@test isfinite(sol_native.SOL_main.t_elapsed)

if rank == 0
    println("[DEBUG] All conversion tests completed")
    flush(stdout)
end

end  # End of QuietTestSet

# Aggregate per-rank counts and print a single summary on root
local_counts = [
    get(ts.counts, :pass, 0),
    get(ts.counts, :fail, 0),
    get(ts.counts, :error, 0),
    get(ts.counts, :broken, 0),
    get(ts.counts, :skip, 0),
]

global_counts = similar(local_counts)
MPI.Allreduce!(local_counts, global_counts, +, comm)

if rank == 0
    println("Test Summary: Conversion tests (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Conversion test file completed successfully")
    flush(stdout)
end
