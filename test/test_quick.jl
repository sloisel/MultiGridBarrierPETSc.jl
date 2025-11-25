using Test
using MPI

# Use MultiGridBarrierPETSc initializer (MPI then PETSc)
using MultiGridBarrierPETSc
MultiGridBarrierPETSc.Init()

# Now load dependencies for tests
using SafePETSc
using MultiGridBarrier
using LinearAlgebra
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

if rank == 0
    println("[DEBUG] Quick integration test starting")
    flush(stdout)
end

# Keep output tidy and aggregate at the end
ts = @testset QuietTestSet "Quick integration tests" begin

for L in 1:3
    if rank == 0
        println("[DEBUG] Test L=$L: Comparing PETSc and native fem2d_petsc_solve")
        flush(stdout)
    end

    # Solve with PETSc distributed types
    sol_petsc = MultiGridBarrierPETSc.fem2d_petsc_solve(Float64; L=L, p=1.0, verbose=false)

    # Solve with native Julia types (sequential, no logfile needed)
    sol_native = MultiGridBarrier.fem2d_solve(Float64; L=L, p=1.0, verbose=false)

    # Convert PETSc solution to native for comparison
    z_petsc = Matrix(sol_petsc.z)
    z_native = sol_native.z

    # Compare solutions
    diff = norm(z_petsc - z_native)
    rel_diff = diff / norm(z_native)

    if rank == 0
        println("[DEBUG]   L=$L: Relative difference = $rel_diff")
        flush(stdout)
    end

    # Note: Differences can occur due to inexact PETSc iterative solves
    tol = 1e-6
    @test rel_diff < tol

    SafePETSc.SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)
end

if rank == 0
    println("[DEBUG] All quick tests completed")
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
    println("Test Summary: Quick integration tests (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

# Ensure all ranks reach this point before deciding outcome
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Quick integration test file completed successfully")
    flush(stdout)
end
