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
    println("[DEBUG] Parabolic integration test starting")
    flush(stdout)
end

# Keep output tidy and aggregate at the end
ts = @testset QuietTestSet "Parabolic integration tests" begin

# Test 1D parabolic solve with PETSc geometry
if rank == 0
    println("[DEBUG] Test 1D parabolic solve with PETSc geometry")
    flush(stdout)
end

g_petsc_1d = fem1d_petsc(Float64; L=2)
sol_parabolic_1d = parabolic_solve(g_petsc_1d; h=0.5, p=1.0, verbose=false)

@test length(sol_parabolic_1d.ts) == 3  # t=0, 0.5, 1.0
@test length(sol_parabolic_1d.u) == 3

if rank == 0
    println("[DEBUG]   1D parabolic: $(length(sol_parabolic_1d.ts)) timesteps")
    flush(stdout)
end

SafePETSc.SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 2D parabolic solve with PETSc geometry
if rank == 0
    println("[DEBUG] Test 2D parabolic solve with PETSc geometry")
    flush(stdout)
end

g_petsc_2d = fem2d_petsc(Float64; L=1)
sol_parabolic_2d = parabolic_solve(g_petsc_2d; h=0.5, p=1.0, verbose=false)

@test length(sol_parabolic_2d.ts) == 3  # t=0, 0.5, 1.0
@test length(sol_parabolic_2d.u) == 3

if rank == 0
    println("[DEBUG]   2D parabolic: $(length(sol_parabolic_2d.ts)) timesteps")
    flush(stdout)
end

SafePETSc.SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test parabolic solve with custom time parameters
if rank == 0
    println("[DEBUG] Test parabolic solve with custom time parameters")
    flush(stdout)
end

g_petsc = fem1d_petsc(Float64; L=2)
sol_custom = parabolic_solve(g_petsc; h=0.25, t0=0.0, t1=1.0, p=1.0, verbose=false)

@test length(sol_custom.ts) == 5  # t=0, 0.25, 0.5, 0.75, 1.0
@test sol_custom.ts[1] ≈ 0.0
@test sol_custom.ts[end] ≈ 1.0

if rank == 0
    println("[DEBUG]   Custom params: $(length(sol_custom.ts)) timesteps")
    flush(stdout)
end

SafePETSc.SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] All parabolic tests completed")
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
    println("Test Summary: Parabolic integration tests (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

# Ensure all ranks reach this point before deciding outcome
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Parabolic integration test file completed successfully")
    flush(stdout)
end
