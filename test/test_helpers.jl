using Test
using MPI

# Use MultiGridBarrierPETSc initializer (MPI then PETSc)
using MultiGridBarrierPETSc
MultiGridBarrierPETSc.Init()

# Now load dependencies for tests
using SafePETSc
using SafePETSc: MPIDENSE, MPIAIJ
using LinearAlgebra
using SparseArrays
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness: QuietTestSet

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

if rank == 0
    println("[DEBUG] Helper functions test starting")
    flush(stdout)
end

# Keep output tidy and aggregate at the end
ts = @testset QuietTestSet "Helper functions tests" begin

# Test 1: amgb_zeros with MPIAIJ
if rank == 0
    println("[DEBUG] Test 1: amgb_zeros with MPIAIJ sparse matrix")
    flush(stdout)
end

A_proto = SafePETSc.Mat_uniform(spzeros(10, 10); Prefix=MPIAIJ)
Z = MultiGridBarrierPETSc.amgb_zeros(A_proto, 5, 5)
@test Z isa SafePETSc.Mat
@test size(Z) == (5, 5)

SafePETSc.SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 2: amgb_zeros with MPIDENSE
if rank == 0
    println("[DEBUG] Test 2: amgb_zeros with MPIDENSE dense matrix")
    flush(stdout)
end

A_proto_dense = SafePETSc.Mat_uniform(zeros(10, 10); Prefix=MPIDENSE)
Z_dense = MultiGridBarrierPETSc.amgb_zeros(A_proto_dense, 4, 6)
@test Z_dense isa SafePETSc.Mat
@test size(Z_dense) == (4, 6)

SafePETSc.SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 3: amgb_all_isfinite with valid Vec
if rank == 0
    println("[DEBUG] Test 3: amgb_all_isfinite with finite values")
    flush(stdout)
end

v_finite = SafePETSc.Vec_uniform([1.0, 2.0, 3.0]; Prefix=MPIDENSE)
@test MultiGridBarrierPETSc.amgb_all_isfinite(v_finite) == true

SafePETSc.SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 4: amgb_all_isfinite with invalid Vec
if rank == 0
    println("[DEBUG] Test 4: amgb_all_isfinite with infinite values")
    flush(stdout)
end

v_inf = SafePETSc.Vec_uniform([1.0, Inf, 3.0]; Prefix=MPIDENSE)
@test MultiGridBarrierPETSc.amgb_all_isfinite(v_inf) == false

SafePETSc.SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 5: amgb_hcat for horizontal concatenation
if rank == 0
    println("[DEBUG] Test 5: amgb_hcat for horizontal concatenation")
    flush(stdout)
end

A = SafePETSc.Mat_uniform(sparse([1.0 2.0; 3.0 4.0]); Prefix=MPIAIJ)
B = SafePETSc.Mat_uniform(sparse([5.0 6.0; 7.0 8.0]); Prefix=MPIAIJ)
C = MultiGridBarrierPETSc.amgb_hcat(A, B)
@test C isa SafePETSc.Mat
@test size(C) == (2, 4)

SafePETSc.SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 6: amgb_diag with Vec
if rank == 0
    println("[DEBUG] Test 6: amgb_diag with Vec")
    flush(stdout)
end

A_proto = SafePETSc.Mat_uniform(spzeros(10, 10); Prefix=MPIAIJ)
v = SafePETSc.Vec_uniform([1.0, 2.0, 3.0]; Prefix=MPIDENSE)
D = MultiGridBarrierPETSc.amgb_diag(A_proto, v)
@test D isa SafePETSc.Mat
@test size(D) == (3, 3)

SafePETSc.SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 7: amgb_diag with Vector
if rank == 0
    println("[DEBUG] Test 7: amgb_diag with Vector")
    flush(stdout)
end

A_proto = SafePETSc.Mat_uniform(spzeros(10, 10); Prefix=MPIAIJ)
v_native = [1.0, 2.0, 3.0, 4.0]
D = MultiGridBarrierPETSc.amgb_diag(A_proto, v_native)
@test D isa SafePETSc.Mat
@test size(D) == (4, 4)

SafePETSc.SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 8: amgb_blockdiag
if rank == 0
    println("[DEBUG] Test 8: amgb_blockdiag for block diagonal construction")
    flush(stdout)
end

A = SafePETSc.Mat_uniform(sparse(1.0 * I(2)); Prefix=MPIAIJ)
B = SafePETSc.Mat_uniform(sparse(1.0 * I(3)); Prefix=MPIAIJ)
C = MultiGridBarrierPETSc.amgb_blockdiag(A, B)
@test C isa SafePETSc.Mat
@test size(C) == (5, 5)

SafePETSc.SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 9: map_rows
if rank == 0
    println("[DEBUG] Test 9: map_rows for row-wise operations")
    flush(stdout)
end

x = SafePETSc.Mat_uniform([1.0 2.0; 3.0 4.0]; Prefix=MPIDENSE)
result = MultiGridBarrierPETSc.map_rows(row -> sum(row), x)
@test result isa SafePETSc.Vec

SafePETSc.SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 10: Base.minimum for Vec
if rank == 0
    println("[DEBUG] Test 10: Base.minimum for Vec")
    flush(stdout)
end

v = SafePETSc.Vec_uniform([5.0, 2.0, 8.0, 1.0]; Prefix=MPIDENSE)
min_val = minimum(v)
@test min_val == 1.0

SafePETSc.SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 11: Base.maximum for Vec
if rank == 0
    println("[DEBUG] Test 11: Base.maximum for Vec")
    flush(stdout)
end

v = SafePETSc.Vec_uniform([5.0, 2.0, 8.0, 1.0]; Prefix=MPIDENSE)
max_val = maximum(v)
@test max_val == 8.0

SafePETSc.SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] All helper tests completed")
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
    println("Test Summary: Helper functions tests (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

# Ensure all ranks reach this point before deciding outcome
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] Helper functions test file completed successfully")
    flush(stdout)
end
