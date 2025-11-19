using Test
using MPI

# Helper to run individual test files with MPI
function run_mpi_test(test_file::AbstractString; nprocs::Integer=4, expect_success::Bool=true)
    test_path = joinpath(@__DIR__, test_file)
    mpiexec_cmd = MPI.mpiexec()
    test_proj = Base.active_project()
    cmd = `$mpiexec_cmd -n $nprocs $(Base.julia_cmd()) --project=$test_proj $test_path`
    proc = run(ignorestatus(cmd))
    ok = success(proc)
    if ok != expect_success
        @info "MPI test exit status mismatch" test_file=test_file ok=ok expect_success=expect_success exitcode=proc.exitcode
    end
    @test ok == expect_success
end

@testset "MultiGridBarrierPETSc.jl" begin
    @testset "Helper functions" begin
        run_mpi_test("test_helpers.jl")
    end

    @testset "Quick integration test" begin
        run_mpi_test("test_quick.jl")
    end
end
