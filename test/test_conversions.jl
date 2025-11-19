using Test
using MPI
using SafePETSc
using MultiGridBarrierPETSc
using MultiGridBarrier
using LinearAlgebra
using SparseArrays

# Initialize MPI and PETSc
SafePETSc.Init()

# Get MPI info
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# Create a small native geometry
if rank == 0
    println("[DEBUG] Testing geometry conversion round-trip")
end

# Create a simple native geometry using fem2d
g_native_original = fem2d(; maxh=0.3)

# Convert to PETSc
g_petsc = geometry_native_to_petsc(g_native_original)

# Convert back to native
g_native_converted = geometry_petsc_to_native(g_petsc)

# Test that the conversions preserve values
@testset "Geometry round-trip conversion" begin
    # Test x (coordinates)
    @test g_native_converted.x ≈ g_native_original.x

    # Test w (weights)
    @test g_native_converted.w ≈ g_native_original.w

    # Test operators
    for key in keys(g_native_original.operators)
        @test g_native_converted.operators[key] ≈ g_native_original.operators[key]
    end

    # Test subspaces
    for key in keys(g_native_original.subspaces)
        for i in 1:length(g_native_original.subspaces[key])
            @test g_native_converted.subspaces[key][i] ≈ g_native_original.subspaces[key][i]
        end
    end

    # Test refine and coarsen
    for i in 1:length(g_native_original.refine)
        @test g_native_converted.refine[i] ≈ g_native_original.refine[i]
    end

    for i in 1:length(g_native_original.coarsen)
        @test g_native_converted.coarsen[i] ≈ g_native_original.coarsen[i]
    end

    if rank == 0
        println("[DEBUG] Geometry round-trip conversion passed")
    end
end

# Test AMGBSOL conversion
if rank == 0
    println("[DEBUG] Testing AMGBSOL conversion")
end

# Create a PETSc geometry and solve
g_petsc_solve = fem2d_petsc(Float64; maxh=0.3)
sol_petsc = amgb(g_petsc_solve; p=2.0, verbose=false)

# Convert solution to native
sol_native = sol_petsc_to_native(sol_petsc)

@testset "AMGBSOL conversion" begin
    # Test z
    @test sol_native.z ≈ SafePETSc.J(sol_petsc.z)

    # Test log (should be identical string)
    @test sol_native.log == sol_petsc.log

    # Test geometry conversion
    @test sol_native.geometry.x ≈ SafePETSc.J(sol_petsc.geometry.x)
    @test sol_native.geometry.w ≈ SafePETSc.J(sol_petsc.geometry.w)

    # Test that native types are correct
    @test typeof(sol_native.z) <: Union{Matrix{Float64}, Vector{Float64}}
    @test typeof(sol_native.geometry.x) == Matrix{Float64}
    @test typeof(sol_native.geometry.w) == Vector{Float64}

    for key in keys(sol_native.geometry.operators)
        @test typeof(sol_native.geometry.operators[key]) == SparseMatrixCSC{Float64, Int}
    end

    if rank == 0
        println("[DEBUG] AMGBSOL conversion passed")
    end
end

# Clean up
SafePETSc.SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

if rank == 0
    println("[DEBUG] All conversion tests passed")
end
