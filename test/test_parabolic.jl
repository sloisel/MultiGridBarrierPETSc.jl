#!/usr/bin/env julia
# Test script to see if parabolic_solve works with PETSc types

using SafePETSc
SafePETSc.Init()
using SafePETSc: io0

using MPI
using MultiGridBarrierPETSc
using MultiGridBarrier

MultiGridBarrierPETSc.Init()

comm = MPI.COMM_WORLD

println(io0(), "\n", "="^60)
println(io0(), "Testing parabolic_solve with PETSc types")
println(io0(), "="^60, "\n")

# First, let's verify that static fem1d_petsc_solve works
println(io0(), "[TEST 1] Testing static fem1d_petsc_solve (sanity check)...")

try
    sol_static = fem1d_petsc_solve(Float64; L=2, p=1.0, verbose=false)
    println(io0(), "[OK] Static fem1d_petsc_solve works!")
    println(io0(), "     Solution size: ", size(sol_static.z))
catch e
    println(io0(), "[FAIL] Static fem1d_petsc_solve failed:")
    showerror(io0(), e, catch_backtrace())
end

MPI.Barrier(comm)

# Now let's try parabolic_solve with a PETSc geometry
println(io0(), "\n[TEST 2] Testing parabolic_solve with PETSc geometry...")

try
    # Create a PETSc-based 1D geometry
    g_petsc = fem1d_petsc(Float64; L=2)

    println(io0(), "     Created PETSc geometry")
    println(io0(), "     x type: ", typeof(g_petsc.x))
    println(io0(), "     w type: ", typeof(g_petsc.w))

    # Try calling parabolic_solve
    sol_parabolic = parabolic_solve(g_petsc; h=0.5, p=1.0, verbose=false)

    println(io0(), "[OK] parabolic_solve with PETSc geometry works!")
    println(io0(), "     Solution type: ", typeof(sol_parabolic))
    println(io0(), "     u size: ", size(sol_parabolic.u))
catch e
    println(io0(), "[FAIL] parabolic_solve with PETSc geometry failed:")
    showerror(io0(), e)
    println(io0(), "\n\nFull stacktrace:")
    for (exc, bt) in current_exceptions()
        showerror(io0(), exc, bt)
        println(io0())
    end
end

MPI.Barrier(comm)

# Also try 2D case
println(io0(), "\n[TEST 3] Testing parabolic_solve with 2D PETSc geometry...")

try
    # Create a PETSc-based 2D geometry (small for quick test)
    g_petsc_2d = fem2d_petsc(Float64; L=1)

    println(io0(), "     Created 2D PETSc geometry")

    # Try calling parabolic_solve
    sol_parabolic_2d = parabolic_solve(g_petsc_2d; h=0.5, p=1.0, verbose=false)

    println(io0(), "[OK] parabolic_solve with 2D PETSc geometry works!")
    println(io0(), "     u size: ", size(sol_parabolic_2d.u))
catch e
    println(io0(), "[FAIL] parabolic_solve with 2D PETSc geometry failed:")
    showerror(io0(), e)
    println(io0(), "\n\nFull stacktrace:")
    for (exc, bt) in current_exceptions()
        showerror(io0(), exc, bt)
        println(io0())
    end
end

MPI.Barrier(comm)

println(io0(), "\n", "="^60)
println(io0(), "Test completed")
println(io0(), "="^60, "\n")

# Clean up PETSc objects
SafePETSc.SafeMPI.check_and_destroy!()

MPI.Barrier(comm)
