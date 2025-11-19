# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

MultiGridBarrierPETSc.jl is a Julia package that bridges MultiGridBarrier.jl and SafePETSc.jl, enabling distributed PETSc computations for multigrid barrier methods. It extends MultiGridBarrier's API to work with PETSc's distributed Mat and Vec types.

## Essential Commands

### Running Tests
```bash
# Run all tests with 4 MPI ranks (recommended)
julia --project=. -e 'using Pkg; Pkg.test()'

# Run individual test modules
julia --project=. -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project=$(Base.active_project()) test/test_helpers.jl`)'
```

### Development Setup
```bash
cd path/to/MultiGridBarrierPETSc.jl
julia --project
julia> using Pkg
julia> Pkg.instantiate()
```

## Architecture Overview

### Core Module Structure
The main module (`src/MultiGridBarrierPETSc.jl`) has three primary responsibilities:

1. **API Extensions (lines 45-91)**: Extends MultiGridBarrier functions to work with PETSc types
   - `amgb_zeros`, `amgb_all_isfinite`, `amgb_hcat`, `amgb_diag`, `amgb_blockdiag`, `map_rows`
   - Dispatches based on Mat/Vec storage type (MPIAIJ for sparse, MPIDENSE for dense)

2. **Geometry Conversion (lines 97-180)**: `geometry_native_to_petsc()` converts native Julia arrays to PETSc distributed types
   - Coordinates (x) and weights (w) → MPIDENSE storage
   - Operators and subspace matrices → MPIAIJ sparse storage
   - Preserves multigrid hierarchy (refine/coarsen matrices)

3. **Public API (lines 186-246)**: Two main user-facing functions
   - `fem2d_petsc()`: Creates PETSc-based Geometry from fem2d parameters
   - `fem2d_petsc_solve()`: End-to-end solve using amgb with PETSc types

### Test Infrastructure
Tests follow SafePETSc patterns with 4-way MPI execution:

- **mpi_test_harness.jl**: QuietTestSet implementation for clean MPI test output
- **runtests.jl**: Orchestrates test execution with precompilation and `run_mpi_test()` helper
- Each test file follows pattern: Initialize → QuietTestSet → Aggregate results across ranks → Exit handling

Key testing pattern:
```julia
ts = @testset MPITestHarness.QuietTestSet "..." begin
    # Tests with rank 0 debug output
    if rank == 0
        println("[DEBUG] Test description")
    end
    @test ...
    SafePETSc.SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)
end
# Aggregate results with MPI.Allreduce
```

### Critical Type Mappings
- Native `Matrix{T}` → `Mat{T, MPIDENSE}` (dense distributed)
- Native `SparseMatrixCSC{T}` → `Mat{T, MPIAIJ}` (sparse distributed)
- Native `Vector{T}` → `Vec{T, MPIDENSE}` (distributed vector)
- Geometry fields maintain consistent PETSc types after conversion

### Dependencies
- **MultiGridBarrier.jl** v0.11.25: Provides base multigrid barrier solver
- **SafePETSc.jl**: Provides PETSc bindings with automatic memory management
- **MPI.jl**: Required for distributed computation

### MPI Initialization
All code using this package must initialize SafePETSc first:
```julia
using SafePETSc
SafePETSc.Init()  # Initializes both MPI and PETSc
using MultiGridBarrierPETSc
```

### Key Invariants
- All MPI operations are collective (all ranks must participate)
- Geometry conversion preserves dimensions and values exactly
- PETSc types use specific storage formats: MPIDENSE for geometry data, MPIAIJ for operators
- Tests always run with 4 MPI ranks for consistency

### Git and github.

- Do not automatically commit changes into git unless asked by the user.
- Do not automatically push commits to github unless asked by the user.
- When the user gives permission to git commit and/or push once, that is not carte blanche to do it later.
- Do not tag releases, TagBot will tag on github.

### IO

Use io0() from SafePETSc to print or otherwise perform io from a single rank. It can be used as follows: `println(io0(),"Hello from rank 0!"). The Vec{T,Prefix} and Mat{T,Prefix} implement `show(...)` methods, but these are collective. Therefore, you can do `println(io0(),A)` and it will print the Vec or Mat once on rank 0.

## Directory structure
- Source files for the library are in src/
- Test scripts are in test/
- User-centric examples are in examples/. The number of examples should be modest, not to exceed 10. If there are more than 10, consult the user on what to do. There should be at most one example per main feature.
- Documentation with Documenter is in docs/. Make sure the documentation builds, is free of errors and warnings as much as possible. All user-visible API should be documented, internal functions should not be documented. The User Guide should describe all main features.
