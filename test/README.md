# Testing MultiGridBarrierPETSc.jl

MultiGridBarrierPETSc requires MPI for testing. Tests are automatically run with multiple MPI ranks.

## Running Tests

### Using Pkg.test() (recommended)

```julia
using Pkg
Pkg.test("MultiGridBarrierPETSc")
```

Or from the command line:
```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

This automatically spawns 4 MPI processes to run the tests.

### Running MPI tests directly

You can also run individual test files directly using Julia's MPI launcher:

```bash
julia --project=. -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project=. test/test_helpers.jl`)'
julia --project=. -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project=. test/test_geometry_conversion.jl`)'
julia --project=. -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project=. test/test_fem2d_petsc.jl`)'
julia --project=. -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project=. test/test_quick.jl`)'
```

This uses `MPI.mpiexec()` to get the correct MPI launcher configured for your Julia installation.

## Test Suite Overview

### `test_helpers.jl`
Tests the MultiGridBarrier API function extensions for SafePETSc types:
- `amgb_zeros`: Creating zero matrices with appropriate PETSc storage
- `amgb_all_isfinite`: Validation of finite values in PETSc Vec
- `amgb_hcat`: Horizontal concatenation preserving sparsity type
- `amgb_diag`: Diagonal matrix creation from Vec/Vector
- `amgb_blockdiag`: Block diagonal matrix construction
- `map_rows`: Row-wise operations on PETSc matrices
- `Base.minimum/maximum`: Overloads for PETSc Vec

### `test_geometry_conversion.jl`
Tests the native → PETSc Geometry conversion pipeline:
- Type correctness (MPIDENSE for x/w, MPIAIJ for operators)
- Dimension preservation across conversion
- Value preservation (comparing Vector/Matrix materializations)
- Operator existence and properties (:id, :dx, :dy)
- Subspace structure preservation (:dirichlet, :full, :uniform)
- Multi-level hierarchy (refine/coarsen) preservation

### `test_fem2d_petsc.jl`
Tests the `fem2d_petsc()` public API:
- Returns valid Geometry with PETSc types
- Correct element type support (Float64, Float32)
- Different multigrid levels (L=1, L=2)
- PETSc type consistency

### `test_quick.jl`
Tests end-to-end solving with `fem2d_petsc_solve()`:
- Returns valid AMGBSOL structure
- Solution dimension consistency
- Different problem parameters (p, L)
- Comparison with native MultiGridBarrier solutions

## Test Infrastructure

Uses `mpi_test_harness.jl` for `QuietTestSet` - a custom test set that:
- Suppresses individual test output for clean terminal
- Aggregates test counts across all MPI ranks
- Enables per-rank test execution with global result reporting

All tests run with **4 MPI ranks** for thorough distributed testing.

## Requirements

- MPI must be installed and configured
- The package must be set up to use system MPI
- MultiGridBarrier.jl v0.11.25 or compatible must be available
