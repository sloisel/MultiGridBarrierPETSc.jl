# API Reference

This page provides detailed documentation for all exported functions in MultiGridBarrierPETSc.jl.

!!! note "All Functions Are Collective"
    All functions documented here are **MPI collective operations**. Every MPI rank must call these functions together with the same parameters. Failure to do so will result in deadlock.

## Initialization

This function must be called before using any other MultiGridBarrierPETSc functionality.

```@docs
Init
```

## High-Level API

These functions provide the simplest interface for solving problems with PETSc types.

### 1D Problems

```@docs
fem1d_petsc
fem1d_petsc_solve
```

### 2D Problems

```@docs
fem2d_petsc
fem2d_petsc_solve
```

### 3D Problems

```@docs
fem3d_petsc
fem3d_petsc_solve
```

## Type Conversion API

These functions convert between native Julia types and PETSc distributed types.
The `petsc_to_native` function dispatches on type, handling both `Geometry` and `AMGBSOL` objects.

```@docs
native_to_petsc
petsc_to_native
```

## Type Mappings Reference

### Native to PETSc Conversions

When converting from native Julia types to PETSc distributed types:

| Native Type | PETSc Type | PETSc Prefix | Usage |
|-------------|------------|--------------|-------|
| `Matrix{T}` | `Mat{T, MPIDENSE}` | MPIDENSE | Geometry coordinates, dense operators |
| `Vector{T}` | `Vec{T, MPIDENSE}` | MPIDENSE | Weights, dense vectors |
| `SparseMatrixCSC{T,Int}` | `Mat{T, MPIAIJ}` | MPIAIJ | Sparse operators, subspace matrices |

### PETSc to Native Conversions

When converting from PETSc distributed types back to native Julia types:

| PETSc Type | Native Type | Conversion Method |
|------------|-------------|-------------------|
| `Mat{T, MPIDENSE}` | `Matrix{T}` | `SafePETSc.J()` |
| `Mat{T, MPIAIJ}` | `SparseMatrixCSC{T,Int}` | `SafePETSc.J()` |
| `Vec{T, MPIDENSE}` | `Vector{T}` | `SafePETSc.J()` |

## Geometry Structure

The `Geometry` type from MultiGridBarrier is parameterized by its storage types:

**Native Geometry:**
```julia
Geometry{T, Matrix{T}, Vector{T}, SparseMatrixCSC{T,Int}, Discretization}
```

**PETSc Geometry:**
```julia
Geometry{T, Mat{T,MPIDENSE}, Vec{T,MPIDENSE}, Mat{T,MPIAIJ}, Discretization}
```

### Fields

- **`discretization`**: Discretization information (domain, mesh, etc.)
- **`x`**: Geometry coordinates (Matrix or Mat)
- **`w`**: Quadrature weights (Vector or Vec)
- **`operators`**: Dictionary of operators (id, laplacian, mass, etc.)
- **`subspaces`**: Dictionary of subspace projection matrices
- **`refine`**: Vector of refinement matrices (coarse → fine)
- **`coarsen`**: Vector of coarsening matrices (fine → coarse)

## Solution Structure

The `AMGBSOL` type from MultiGridBarrier contains the complete solution:

### Fields

- **`z`**: Solution matrix/vector
- **`SOL_feasibility`**: NamedTuple with feasibility phase information
- **`SOL_main`**: NamedTuple with main solve information
  - `objective`: Final objective function value
  - `primal_residual`: Primal feasibility residual
  - `dual_residual`: Dual feasibility residual
- **`log`**: Vector of iteration logs
- **`geometry`**: The geometry used for solving

## MPI and IO Utilities

### SafePETSc.io0()

Returns an IO stream that only writes on rank 0:

```julia
println(io0(), "This prints once from rank 0")
println(io0(), my_petsc_vec)  # Collective show() of Vec
```

### MPI Rank Information

```julia
using MPI

rank = MPI.Comm_rank(MPI.COMM_WORLD)  # Current rank (0 to nranks-1)
nranks = MPI.Comm_size(MPI.COMM_WORLD)  # Total number of ranks
```

## PETSc Configuration

### MUMPS Solver

The `Init()` function automatically configures PETSc to use MUMPS for sparse matrices:

```julia
# Equivalent PETSc options set automatically:
# -MPIAIJ_ksp_type preonly          # No iterative solver, just direct solve
# -MPIAIJ_pc_type lu                # LU factorization
# -MPIAIJ_pc_factor_mat_solver_type mumps  # Use MUMPS for factorization
```

**Matrix Type Configuration:**
- **Sparse matrices (MPIAIJ)**: Use MUMPS direct solver for exact solves
- **Dense matrices (MPIDENSE)**: Use PETSc's default dense LU solver

This ensures exact direct solves for linear systems in the barrier method's Newton iterations.

## Examples

### Type Conversion Round-Trip

```julia
using MultiGridBarrierPETSc
using MultiGridBarrier
using LinearAlgebra
MultiGridBarrierPETSc.Init()

# Create native geometry
g_native = fem2d(; maxh=0.3)

# Convert to PETSc
g_petsc = native_to_petsc(g_native)

# Solve with PETSc types
sol_petsc = amgb(g_petsc; p=2.0)

# Convert back to native
sol_native = petsc_to_native(sol_petsc)
g_back = petsc_to_native(g_petsc)

# Verify round-trip accuracy
@assert norm(g_native.x - g_back.x) < 1e-10
@assert norm(g_native.w - g_back.w) < 1e-10
```

### Accessing Operator Matrices

```julia
# Native geometry
g_native = fem2d(; maxh=0.2)
lap_native = g_native.operators[:laplacian]  # SparseMatrixCSC

# PETSc geometry
g_petsc = native_to_petsc(g_native)
lap_petsc = g_petsc.operators[:laplacian]  # Mat{Float64, MPIAIJ}

# Convert back if needed
lap_back = SafePETSc.J(lap_petsc)  # SparseMatrixCSC
```

## Integration with MultiGridBarrier

All MultiGridBarrier functions work seamlessly with PETSc types:

```julia
using MultiGridBarrier: amgb, amgb_solve

# Create PETSc geometry
g = fem2d_petsc(Float64; L=3)

# Use MultiGridBarrier functions directly
sol = amgb(g; p=1.0, verbose=true)
sol = amgb_solve(g; p=1.5, maxit=50, tol=1e-10)
```

The package extends MultiGridBarrier's internal API (`amgb_zeros`, `amgb_hcat`, `amgb_diag`, etc.) to work with PETSc types automatically.
