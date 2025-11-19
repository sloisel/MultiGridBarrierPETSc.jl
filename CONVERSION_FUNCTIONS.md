# PETSc-to-Native Conversion Functions

## Overview

This document describes the conversion functions added to `MultiGridBarrierPETSc.jl` for converting between PETSc distributed types and native Julia types.

## New Functions

### 1. `geometry_native_to_petsc` (enhanced with full typing)

**Signature:**
```julia
geometry_native_to_petsc(
    g_native::Geometry{T, Matrix{T}, Vector{T}, SparseMatrixCSC{T,Int}, Discretization}
) where {T, Discretization}
```

**Description:**
Converts a native Geometry object (with Julia arrays) to use PETSc distributed types. Now fully typed with explicit type parameters.

**Conversions:**
- `x::Matrix{T}` → `Mat{T, MPIDENSE}`
- `w::Vector{T}` → `Vec{T, MPIDENSE}`
- `operators[key]::SparseMatrixCSC{T,Int}` → `Mat{T, MPIAIJ}`
- `subspaces[key][i]::SparseMatrixCSC{T,Int}` → `Mat{T, MPIAIJ}`
- `refine[i]::SparseMatrixCSC{T,Int}` → `Mat{T, MPIAIJ}`
- `coarsen[i]::SparseMatrixCSC{T,Int}` → `Mat{T, MPIAIJ}`

### 2. `geometry_petsc_to_native` (new)

**Signature:**
```julia
geometry_petsc_to_native(
    g_petsc::Geometry{T, Mat{T,XPrefix}, Vec{T,WPrefix}, Mat{T,MPrefix}, Discretization}
) where {T, XPrefix, WPrefix, MPrefix, Discretization}
```

**Description:**
Converts a PETSc Geometry object back to native Julia arrays. This is the inverse of `geometry_native_to_petsc`.

**Conversions:**
- `Mat{T, MPIDENSE}` → `Matrix{T}`
- `Vec{T, MPIDENSE}` → `Vector{T}`
- `Mat{T, MPIAIJ}` → `SparseMatrixCSC{T,Int}`

**Implementation:**
Uses `SafePETSc.J()` which automatically handles dense vs sparse conversion based on the Mat's storage type.

### 3. `sol_petsc_to_native` (new)

**Signature:**
```julia
sol_petsc_to_native(
    sol_petsc::AMGBSOL{T, XType, WType, MType, Discretization}
) where {T, XType, WType, MType, Discretization}
```

**Description:**
Converts an AMGBSOL solution object from PETSc types back to native Julia types. Performs a deep conversion of the entire solution structure.

**Conversions:**
- `z`: `Mat{T,Prefix}` → `Matrix{T}` or `Vec{T,Prefix}` → `Vector{T}`
- `SOL_feasibility`: NamedTuple with PETSc types → NamedTuple with native types
- `SOL_main`: NamedTuple with PETSc types → NamedTuple with native types
- `geometry`: Geometry with PETSc types → Geometry with native types
- `log`: String (unchanged)

**Implementation:**
Recursively converts all PETSc types within the solution structure using `SafePETSc.J()`:
- Helper function `convert_namedtuple()` handles NamedTuples
- Helper function `convert_value()` handles individual values, arrays, and PETSc types
- Preserves the structure while converting all PETSc types to native equivalents

## Usage Example

```julia
using SafePETSc
using MultiGridBarrierPETSc

# Initialize
SafePETSc.Init()

# Create and solve with PETSc types
g_petsc = fem2d_petsc(Float64; maxh=0.1)
sol_petsc = amgb(g_petsc; p=2.0, verbose=true)

# Convert solution back to native Julia types for analysis
sol_native = sol_petsc_to_native(sol_petsc)

# Now you can use native Julia operations on sol_native
# sol_native.z is Matrix{Float64} or Vector{Float64}
# sol_native.geometry.operators[:grad] is SparseMatrixCSC{Float64, Int}
# etc.

# You can also convert geometries independently
g_native = geometry_petsc_to_native(g_petsc)

# And convert back
g_petsc_again = geometry_native_to_petsc(g_native)
```

## Key Design Decisions

1. **Use of SafePETSc.J()**: This function automatically handles the conversion based on the Mat's storage type, returning either `Matrix{T}` for dense matrices or `SparseMatrixCSC{T,Int}` for sparse matrices.

2. **Full Type Annotations**: All functions are fully typed to ensure type stability and catch errors at compile time.

3. **Collective Operations**: All conversion functions are MPI collective operations - all ranks must call them together.

4. **Preserves Order**: Uses `sort(collect(keys(...)))` to ensure deterministic iteration order across all MPI ranks.

5. **Deep Conversion**: The `sol_petsc_to_native` function performs a deep conversion, recursively handling nested structures like NamedTuples and arrays.

## Testing

A comprehensive test suite is provided in `test/test_conversions.jl` that verifies:
- Round-trip conversion of Geometry objects (native → PETSc → native)
- AMGBSOL conversion with a real solve
- Type correctness of converted objects
- Value preservation through conversions

Run tests with:
```bash
julia --project=. -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project=$(Base.active_project()) test/test_conversions.jl`)'
```

## Exports

The following functions are now exported by the module:
- `fem2d_petsc`
- `fem2d_petsc_solve`
- `geometry_native_to_petsc`
- `geometry_petsc_to_native`
- `sol_petsc_to_native`
