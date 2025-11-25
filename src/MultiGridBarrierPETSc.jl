"""
    MultiGridBarrierPETSc

A module that provides a convenient interface for using MultiGridBarrier with PETSc
distributed types through SafePETSc.

# Exports
- `Init`: Initialize MultiGridBarrierPETSc with MPI, PETSc, and solver options
- `fem1d_petsc`: Creates a PETSc-based Geometry from fem1d parameters
- `fem1d_petsc_solve`: Solves a fem1d problem using amgb with PETSc types
- `fem2d_petsc`: Creates a PETSc-based Geometry from fem2d parameters
- `fem2d_petsc_solve`: Solves a fem2d problem using amgb with PETSc types
- `fem3d_petsc`: Creates a PETSc-based Geometry from fem3d parameters
- `fem3d_petsc_solve`: Solves a fem3d problem using amgb with PETSc types
- `native_to_petsc`: Converts native Geometry to PETSc distributed types
- `petsc_to_native`: Converts PETSc Geometry or AMGBSOL back to native Julia types

# Usage
```julia
using MultiGridBarrierPETSc
MultiGridBarrierPETSc.Init()  # Initialize MPI and PETSc

# 1D: Create PETSc geometry and solve
g1d = fem1d_petsc(Float64; L=4)
sol1d = fem1d_petsc_solve(Float64; L=4, p=1.0, verbose=true)

# 2D: Create PETSc geometry and solve
g = fem2d_petsc(Float64; maxh=0.1)
sol = fem2d_petsc_solve(Float64; maxh=0.1, p=2.0, verbose=true)

# 3D: Create PETSc geometry and solve
g3d = fem3d_petsc(Float64; L=2, k=3)
sol3d = fem3d_petsc_solve(Float64; L=2, k=3, p=1.0, verbose=true)

# Convert solution back to native types for analysis
sol_native = petsc_to_native(sol)
```
"""
module MultiGridBarrierPETSc

using MPI
using SafePETSc
using SafePETSc: MPIDENSE, MPIAIJ
using LinearAlgebra
using SparseArrays
using MultiGridBarrier
using MultiGridBarrier: Geometry, AMGBSOL, fem1d, FEM1D
using MultiGridBarrier3d
using MultiGridBarrier3d: fem3d, FEM3D

# ============================================================================
# MultiGridBarrier API Implementation for SafePETSc Types
# ============================================================================

# Import the functions we need to extend
import MultiGridBarrier: amgb_zeros, amgb_all_isfinite, amgb_hcat, amgb_diag, amgb_blockdiag, map_rows

# amgb_zeros: Create zero matrices with appropriate type
MultiGridBarrier.amgb_zeros(::Mat{T, MPIAIJ}, m, n) where {T} = Mat_uniform(spzeros(T, m, n); Prefix=MPIAIJ)
MultiGridBarrier.amgb_zeros(::LinearAlgebra.Adjoint{T, <:Mat{T, MPIAIJ}}, m, n) where {T} = Mat_uniform(spzeros(T, m, n); Prefix=MPIAIJ)
MultiGridBarrier.amgb_zeros(::Mat{T, MPIDENSE}, m, n) where {T} = Mat_uniform(zeros(T, m, n); Prefix=MPIDENSE)
MultiGridBarrier.amgb_zeros(::LinearAlgebra.Adjoint{T, <:Mat{T, MPIDENSE}}, m, n) where {T} = Mat_uniform(zeros(T, m, n); Prefix=MPIDENSE)

# amgb_all_isfinite: Check if all elements are finite
MultiGridBarrier.amgb_all_isfinite(z::Vec{T}) where {T} = all(isfinite.(Vector(z)))

# amgb_hcat: Horizontal concatenation
# Just use the built-in hcat - it handles partitions correctly
MultiGridBarrier.amgb_hcat(A::Mat...) = hcat(A...)

# amgb_diag: Create diagonal matrix from vector
MultiGridBarrier.amgb_diag(::Mat{T, MPIAIJ}, z::Vec{T}, m=length(z), n=length(z)) where {T} =
    spdiagm(m, n, 0 => z; prefix=MPIAIJ)
MultiGridBarrier.amgb_diag(::Mat{T, MPIAIJ}, z::Vector{T}, m=length(z), n=length(z)) where {T} =
    Mat_uniform(spdiagm(m, n, 0 => z); Prefix=MPIAIJ)
MultiGridBarrier.amgb_diag(::Mat{T, MPIDENSE}, z::Vec{T}, m=length(z), n=length(z)) where {T} =
    spdiagm(m, n, 0 => z; prefix=MPIDENSE)
MultiGridBarrier.amgb_diag(::Mat{T, MPIDENSE}, z::Vector{T}, m=length(z), n=length(z)) where {T} =
    Mat_uniform(spdiagm(m, n, 0 => z); Prefix=MPIDENSE)

# amgb_blockdiag: Block diagonal concatenation
MultiGridBarrier.amgb_blockdiag(args::Mat{T, MPIAIJ}...) where {T} = blockdiag(args...)
MultiGridBarrier.amgb_blockdiag(args::Mat{T, MPIDENSE}...) where {T} = blockdiag(args...)

# map_rows: Apply function to each row
function MultiGridBarrier.map_rows(f, A::Union{Vec{T}, Mat{T}, LinearAlgebra.Adjoint{T, <:Mat{T}}}...) where {T}
    # Materialize any adjoint matrices using Mat(A') constructor
    materialized = [a isa LinearAlgebra.Adjoint ? Mat(a) : a for a in A]
    return SafePETSc.map_rows(f, materialized...)
end

# Additional base functions needed for MultiGridBarrier
# These must compute GLOBAL min/max across all ranks using MPI reductions
function Base.minimum(v::Vec{T}) where {T}
    local_min = minimum(SafePETSc.PETSc.unsafe_localarray(v.obj.v; read=true))
    return MPI.Allreduce(local_min, MPI.MIN, MPI.COMM_WORLD)
end

function Base.maximum(v::Vec{T}) where {T}
    local_max = maximum(SafePETSc.PETSc.unsafe_localarray(v.obj.v; read=true))
    return MPI.Allreduce(local_max, MPI.MAX, MPI.COMM_WORLD)
end

# ============================================================================
# Type Conversion
# ============================================================================

"""
    native_to_petsc(g_native::Geometry{T, Matrix{T}, Vector{T}, SparseMatrixCSC{T,Int}, Discretization}) where {T, Discretization}

**Collective**

Convert a native Geometry object (with Julia arrays) to use PETSc distributed types.

This is a collective operation. Each rank calls fem2d() to get the same native
geometry, then this function converts:
- x::Matrix{T} -> x::Mat{T, MPIDENSE}
- w::Vector{T} -> w::Vec{T, MPIDENSE}
- operators[key]::SparseMatrixCSC{T,Int} -> operators[key]::Mat{T, MPIAIJ}
- subspaces[key][i]::SparseMatrixCSC{T,Int} -> subspaces[key][i]::Mat{T, MPIAIJ}

The MPIDENSE prefix indicates dense storage (for geometry data and weights),
while MPIAIJ indicates sparse storage (for operators and subspace matrices).
"""
function native_to_petsc(g_native::Geometry{T, Matrix{T}, Vector{T}, SparseMatrixCSC{T,Int}, Discretization}) where {T, Discretization}
    # Convert x (geometry coordinates) to MPIDENSE Mat
    x_petsc = SafePETSc.Mat_uniform(g_native.x; Prefix=MPIDENSE)

    # Convert w (weights) to MPIDENSE Vec (weights are uniform/dense data)
    w_petsc = SafePETSc.Vec_uniform(g_native.w; Prefix=MPIDENSE)

    # Convert all operators to MPIAIJ Mat
    # Mat_uniform distributes the uniform matrix across ranks as MPIAIJ (sparse, partitioned)
    # Sort keys to ensure deterministic order across all ranks
    operators_petsc = Dict{Symbol, Any}()
    for key in sort(collect(keys(g_native.operators)))
        op = g_native.operators[key]
        operators_petsc[key] = SafePETSc.Mat_uniform(op; Prefix=MPIAIJ)
    end

    # Convert all subspace matrices to MPIAIJ Mat
    # Sort keys and use explicit loops to ensure all ranks iterate in sync
    subspaces_petsc = Dict{Symbol, Vector{Any}}()
    for key in sort(collect(keys(g_native.subspaces)))
        subspace_vec = g_native.subspaces[key]
        petsc_vec = Vector{Any}(undef, length(subspace_vec))
        for i in 1:length(subspace_vec)
            petsc_vec[i] = SafePETSc.Mat_uniform(subspace_vec[i]; Prefix=MPIAIJ)
        end
        subspaces_petsc[key] = petsc_vec
    end

    # Convert refine and coarsen vectors to MPIAIJ Mat
    refine_petsc = Vector{Any}(undef, length(g_native.refine))
    for i in 1:length(g_native.refine)
        refine_petsc[i] = SafePETSc.Mat_uniform(g_native.refine[i]; Prefix=MPIAIJ)
    end

    coarsen_petsc = Vector{Any}(undef, length(g_native.coarsen))
    for i in 1:length(g_native.coarsen)
        coarsen_petsc[i] = SafePETSc.Mat_uniform(g_native.coarsen[i]; Prefix=MPIAIJ)
    end

    # Determine PETSc types for Geometry type parameters
    XType = typeof(x_petsc)
    WType = typeof(w_petsc)
    MType = typeof(operators_petsc[:id])  # Use id operator as representative
    DType = typeof(g_native.discretization)

    # Create typed dicts and vectors for Geometry constructor
    operators_typed = Dict{Symbol, MType}()
    for key in keys(operators_petsc)
        operators_typed[key] = operators_petsc[key]
    end

    subspaces_typed = Dict{Symbol, Vector{MType}}()
    for key in keys(subspaces_petsc)
        subspaces_typed[key] = convert(Vector{MType}, subspaces_petsc[key])
    end

    refine_typed = convert(Vector{MType}, refine_petsc)
    coarsen_typed = convert(Vector{MType}, coarsen_petsc)

    # Create new Geometry with PETSc types using explicit type parameters
    return Geometry{Float64, XType, WType, MType, DType}(
        g_native.discretization,
        x_petsc,
        w_petsc,
        subspaces_typed,
        operators_typed,
        refine_typed,
        coarsen_typed
    )
end

"""
    petsc_to_native(g_petsc::Geometry{T, Mat{T,XPrefix}, Vec{T,WPrefix}, Mat{T,MPrefix}, Discretization}) where {T, XPrefix, WPrefix, MPrefix, Discretization}

**Collective**

Convert a PETSc Geometry object (with distributed PETSc types) back to native Julia arrays.

This is a collective operation. This function converts:
- x::Mat{T, MPIDENSE} -> x::Matrix{T}
- w::Vec{T, MPIDENSE} -> w::Vector{T}
- operators[key]::Mat{T, MPIAIJ} -> operators[key]::SparseMatrixCSC{T,Int}
- subspaces[key][i]::Mat{T, MPIAIJ} -> subspaces[key][i]::SparseMatrixCSC{T,Int}

Uses SafePETSc.J() which automatically handles dense vs sparse conversion based on the Mat's storage type.
"""
function petsc_to_native(g_petsc::Geometry{T, Mat{T,XPrefix}, Vec{T,WPrefix}, Mat{T,MPrefix}, Discretization}) where {T, XPrefix, WPrefix, MPrefix, Discretization}
    # Convert x (geometry coordinates) from MPIDENSE Mat to Matrix
    x_native = SafePETSc.J(g_petsc.x)

    # Convert w (weights) from MPIDENSE Vec to Vector
    w_native = SafePETSc.J(g_petsc.w)

    # Convert all operators from MPIAIJ Mat to SparseMatrixCSC
    # Sort keys to ensure deterministic order across all ranks
    operators_native = Dict{Symbol, SparseMatrixCSC{T,Int}}()
    for key in sort(collect(keys(g_petsc.operators)))
        op = g_petsc.operators[key]
        operators_native[key] = SafePETSc.J(op)
    end

    # Convert all subspace matrices from MPIAIJ Mat to SparseMatrixCSC
    # Sort keys and use explicit loops to ensure all ranks iterate in sync
    subspaces_native = Dict{Symbol, Vector{SparseMatrixCSC{T,Int}}}()
    for key in sort(collect(keys(g_petsc.subspaces)))
        subspace_vec = g_petsc.subspaces[key]
        native_vec = Vector{SparseMatrixCSC{T,Int}}(undef, length(subspace_vec))
        for i in 1:length(subspace_vec)
            native_vec[i] = SafePETSc.J(subspace_vec[i])
        end
        subspaces_native[key] = native_vec
    end

    # Convert refine and coarsen vectors from MPIAIJ Mat to SparseMatrixCSC
    refine_native = Vector{SparseMatrixCSC{T,Int}}(undef, length(g_petsc.refine))
    for i in 1:length(g_petsc.refine)
        refine_native[i] = SafePETSc.J(g_petsc.refine[i])
    end

    coarsen_native = Vector{SparseMatrixCSC{T,Int}}(undef, length(g_petsc.coarsen))
    for i in 1:length(g_petsc.coarsen)
        coarsen_native[i] = SafePETSc.J(g_petsc.coarsen[i])
    end

    # Create new Geometry with native Julia types using explicit type parameters
    return Geometry{T, Matrix{T}, Vector{T}, SparseMatrixCSC{T,Int}, Discretization}(
        g_petsc.discretization,
        x_native,
        w_native,
        subspaces_native,
        operators_native,
        refine_native,
        coarsen_native
    )
end

"""
    petsc_to_native(sol_petsc::AMGBSOL{T, XType, WType, MType, Discretization}) where {T, XType, WType, MType, Discretization}

**Collective**

Convert an AMGBSOL solution object from PETSc types back to native Julia types.

This is a collective operation that performs a deep conversion of the solution structure:
- z: Mat{T,Prefix} -> Matrix{T} or Vec{T,Prefix} -> Vector{T} (depending on the type)
- SOL_feasibility: NamedTuple with PETSc types -> NamedTuple with native types
- SOL_main: NamedTuple with PETSc types -> NamedTuple with native types
- geometry: Geometry with PETSc types -> Geometry with native types

Uses SafePETSc.J() which automatically handles dense vs sparse conversion based on the Mat's storage type.
"""
function petsc_to_native(sol_petsc::AMGBSOL{T, XType, WType, MType, Discretization}) where {T, XType, WType, MType, Discretization}
    # Convert z using J - handles both Mat and Vec types
    z_native = SafePETSc.J(sol_petsc.z)

    # Helper function to recursively convert NamedTuples with PETSc types
    function convert_namedtuple(nt)
        if nt === nothing
            return nothing
        end
        # Convert each field in the NamedTuple
        converted_fields = []
        for (name, value) in pairs(nt)
            converted_value = convert_value(value)
            push!(converted_fields, name => converted_value)
        end
        return NamedTuple(converted_fields)
    end

    # Helper function to convert individual values
    function convert_value(value)
        if isa(value, Mat) || isa(value, Vec)
            return SafePETSc.J(value)
        elseif isa(value, Array)
            # Recursively convert arrays
            return map(convert_value, value)
        else
            # For non-PETSc types (numbers, strings, etc.), return as-is
            return value
        end
    end

    # Convert SOL_feasibility and SOL_main NamedTuples
    SOL_feasibility_native = convert_namedtuple(sol_petsc.SOL_feasibility)
    SOL_main_native = convert_namedtuple(sol_petsc.SOL_main)

    # Convert the geometry
    geometry_native = petsc_to_native(sol_petsc.geometry)

    # Determine native types
    ZType = typeof(z_native)

    # Create and return the native AMGBSOL
    return AMGBSOL{T, ZType, Vector{T}, SparseMatrixCSC{T,Int}, Discretization}(
        z_native,
        SOL_feasibility_native,
        SOL_main_native,
        sol_petsc.log,
        geometry_native
    )
end

# ============================================================================
# Public API
# ============================================================================

# Module-level flag to track whether MultiGridBarrierPETSc has been initialized
const MGBPETSC_INITIALIZED = Ref(false)

"""
    Init(; options="-MPIAIJ_ksp_type preonly -MPIAIJ_pc_type lu -MPIAIJ_pc_factor_mat_solver_type mumps")

**Collective**

Initialize MultiGridBarrierPETSc by setting up MPI, PETSc, and solver options.

This function should be called once before using any MultiGridBarrierPETSc functionality.
It will:
1. Initialize MPI and PETSc if not already initialized
2. Configure PETSc solver options (default: use MUMPS direct solver for sparse matrices)

The default options configure MPIAIJ (sparse) matrices to use:
- `-ksp_type preonly`: Don't use iterative solver, just apply preconditioner
- `-pc_type lu`: Use LU factorization as preconditioner
- `-pc_factor_mat_solver_type mumps`: Use MUMPS sparse direct solver for the factorization

Dense matrices (MPIDENSE) use PETSc's default dense LU solver.

# Arguments
- `options::String`: PETSc options string to set (default: MUMPS direct solver for sparse matrices)

# Example
```julia
using MultiGridBarrierPETSc
MultiGridBarrierPETSc.Init()  # Use default MUMPS solver for sparse matrices

# Or with custom options:
MultiGridBarrierPETSc.Init(options="-MPIAIJ_ksp_type cg -MPIAIJ_pc_type jacobi")
```
"""
function Init(; options="-MPIAIJ_ksp_type preonly -MPIAIJ_pc_type lu -MPIAIJ_pc_factor_mat_solver_type mumps")
    # Only initialize once
    if MGBPETSC_INITIALIZED[]
        return
    end

    # Initialize MPI and PETSc if not already initialized
    if !SafePETSc.Initialized()
        SafePETSc.Init()
    end

    # Get rank for output
    rank = MPI.Comm_rank(MPI.COMM_WORLD)

    # Set PETSc options
    if rank == 0
        println("Initializing MultiGridBarrierPETSc with solver options...")
    end
    SafePETSc.petsc_options_insert_string(options)

    MGBPETSC_INITIALIZED[] = true
end

"""
    fem1d_petsc(::Type{T}=Float64; kwargs...) where {T}

**Collective**

Create a PETSc-based Geometry from fem1d parameters.

This function calls `fem1d(kwargs...)` to create a native 1D geometry, then converts
it to use PETSc distributed types (Mat and Vec) for distributed computing.

Note: Call `MultiGridBarrierPETSc.Init()` before using this function.

# Arguments
- `T::Type`: Element type for the geometry (default: Float64)
- `kwargs...`: Additional keyword arguments passed to `fem1d()`:
  - `L::Int`: Number of multigrid levels (default: 4), creating 2^L elements

# Returns
A Geometry object with PETSc distributed types.

# Example
```julia
MultiGridBarrierPETSc.Init()
g = fem1d_petsc(Float64; L=4)
```
"""
function fem1d_petsc(::Type{T}=Float64; kwargs...) where {T}
    # Create native 1D geometry
    g_native = fem1d(T; kwargs...)

    # Convert to PETSc types
    return native_to_petsc(g_native)
end

"""
    fem1d_petsc_solve(::Type{T}=Float64; kwargs...) where {T}

**Collective**

Solve a fem1d problem using amgb with PETSc distributed types.

This is a convenience function that combines `fem1d_petsc` and `amgb` into a
single call. It creates a PETSc-based 1D geometry and solves the barrier problem.

# Arguments
- `T::Type`: Element type for the geometry (default: Float64)
- `kwargs...`: Keyword arguments passed to both `fem1d_petsc` and `amgb`
  - `L::Int`: Number of multigrid levels (passed to fem1d)
  - `p`: Power parameter for the barrier (passed to amgb)
  - `verbose`: Verbosity flag (passed to amgb)
  - Other arguments specific to fem1d or amgb

# Returns
The solution object from `amgb`.

# Example
```julia
sol = fem1d_petsc_solve(Float64; L=4, p=1.0, verbose=true)
println("Solution norm: ", norm(sol.z))
```
"""
function fem1d_petsc_solve(::Type{T}=Float64; kwargs...) where {T}
    # Create PETSc 1D geometry
    g = fem1d_petsc(T; kwargs...)

    # Solve using amgb (amgb auto-detects 1D from geometry.discretization)
    return amgb(g; kwargs...)
end

"""
    fem2d_petsc(::Type{T}=Float64; kwargs...) where {T}

**Collective**

Create a PETSc-based Geometry from fem2d parameters.

This function calls `fem2d(kwargs...)` to create a native geometry, then converts
it to use PETSc distributed types (Mat and Vec) for distributed computing.

Note: Call `MultiGridBarrierPETSc.Init()` before using this function.

# Arguments
- `T::Type`: Element type for the geometry (default: Float64)
- `kwargs...`: Additional keyword arguments passed to `fem2d()`

# Returns
A Geometry object with PETSc distributed types.

# Example
```julia
MultiGridBarrierPETSc.Init()
g = fem2d_petsc(Float64; maxh=0.1)
```
"""
function fem2d_petsc(::Type{T}=Float64; kwargs...) where {T}
    # Create native geometry
    g_native = fem2d(; kwargs...)

    # Convert to PETSc types
    return native_to_petsc(g_native)
end

"""
    fem2d_petsc_solve(::Type{T}=Float64; kwargs...) where {T}

**Collective**

Solve a fem2d problem using amgb with PETSc distributed types.

This is a convenience function that combines `fem2d_petsc` and `amgb` into a
single call. It creates a PETSc-based geometry and solves the barrier problem.

# Arguments
- `T::Type`: Element type for the geometry (default: Float64)
- `kwargs...`: Keyword arguments passed to both `fem2d_petsc` and `amgb`
  - `maxh`: Maximum mesh size (passed to fem2d)
  - `p`: Power parameter for the barrier (passed to amgb)
  - `verbose`: Verbosity flag (passed to amgb)
  - Other arguments specific to fem2d or amgb

# Returns
The solution object from `amgb`.

# Example
```julia
sol = fem2d_petsc_solve(Float64; maxh=0.1, p=2.0, verbose=true)
println("Solution norm: ", norm(sol.z))
```
"""
function fem2d_petsc_solve(::Type{T}=Float64; kwargs...) where {T}
    # Create PETSc geometry
    g = fem2d_petsc(T; kwargs...)

    # Solve using amgb
    return amgb(g; kwargs...)
end

"""
    fem3d_petsc(::Type{T}=Float64; kwargs...) where {T}

**Collective**

Create a PETSc-based Geometry from fem3d parameters.

This function calls `fem3d(kwargs...)` to create a native 3D geometry, then converts
it to use PETSc distributed types (Mat and Vec) for distributed computing.

Note: Call `MultiGridBarrierPETSc.Init()` before using this function.

# Arguments
- `T::Type`: Element type for the geometry (default: Float64)
- `kwargs...`: Additional keyword arguments passed to `fem3d()`:
  - `L::Int`: Number of multigrid levels (default: 2)
  - `k::Int`: Polynomial order of elements (default: 3)
  - `K`: Coarse Q1 mesh as an N×3 matrix (optional, defaults to unit cube)

# Returns
A Geometry object with PETSc distributed types.

# Example
```julia
MultiGridBarrierPETSc.Init()
g = fem3d_petsc(Float64; L=2, k=3)
```
"""
function fem3d_petsc(::Type{T}=Float64; kwargs...) where {T}
    # Create native 3D geometry
    g_native = fem3d(T; kwargs...)

    # Convert to PETSc types
    return native_to_petsc(g_native)
end

"""
    fem3d_petsc_solve(::Type{T}=Float64; kwargs...) where {T}

**Collective**

Solve a fem3d problem using amgb with PETSc distributed types.

This is a convenience function that combines `fem3d_petsc` and `amgb` into a
single call. It creates a PETSc-based 3D geometry and solves the barrier problem.

# Arguments
- `T::Type`: Element type for the geometry (default: Float64)
- `kwargs...`: Keyword arguments passed to both `fem3d_petsc` and `amgb`
  - `L::Int`: Number of multigrid levels (passed to fem3d)
  - `k::Int`: Polynomial order of elements (passed to fem3d)
  - `p`: Power parameter for the barrier (passed to amgb)
  - `verbose`: Verbosity flag (passed to amgb)
  - `D`: Operator structure matrix (passed to amgb, defaults to 3D operators)
  - `f`: Source term function (passed to amgb, defaults to 3D source)
  - `g`: Boundary condition function (passed to amgb, defaults to 3D BCs)
  - Other arguments specific to fem3d or amgb

# Returns
The solution object from `amgb`.

# Example
```julia
sol = fem3d_petsc_solve(Float64; L=2, k=3, p=1.0, verbose=true)
println("Solution norm: ", norm(sol.z))
```
"""
function fem3d_petsc_solve(::Type{T}=Float64;
    D = [:u :id; :u :dx; :u :dy; :u :dz; :s :id],
    f = (x) -> T[0.5, 0.0, 0.0, 0.0, 1.0],
    g = (x) -> T[x[1]^2 + x[2]^2 + x[3]^2, 100.0],
    kwargs...) where {T}
    # Create PETSc 3D geometry
    geom = fem3d_petsc(T; kwargs...)

    # Solve using amgb with 3D-specific defaults
    return amgb(geom; D=D, f=f, g=g, kwargs...)
end

# Export the public API
export Init
export fem1d_petsc, fem1d_petsc_solve
export fem2d_petsc, fem2d_petsc_solve
export fem3d_petsc, fem3d_petsc_solve
export native_to_petsc, petsc_to_native

end # module MultiGridBarrierPETSc
