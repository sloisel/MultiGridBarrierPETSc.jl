# User Guide

This guide covers the essential workflows for using MultiGridBarrierPETSc.jl.

## Initialization

Every program using MultiGridBarrierPETSc.jl must initialize the package before calling any functions:

```julia
using MultiGridBarrierPETSc
MultiGridBarrierPETSc.Init()  # Initialize MPI and PETSc before calling functions
```

The `Init()` function:
- Initializes MPI and PETSc if not already initialized
- Configures MUMPS sparse direct solver for accurate linear solves
- Only needs to be called once at the start of your program

## Basic Workflow

The typical workflow consists of three steps:

1. **Solve with PETSc types** (distributed computation)
2. **Convert to native types** (for analysis/plotting)
3. **Visualize or analyze** (using MultiGridBarrier's tools)

### Complete Example with Visualization

Here's a complete example that solves a 2D FEM problem, converts the solution, and plots it:

```julia
using MultiGridBarrierPETSc
using MultiGridBarrier
using PyPlot
MultiGridBarrierPETSc.Init()

# Step 1: Solve with PETSc distributed types (L=3 for fast documentation builds)
sol_petsc = fem2d_petsc_solve(Float64; L=3, p=1.0, verbose=false)

# Step 2: Convert solution to native Julia types
sol_native = petsc_to_native(sol_petsc)

# Step 3: Plot the solution using MultiGridBarrier's plot function
figure(figsize=(10, 8))
plot(sol_native)
title("Multigrid Barrier Solution (L=3)")
tight_layout()
savefig("solution_plot.png")
println(io0(), "Solution plotted!")
```

!!! tip "Running This Example"
    Save this code to a file (e.g., `visualize.jl`) and run with:
    ```bash
    julia -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) visualize.jl`)'
    ```

    This uses `MPI.mpiexec()` to get the correct MPI launcher configured for your Julia installation, avoiding compatibility issues with system `mpiexec`. Add `--project` or other Julia options as needed for your environment.

    This will create `solution_plot.png` showing the computed solution.

## Understanding MPI Collective Operations

!!! warning "All Functions Are Collective"
    All exported functions in MultiGridBarrierPETSc.jl are **MPI collective operations**. This means:
    - All MPI ranks must call the function
    - All ranks must call it with the same parameters
    - Deadlock will occur if only some ranks call a collective function

**Correct usage:**
```julia
# All ranks execute this together
sol = fem2d_petsc_solve(Float64; L=2, p=1.0)
```

**Incorrect usage (causes deadlock):**
```julia
rank = MPI.Comm_rank(MPI.COMM_WORLD)
if rank == 0
    sol = fem2d_petsc_solve(Float64; L=2, p=1.0)  # ✗ Only rank 0 calls - DEADLOCK!
end
```

## Type Conversions

### Native to PETSc

Convert native Julia arrays to PETSc distributed types:

```julia
using MultiGridBarrier

# Create native geometry
g_native = fem2d(; maxh=0.3)

# Convert to PETSc types for distributed computation
g_petsc = native_to_petsc(g_native)
```

**Type mappings:**

| Native Type | PETSc Type | Storage |
|-------------|------------|---------|
| `Matrix{T}` | `Mat{T, MPIDENSE}` | Dense distributed |
| `Vector{T}` | `Vec{T}` | Dense distributed |
| `SparseMatrixCSC{T,Int}` | `Mat{T, MPIAIJ}` | Sparse distributed |

### PETSc to Native

Convert PETSc types back to native Julia arrays:

```julia
# Create and solve with PETSc types
g_petsc = fem2d_petsc(Float64; maxh=0.3)
sol_petsc = amgb(g_petsc; p=2.0)

# Convert back for analysis
g_native = petsc_to_native(g_petsc)
sol_native = petsc_to_native(sol_petsc)

# Now you can use native Julia operations
using LinearAlgebra
z_matrix = sol_native.z
solution_norm = norm(z_matrix)
println(io0(), "Solution norm: ", solution_norm)
```

## Advanced Usage

### Custom Geometry Workflow

For more control, construct geometries manually:

```julia
using MultiGridBarrierPETSc
using MultiGridBarrier
MultiGridBarrierPETSc.Init()

# 1. Create native geometry with specific parameters
g_native = fem2d(; maxh=0.2, L=2)

# 2. Convert to PETSc for distributed solving
g_petsc = native_to_petsc(g_native)

# 3. Solve with custom barrier parameters
sol_petsc = amgb(g_petsc;
    p=1.5,           # Barrier power parameter
    verbose=true,    # Print convergence info
    maxit=100,       # Maximum iterations
    tol=1e-8)        # Convergence tolerance

# 4. Convert solution back
sol_native = petsc_to_native(sol_petsc)

# 5. Access solution components
println(io0(), "Newton steps: ", sum(sol_native.SOL_main.its))
println(io0(), "Elapsed time: ", sol_native.SOL_main.t_elapsed, " seconds")
```

### Comparing PETSc vs Native Solutions

Verify that PETSc and native implementations give the same results:

```julia
using MultiGridBarrierPETSc
using MultiGridBarrier
using LinearAlgebra
MultiGridBarrierPETSc.Init()

# Solve with PETSc (distributed)
sol_petsc_dist = fem2d_petsc_solve(Float64; L=2, p=1.0, verbose=false)
z_petsc = petsc_to_native(sol_petsc_dist).z

# Solve with native (sequential, on rank 0)
rank = MPI.Comm_rank(MPI.COMM_WORLD)
if rank == 0
    sol_native = MultiGridBarrier.fem2d_solve(Float64; L=2, p=1.0, verbose=false)
    z_native = sol_native.z

    # Compare solutions
    diff = norm(z_petsc - z_native) / norm(z_native)
    println("Relative difference: ", diff)
    @assert diff < 1e-10 "Solutions should match!"
end
```

## IO and Output

### Printing from One Rank

Use `io0()` to print from rank 0 only:

```julia
using SafePETSc

# This prints once (from rank 0)
println(io0(), "Hello from rank 0!")

# Without io0(), this prints from ALL ranks
println("Hello from rank ", MPI.Comm_rank(MPI.COMM_WORLD))
```

### Displaying PETSc Types

PETSc Vec and Mat types implement `show()` methods, but these are collective:

```julia
using SafePETSc  # For io0()
using MultiGridBarrierPETSc
MultiGridBarrierPETSc.Init()

g = fem2d_petsc(Float64; maxh=0.5)

# Collective operation - all ranks participate
println(io0(), g.w)  # Prints the Vec once on rank 0
```

## Performance Considerations

### Mesh Size and MPI Ranks

For efficient parallel computation:

- **Small problems** (L ≤ 3): Use 1-4 MPI ranks
- **Medium problems** (L = 4-5): Use 4-16 MPI ranks
- **Large problems** (L ≥ 6): Use 16+ MPI ranks

### MUMPS Solver Configuration

The MUMPS sparse direct solver is configured automatically during `Init()`:

```julia
MultiGridBarrierPETSc.Init()  # Prints "Initializing MultiGridBarrierPETSc with solver options..."
```

MUMPS provides exact sparse direct solves for MPIAIJ (sparse) matrices, ensuring accurate Newton iterations in the barrier method.

## Common Patterns

### Solve and Extract Specific Values

```julia
using SafePETSc  # For io0()
using MultiGridBarrierPETSc
MultiGridBarrierPETSc.Init()

sol = fem2d_petsc_solve(Float64; L=3, p=1.0)
sol_native = petsc_to_native(sol)

# Access solution data
z = sol_native.z  # Solution matrix
iters = sum(sol_native.SOL_main.its)  # Total Newton steps
elapsed = sol_native.SOL_main.t_elapsed  # Elapsed time in seconds

println(io0(), "Converged in $iters iterations")
println(io0(), "Elapsed time: $elapsed seconds")
```

## 1D Problems

MultiGridBarrierPETSc supports 1D finite element problems through MultiGridBarrier.jl.

### Basic 1D Example

```julia
using MultiGridBarrierPETSc
using SafePETSc  # For io0()
MultiGridBarrierPETSc.Init()

# Solve a 1D problem with 4 multigrid levels (2^4 = 16 elements)
sol = fem1d_petsc_solve(Float64; L=4, p=1.0, verbose=true)

# Convert solution to native types for analysis
sol_native = petsc_to_native(sol)

println(io0(), "Solution computed successfully!")
println(io0(), "Newton steps: ", sum(sol_native.SOL_main.its))
```

### 1D Geometry Creation

For more control, create the geometry separately:

```julia
using MultiGridBarrierPETSc
using MultiGridBarrier
MultiGridBarrierPETSc.Init()

# Create 1D PETSc geometry
g = fem1d_petsc(Float64; L=4)

# Solve with custom parameters
sol = amgb(g;
    p=1.0,           # Barrier power parameter
    verbose=true,    # Print convergence info
    maxit=100)       # Maximum iterations

# Convert solution back to native types
sol_native = petsc_to_native(sol)
```

### 1D Parameters

The `fem1d_petsc` and `fem1d_petsc_solve` functions accept:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `L` | Number of multigrid levels (creates 2^L elements) | 4 |

### Comparing 1D PETSc vs Native Solutions

```julia
using MultiGridBarrierPETSc
using MultiGridBarrier
using LinearAlgebra
using SafePETSc  # For io0()
MultiGridBarrierPETSc.Init()

# Solve with PETSc (distributed)
sol_petsc = fem1d_petsc_solve(Float64; L=4, p=1.0, verbose=false)
z_petsc = petsc_to_native(sol_petsc).z

# Solve with native (sequential)
sol_native = MultiGridBarrier.fem1d_solve(Float64; L=4, p=1.0, verbose=false)
z_native = sol_native.z

# Compare solutions
diff = norm(z_petsc - z_native) / norm(z_native)
println(io0(), "Relative difference: ", diff)
```

## 3D Problems

MultiGridBarrierPETSc also supports 3D hexahedral finite elements through MultiGridBarrier.jl.

### Basic 3D Example

```julia
using MultiGridBarrierPETSc
using SafePETSc  # For io0()
MultiGridBarrierPETSc.Init()

# Solve a 3D problem with Q3 elements and 2 multigrid levels
sol = fem3d_petsc_solve(Float64; L=2, k=3, p=1.0, verbose=true)

# Convert solution to native types for analysis
sol_native = petsc_to_native(sol)

println(io0(), "Solution computed successfully!")
println(io0(), "Newton steps: ", sum(sol_native.SOL_main.its))
```

### 3D Geometry Creation

For more control, create the geometry separately:

```julia
using MultiGridBarrierPETSc
using MultiGridBarrier
MultiGridBarrierPETSc.Init()

# Create 3D PETSc geometry
g = fem3d_petsc(Float64; L=2, k=3)

# Solve with custom parameters
sol = amgb(g;
    p=1.0,           # Barrier power parameter
    verbose=true,    # Print convergence info
    maxit=100)       # Maximum iterations

# Convert solution back to native types
sol_native = petsc_to_native(sol)
```

### 3D Parameters

The `fem3d_petsc` and `fem3d_petsc_solve` functions accept:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `L` | Number of multigrid levels | 2 |
| `k` | Polynomial order of elements (Q_k) | 3 |
| `K` | Coarse Q1 mesh (N×3 matrix, 8 vertices per hex) | Unit cube [-1,1]³ |

### Comparing 3D PETSc vs Native Solutions

```julia
using MultiGridBarrierPETSc
using MultiGridBarrier
using LinearAlgebra
using SafePETSc  # For io0()
MultiGridBarrierPETSc.Init()

# Solve with PETSc (distributed)
sol_petsc = fem3d_petsc_solve(Float64; L=2, k=2, p=1.0, verbose=false)
z_petsc = petsc_to_native(sol_petsc).z

# Solve with native (sequential)
sol_native = MultiGridBarrier.fem3d_solve(Float64; L=2, k=2, p=1.0, verbose=false)
z_native = sol_native.z

# Compare solutions
diff = norm(z_petsc - z_native) / norm(z_native)
println(io0(), "Relative difference: ", diff)
```

## Time-Dependent (Parabolic) Problems

MultiGridBarrierPETSc supports time-dependent parabolic PDEs through MultiGridBarrier.jl's `parabolic_solve` function. This solves p-Laplace heat equations using implicit Euler timestepping.

### Basic Parabolic Example

```julia
using MultiGridBarrierPETSc
using MultiGridBarrier
using SafePETSc  # For io0()
MultiGridBarrierPETSc.Init()

# Create PETSc geometry
g = fem2d_petsc(Float64; L=2)

# Solve time-dependent problem from t=0 to t=1 with timestep h=0.2
sol = parabolic_solve(g; h=0.2, p=1.0, verbose=true)

println(io0(), "Parabolic solve completed!")
println(io0(), "Number of timesteps: ", length(sol.ts))
println(io0(), "Time points: ", sol.ts)
```

### Parabolic Parameters

The `parabolic_solve` function accepts:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `h` | Time step size | 0.2 |
| `t0` | Initial time | 0.0 |
| `t1` | Final time | 1.0 |
| `ts` | Custom time grid (overrides t0, t1, h) | `t0:h:t1` |
| `p` | Exponent for p-Laplacian | 1.0 |
| `f1` | Source term function `(t, x) -> T` | `(t,x) -> 0.5` |
| `g` | Initial/boundary condition `(t, x) -> Vector{T}` | Dimension-dependent |
| `verbose` | Print progress bar | true |

### 1D Parabolic Problem

```julia
using MultiGridBarrierPETSc
using MultiGridBarrier
using SafePETSc
MultiGridBarrierPETSc.Init()

# Create 1D PETSc geometry
g = fem1d_petsc(Float64; L=4)

# Solve parabolic problem with finer timesteps
sol = parabolic_solve(g; h=0.1, t1=2.0, p=1.5, verbose=true)

println(io0(), "Solution has ", length(sol.u), " time snapshots")
```

### Accessing Parabolic Solutions

The `ParabolicSOL` structure contains:

```julia
using MultiGridBarrierPETSc
using MultiGridBarrier
using SafePETSc
MultiGridBarrierPETSc.Init()

g = fem2d_petsc(Float64; L=2)
sol = parabolic_solve(g; h=0.25, p=1.0, verbose=false)

# Access solution components
println(io0(), "Geometry type: ", typeof(sol.geometry))
println(io0(), "Time points: ", sol.ts)
println(io0(), "Number of snapshots: ", length(sol.u))

# Each sol.u[k] is a Mat containing the solution at time ts[k]
# Rows are mesh nodes, columns are solution components
```

### Converting Parabolic Solutions to Native Types

Use `petsc_to_native` to convert parabolic solutions back to native Julia types for analysis or plotting:

```julia
using MultiGridBarrierPETSc
using MultiGridBarrier
using SafePETSc
MultiGridBarrierPETSc.Init()

g = fem2d_petsc(Float64; L=2)
sol_petsc = parabolic_solve(g; h=0.25, p=1.0, verbose=false)

# Convert to native types
sol_native = petsc_to_native(sol_petsc)

# Now sol_native.u contains Vector{Matrix{Float64}}
# and sol_native.geometry contains native arrays
println(io0(), "Native u type: ", typeof(sol_native.u))
println(io0(), "Snapshot size: ", size(sol_native.u[1]))
```

### Mathematical Background

The parabolic solver solves the p-Laplace heat equation:

```math
u_t - \nabla \cdot (\|\nabla u\|_2^{p-2}\nabla u) = -f_1
```

using implicit Euler timestepping. At each time step, it solves an optimization problem via the barrier method, making it naturally compatible with PETSc's distributed linear algebra.

## Next Steps

- See the [API Reference](@ref) for detailed function documentation
- Check the `examples/` directory for complete runnable examples
- Consult MultiGridBarrier.jl documentation for barrier method theory and 3D FEM details
