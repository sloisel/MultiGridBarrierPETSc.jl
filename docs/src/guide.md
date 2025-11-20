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
sol_native = sol_petsc_to_native(sol_petsc)

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
g_petsc = geometry_native_to_petsc(g_native)
```

**Type mappings:**

| Native Type | PETSc Type | Storage |
|-------------|------------|---------|
| `Matrix{T}` | `Mat{T, MPIDENSE}` | Dense distributed |
| `Vector{T}` | `Vec{T, MPIDENSE}` | Dense distributed |
| `SparseMatrixCSC{T,Int}` | `Mat{T, MPIAIJ}` | Sparse distributed |

### PETSc to Native

Convert PETSc types back to native Julia arrays:

```julia
# Create and solve with PETSc types
g_petsc = fem2d_petsc(Float64; maxh=0.3)
sol_petsc = amgb(g_petsc; p=2.0)

# Convert back for analysis
g_native = geometry_petsc_to_native(g_petsc)
sol_native = sol_petsc_to_native(sol_petsc)

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
g_petsc = geometry_native_to_petsc(g_native)

# 3. Solve with custom barrier parameters
sol_petsc = amgb(g_petsc;
    p=1.5,           # Barrier power parameter
    verbose=true,    # Print convergence info
    maxit=100,       # Maximum iterations
    tol=1e-8)        # Convergence tolerance

# 4. Convert solution back
sol_native = sol_petsc_to_native(sol_petsc)

# 5. Access solution components
println(io0(), "Objective value: ", sol_native.SOL_main.objective)
println(io0(), "Iterations: ", length(sol_native.log))
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
z_petsc = sol_petsc_to_native(sol_petsc_dist).z

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

MUMPS provides exact sparse direct solves for MPIAIJ (sparse) matrices, ensuring accurate Newton iterations in the barrier method. Dense matrices (MPIDENSE) use PETSc's default dense LU solver.

## Common Patterns

### Solve and Extract Specific Values

```julia
using SafePETSc  # For io0()
using MultiGridBarrierPETSc
MultiGridBarrierPETSc.Init()

sol = fem2d_petsc_solve(Float64; L=3, p=1.0)
sol_native = sol_petsc_to_native(sol)

# Access solution data
z = sol_native.z  # Solution matrix
obj = sol_native.SOL_main.objective  # Objective function value
iters = length(sol_native.log)  # Number of iterations

println(io0(), "Converged in $iters iterations")
println(io0(), "Final objective: $obj")
```

## Next Steps

- See the [API Reference](@ref) for detailed function documentation
- Check the `examples/` directory for complete runnable examples
- Consult MultiGridBarrier.jl documentation for barrier method theory
