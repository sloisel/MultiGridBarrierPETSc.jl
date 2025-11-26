# Installation

## Prerequisites

### MPI and PETSc

**For most users, no manual installation is required.** When you install MultiGridBarrierPETSc.jl, Julia automatically provides:
- `MPI.jl` with `MPI_jll` (bundled MPI implementation)
- `PETSc.jl` with `PETSc_jll` (precompiled PETSc binary)

This works out of the box on most systems.

!!! note "MUMPS Support"
    MultiGridBarrierPETSc.jl uses the MUMPS direct solver. The bundled `PETSc_jll` includes MUMPS on macOS, but may not on some Linux distributions. If you encounter solver errors, you may need a custom PETSc build (see below).

**HPC and Custom Builds**

For high-performance computing environments or when the default `PETSc_jll` lacks MUMPS support, follow the [official PETSc.jl configuration guide](https://juliaparallel.org/PETSc.jl/dev/man/getting_started/#Using-a-custom-build-of-the-library) to link to an external PETSc installation.

**Linux Build Considerations**

On some Linux platforms, there are known incompatibilities between Julia's bundled libraries and PETSc's optional HDF5 and curl features (see [HDF5.jl issue #1079](https://github.com/JuliaIO/HDF5.jl/issues/1079) for details). If you build PETSc from source, you may need to disable these features:

```bash
./configure --with-hdf5=0 --with-ssl=0 [other options...]
```

See the [PETSc installation documentation](https://petsc.org/release/install/install/) for complete configuration options.

## Package Installation

### Basic Installation

```julia
using Pkg
Pkg.add("MultiGridBarrierPETSc")
```

### Development Installation

To install the development version:

```julia
using Pkg
Pkg.add(url="https://github.com/sloisel/MultiGridBarrierPETSc.jl")
```

Or clone and develop locally:

```bash
git clone https://github.com/sloisel/MultiGridBarrierPETSc.jl
cd MultiGridBarrierPETSc.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Verification

Test your installation with 4 MPI ranks:

```bash
cd MultiGridBarrierPETSc.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

All tests should pass. Expected output:
```
Test Summary:                    | Pass  Total
MultiGridBarrierPETSc Test Suite |   ##     ##
```

## Initialization Pattern

!!! tip "Initialization Pattern"
    Load the package first, then initialize:

```julia
# ✓ CORRECT
using MultiGridBarrierPETSc
MultiGridBarrierPETSc.Init()  # Initialize MPI and PETSc after loading package

# ✗ WRONG - Init() must be called before using any functions
using MultiGridBarrierPETSc
# Missing MultiGridBarrierPETSc.Init() - will fail when calling functions
```

## Running MPI Programs

### Interactive REPL (Single Rank)

For development and testing on a single rank:

```julia
using MultiGridBarrierPETSc
MultiGridBarrierPETSc.Init()

# Your code here...
```

### Multi-Rank Execution

For distributed execution, create a script file (e.g., `my_program.jl`):

```julia
using SafePETSc  # For io0()
using MultiGridBarrierPETSc
MultiGridBarrierPETSc.Init()

# Your parallel code here
sol = fem2d_petsc_solve(Float64; L=3, p=1.0)
println(io0(), "Solution computed!")
```

Run with Julia's MPI launcher:

```bash
julia -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) my_program.jl`)'
```

This uses `MPI.mpiexec()` to get the correct MPI launcher configured for your Julia installation, avoiding compatibility issues with system `mpiexec`.

!!! tip "Julia Options"
    Add `--project`, `--threads`, or other Julia options as needed for your environment:
    ```bash
    julia --project=/path/to/project -e 'using MPI; run(`$(MPI.mpiexec()) -n 4 $(Base.julia_cmd()) --project=/path/to/project my_program.jl`)'
    ```

!!! tip "Output from Rank 0 Only"
    Use `io0()` from SafePETSc for output to avoid duplicate messages:
    ```julia
    println(io0(), "This prints once from rank 0")
    ```

## Troubleshooting

### MPI Issues

If you see MPI-related errors, try rebuilding MPI.jl:

```julia
using Pkg; Pkg.build("MPI")
```

### PETSc/MUMPS Issues

If PETSc fails to load:

1. Rebuild SafePETSc: `julia -e 'using Pkg; Pkg.build("SafePETSc")'`
2. Check PETSc installation: `julia -e 'using MultiGridBarrierPETSc; MultiGridBarrierPETSc.Init(); println("OK")'`

### Test Failures

If tests fail:

1. Ensure you're using at least Julia 1.10 (LTS version)
2. Check all dependencies are installed: `Pkg.status()`
3. Run with verbose output: `Pkg.test("MultiGridBarrierPETSc"; test_args=["--verbose"])`

## Next Steps

Once installed, proceed to the [User Guide](@ref) to learn how to use the package.
