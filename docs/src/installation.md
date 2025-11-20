# Installation

## Prerequisites

### 1. MPI Installation

MultiGridBarrierPETSc.jl requires an MPI implementation. Install one of the following:

**macOS (Homebrew):**
```bash
brew install open-mpi
```

**Ubuntu/Debian:**
```bash
sudo apt-get install libopenmpi-dev
```

**Fedora/RHEL:**
```bash
sudo dnf install openmpi-devel
```

**Verify MPI installation:**
```bash
mpiexec --version
```

### 2. PETSc with MUMPS

PETSc with MUMPS is required for the direct solver used in Newton iterations.

**macOS (Homebrew - Recommended):**
```bash
brew install petsc
```
Homebrew's PETSc includes MUMPS support by default.

**Linux - Building from Source:**

On Linux, you'll need to build PETSc from source with MUMPS. Due to compatibility issues with HDF5 and curl in some build environments, we recommend building without these optional dependencies:

```bash
# Download PETSc
git clone -b release https://gitlab.com/petsc/petsc.git petsc
cd petsc

# Configure with MUMPS, without HDF5/curl
./configure \
  --with-debugging=0 \
  --download-mumps \
  --download-scalapack \
  --download-parmetis \
  --download-metis \
  --download-openblas \
  --with-hdf5=0 \
  --with-ssl=0

# Build (use appropriate -j value for your system)
make PETSC_DIR=$PWD PETSC_ARCH=arch-linux-c-opt all

# Set environment variable for Julia
export JULIA_PETSC_LIBRARY=/path/to/petsc/arch-linux-c-opt/lib/libpetsc.so
```

Add the `JULIA_PETSC_LIBRARY` export to your `~/.bashrc` or `~/.zshrc` for persistence.

!!! note "HDF5/curl Build Issues"
    The `--with-hdf5=0 --with-ssl=0` flags disable HDF5 and curl support, which can cause build failures in CI environments and some Linux distributions. These features are not required for MultiGridBarrierPETSc.jl. See [PETSc installation documentation](https://petsc.org/release/install/) for more details.

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
Pkg.add(url="https://github.com/yourusername/MultiGridBarrierPETSc.jl")
```

Or clone and develop locally:

```bash
git clone https://github.com/yourusername/MultiGridBarrierPETSc.jl
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

Run with `mpiexec`:

```bash
mpiexec -n 4 julia --project my_program.jl
```

!!! tip "Output from Rank 0 Only"
    Use `io0()` from SafePETSc for output to avoid duplicate messages:
    ```julia
    println(io0(), "This prints once from rank 0")
    ```

## Troubleshooting

### MPI Not Found

If you see `ERROR: MPI not properly initialized`:

1. Verify MPI is installed: `mpiexec --version`
2. Rebuild MPI.jl: `julia -e 'using Pkg; Pkg.build("MPI")'`
3. Set MPI binary explicitly:
   ```julia
   ENV["JULIA_MPI_BINARY"] = "system"
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
