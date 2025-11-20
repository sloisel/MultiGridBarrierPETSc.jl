# MultiGridBarrierPETSc.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://sloisel.github.io/MultiGridBarrierPETSc.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sloisel.github.io/MultiGridBarrierPETSc.jl/dev/)
[![Build Status](https://github.com/sloisel/MultiGridBarrierPETSc.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/sloisel/MultiGridBarrierPETSc.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/sloisel/MultiGridBarrierPETSc.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/sloisel/MultiGridBarrierPETSc.jl)

**A Julia package that bridges [MultiGridBarrier.jl](https://github.com/sloisel/MultiGridBarrier.jl) and [SafePETSc.jl](https://github.com/sloisel/SafePETSc.jl) for distributed multigrid barrier computations.**

MultiGridBarrierPETSc.jl extends MultiGridBarrier's API to work with PETSc's distributed Mat and Vec types, enabling efficient parallel computation of multigrid barrier methods across multiple MPI ranks.

## Quick Start

```julia
using SafePETSc
using MultiGridBarrierPETSc
SafePETSc.Init()  # Initialize MPI and PETSc

# Solve a 2D finite element problem with PETSc distributed types
sol = fem2d_petsc_solve(Float64; L=3, p=1.0, verbose=false)

# Convert to native types for plotting
using MultiGridBarrier
sol_native = sol_petsc_to_native(sol)
plot(sol_native)
```

## Key Features

- **Drop-in Replacement**: Seamlessly use PETSc types with MultiGridBarrier's API
- **Distributed Computing**: Leverage PETSc's distributed linear algebra for large-scale problems
- **Type Conversion**: Easy conversion between native Julia arrays and PETSc distributed types
- **MPI-Aware**: All operations correctly handle MPI collective requirements
- **Automatic MUMPS**: Direct solver automatically configured for accurate Newton iterations

## Installation

```julia
using Pkg
Pkg.add("MultiGridBarrierPETSc")
```

**Prerequisites:**
- Julia 1.9 or later
- MPI installation (OpenMPI, MPICH, or Intel MPI)
- PETSc with MUMPS (macOS: `brew install petsc`, Linux: build from source)

See the [Installation Guide](https://sloisel.github.io/MultiGridBarrierPETSc.jl/dev/installation/) for detailed instructions.

## Documentation

- **[User Guide](https://sloisel.github.io/MultiGridBarrierPETSc.jl/dev/guide/)**: Workflows, examples, and best practices
- **[API Reference](https://sloisel.github.io/MultiGridBarrierPETSc.jl/dev/api/)**: Complete function documentation
- **[Examples](examples/)**: Runnable example scripts

## Running Examples

```bash
# Basic solve example
mpiexec -n 4 julia --project examples/basic_solve.jl

# Round-trip conversion example
mpiexec -n 4 julia --project examples/roundtrip_conversion.jl
```

## Testing

Run the test suite with 4 MPI ranks:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

## Type Mappings

| Native Julia Type | PETSc Distributed Type | Storage |
|-------------------|------------------------|---------|
| `Matrix{T}` | `Mat{T, MPIDENSE}` | Dense distributed |
| `Vector{T}` | `Vec{T, MPIDENSE}` | Dense distributed |
| `SparseMatrixCSC{T,Int}` | `Mat{T, MPIAIJ}` | Sparse distributed |

## Package Ecosystem

- **[MultiGridBarrier.jl](https://github.com/sloisel/MultiGridBarrier.jl)**: Core multigrid barrier method
- **[SafePETSc.jl](https://github.com/sloisel/SafePETSc.jl)**: Safe PETSc bindings
- **[MPI.jl](https://github.com/JuliaParallel/MPI.jl)**: Julia MPI bindings

## Citation

If you use this package in your research, please cite:

```bibtex
@software{multigridbarrierpetsc,
  author = {Loisel, Sebastien},
  title = {MultiGridBarrierPETSc.jl: Distributed Multigrid Barrier Methods},
  year = {2024},
  url = {https://github.com/sloisel/MultiGridBarrierPETSc.jl}
}
```

## Development

To develop this package locally:

```bash
git clone https://github.com/sloisel/MultiGridBarrierPETSc.jl
cd MultiGridBarrierPETSc.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project -e 'using Pkg; Pkg.test()'
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
