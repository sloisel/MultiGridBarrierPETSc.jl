using Documenter
using MultiGridBarrierPETSc

makedocs(;
    modules=[MultiGridBarrierPETSc],
    authors="Sebastien Loisel and contributors",
    sitename="MultiGridBarrierPETSc.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://sloisel.github.io/MultiGridBarrierPETSc.jl",
        repolink="https://github.com/sloisel/MultiGridBarrierPETSc.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Installation" => "installation.md",
        "User Guide" => "guide.md",
        "API Reference" => "api.md",
    ],
    repo=Documenter.Remotes.GitHub("sloisel", "MultiGridBarrierPETSc.jl"),
    warnonly=true,  # Don't fail on warnings during development
)

deploydocs(;
    repo="github.com/sloisel/MultiGridBarrierPETSc.jl",
    devbranch="main",
)
