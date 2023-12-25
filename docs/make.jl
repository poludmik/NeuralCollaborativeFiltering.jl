using Documenter, NeuralCollaborativeFiltering

makedocs(sitename="NeuralCollaborativeFiltering.jl",
    pages = [
        "Home" => "index.md",
        "Modules" => [
            "models.md",
            "eval.md",
            "scripts.md"
        ]
    ]
)

deploydocs(
    repo = "github.com/poludmik/NeuralCollaborativeFiltering.jl.git",
    branch = "gh-pages",
    target = "build"
)