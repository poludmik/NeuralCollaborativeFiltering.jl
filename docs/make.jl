using Documenter, NeuralCollaborativeFiltering
using DataFrames


makedocs(sitename="NeuralCollaborativeFiltering.jl",
    pages = [
        "Project" => "index.md",
        "Modules" => [
            "models.md",
            "eval.md",
            "scripts.md"
        ]
    ]
)

deploydocs(
    repo = "github.com/poludmik/NeuralCollaborativeFiltering.git",
)