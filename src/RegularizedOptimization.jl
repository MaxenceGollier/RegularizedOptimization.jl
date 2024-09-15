module RegularizedOptimization

# base dependencies
using LinearAlgebra, Logging, Printf

# external dependencies
using ProximalOperators, TSVD

# dependencies from us
using LinearOperators,
  NLPModels, NLPModelsModifiers, RegularizedProblems, ShiftedProximalOperators, SolverCore, SparseMatricesCOO, Krylov

include("utils.jl")
include("input_struct.jl")
include("PG_alg.jl")
include("Fista_alg.jl")
include("splitting.jl")
include("TR_alg.jl")
include("TRDH_alg.jl")
include("R2_alg.jl")
include("LM_alg.jl")
include("LMTR_alg.jl")
include("R2DH.jl")
include("R2N.jl")

end  # module RegularizedOptimization