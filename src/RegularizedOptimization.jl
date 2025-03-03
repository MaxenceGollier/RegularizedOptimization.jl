module RegularizedOptimization

# base dependencies
using LinearAlgebra, Logging, Printf

# external dependencies
using ProximalOperators, TSVD

# dependencies from us
using Krylov, LinearOperators,
  NLPModels, NLPModelsModifiers, RegularizedProblems, ShiftedProximalOperators, SolverCore, SparseMatricesCOO
using Percival: AugLagModel, update_y!, update_μ!

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
include("R2NModel.jl")
include("R2N.jl")
include("AL_alg.jl")
include("L2Penalty.jl")

end  # module RegularizedOptimization
