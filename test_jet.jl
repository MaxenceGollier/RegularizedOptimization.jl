using LinearAlgebra: length
using LinearAlgebra, Random, Test
using ProximalOperators
using NLPModels, NLPModelsModifiers, RegularizedProblems, RegularizedOptimization

nz = 10
options = ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = 10)
bpdn, bpdn_nls, sol = bpdn_model(1)
λ = norm(grad(bpdn, zeros(bpdn.meta.nvar)), Inf) / 10

h = NormL1(λ)

using JET
@report_opt target_modules=(RegularizedOptimization,) R2(bpdn, h, options)