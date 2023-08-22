include("regulopt-tables.jl")
using OptimizationProblems, ADNLPModels, OptimizationProblems.ADNLPProblems
using FletcherPenaltySolver, Percival
using Random
"""
using Gridap, PDENLPModels


n = 20
domain = (-1, 1, -1, 1)
partition = (n, n)
model = CartesianDiscreteModel(domain, partition)

# Definition of the FE-spaces
reffe = ReferenceFE(lagrangian, Float64, 1)
Xpde = TestFESpace(model, reffe; conformity = :H1, dirichlet_tags = "boundary")
y0(x) = 0.0
Ypde = TrialFESpace(Xpde, y0)

reffe_con = ReferenceFE(lagrangian, Float64, 1)
Xcon = TestFESpace(model, reffe_con; conformity = :H1)
Ycon = TrialFESpace(Xcon)
Y = MultiFieldFESpace([Ypde, Ycon])

# Integration machinery
trian = Triangulation(model)
degree = 1
dΩ = Measure(trian, degree)

# Objective function
yd(x) = -x[1]^2
α = 1e-2
function f(y, u)
  ∫(0.5 * (yd - y) * (yd - y) + 0.5 * α * u * u) * dΩ
end

# Definition of the constraint operator
ω = π - 1 / 8
h(x) = -sin(ω * x[1]) * sin(ω * x[2])
function res(y, u, v)
  ∫(∇(v) ⊙ ∇(y) - v * u - v * h) * dΩ
end
op = FEOperator(res, Y, Xpde)

# Definition of the initial guess
npde = Gridap.FESpaces.num_free_dofs(Ypde)
ncon = Gridap.FESpaces.num_free_dofs(Ycon)
x0 = zeros(npde + ncon);
println(npde)
println(ncon)
# Overall, we built a GridapPDENLPModel, which implements the [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) API.
nlp = GridapPDENLPModel(x0, f, trian, Ypde, Ycon, Xpde, Xcon, op, name = "Control elastic membrane")

(nlp.meta.nvar, nlp.meta.ncon)
"""
verbose = 10 # 10
ν = 1.0
ϵ = 1e-3
ϵi = 0.0
ϵri = 0.0
maxIter = 500
maxIter_inner = 100
options =
  ROSolverOptions(ν = ν,β=1e16, ϵa = ϵ, ϵr = ϵ, verbose = verbose, maxIter = maxIter, spectral = true)


options2 = ROSolverOptions(spectral = false, psb = true, ϵa = ϵ, ϵr = ϵ, maxIter = maxIter_inner)

solvers = [:Penalization,:Penalization,:fps_solve,:percival]
subsolvers =
  [:R2,:R2N,:None,:None]
solver_options = [
  options,
  options,
  options,
  options,
]
subsolver_options = [
  options2,
  options2,
  options2,
  options2,
]
models = [
  :None,:LBFGSModel,:None,:None
]
# model

meta = OptimizationProblems.meta
problem_list = meta[(meta.has_equalities_only .== 1) .& (meta.has_bounds.==0) .& (meta.has_fixed_variables.==0) .& (meta.variable_nvar .== 0), :]


problem = problem_list[rand(1:size(problem_list)[1]),:]

nlp = eval(Meta.parse(problem.name))()

x = benchmark_table(
    nlp,
    models,
    solvers,
    subsolvers,
    solver_options,
    subsolver_options,
    tol = ϵ,
    tex = true
);


"""
stats = Penalization(nlp,options,subsolver_options=options2)
yfv = stats.solution[1:Gridap.FESpaces.num_free_dofs(nlp.pdemeta.Ypde)]
yh  = FEFunction(nlp.pdemeta.Ypde, yfv)
ufv = stats.solution[1+Gridap.FESpaces.num_free_dofs(nlp.pdemeta.Ypde):end]
uh  = FEFunction(nlp.pdemeta.Ycon, ufv)
writevtk(nlp.pdemeta.tnrj.trian,"results",cellfields=["uh"=>uh, "yh"=>yh])
"""