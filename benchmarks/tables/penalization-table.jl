include("regulopt-tables.jl")
using OptimizationProblems, ADNLPModels, OptimizationProblems.ADNLPProblems
using Percival
using Random

using Gridap, PDENLPModels,Krylov


n = 20

# Domain
domain = (-1, 1, -1, 1)
partition = (n, n)
model = CartesianDiscreteModel(domain, partition)

# Definition of the spaces:
valuetype = Float64
reffe = ReferenceFE(lagrangian, valuetype, 1)
Xpde = TestFESpace(model, reffe; conformity = :H1, dirichlet_tags = "boundary")
y0(x) = 0.0
Ypde = TrialFESpace(Xpde, y0)

reffe_con = ReferenceFE(lagrangian, valuetype, 1)
Xcon = TestFESpace(model, reffe_con; conformity = :H1)
Ycon = TrialFESpace(Xcon)
Y = MultiFieldFESpace([Ypde, Ycon])

# Integration machinery
trian = Triangulation(model)
degree = 1
dΩ = Measure(trian, degree)

# Objective function:
yd(x) = -x[1]^2
α = 1e-2
function f(y, u)
  ∫(0.5 * (yd - y) * (yd - y) + 0.5 * α * u * u) * dΩ
end

# Definition of the constraint operator
ω = π - 1 / 8
h(x) = -sin(ω * x[1]) * sin(ω * x[2])
function res(yu, v)
  y, u = yu
  ∫(∇(v) ⊙ ∇(y) - v * u) * dΩ #- v * h
end
rhs(v) = ∫(v * h) * dΩ
op = AffineFEOperator(res, rhs, Y, Xpde)

npde = Gridap.FESpaces.num_free_dofs(Ypde)
ncon = Gridap.FESpaces.num_free_dofs(Ycon)

nlp = GridapPDENLPModel(
  zeros(npde + ncon),
  f,
  dΩ,
  Ypde,
  Ycon,
  Xpde,
  Xcon,
  op,
  lvaru = zeros(ncon),
  uvaru = ones(ncon),
  name = "controlelasticmembrane1",
)


(nlp.meta.nvar, nlp.meta.ncon)
verbose = 10 # 10
ν = 1.0
ϵ = 1e-3
ϵi = 0.0
ϵri = 0.0
maxIter = 500
maxIter_inner = 100
options =
  ROSolverOptions(ν = ν,β=1e16, ϵa = 0.0, ϵr = 0.0, verbose = verbose, maxIter = maxIter)


options2 = ROSolverOptions(ϵa = ϵ, ϵr = ϵ, maxIter = maxIter_inner)

solvers = [:L2Penalty,:percival]
subsolvers =
  [:R2,:None]
solver_options = [
  options,
  options,]
subsolver_options = [
  options2,
  options2,]

x = benchmark_table(
    nlp,
    solvers,
    subsolvers,
    solver_options,
    subsolver_options,
    tol = ϵ,
    tex = false
);
