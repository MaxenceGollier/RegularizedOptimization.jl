# # Solve Large-Scale Problem with FletcherPenaltySolver.jl

# In this tutorial we use `fps_solve` to solve a large-scale optimization problem resulting from the discretization of a PDE-constrained optimization problem and compare the solve with Ipopt.

# ## Problem Statement

# Let Ω = (-1,1)², we solve the following distributed Poisson control problem with Dirichlet boundary:
# ```math
#    \left\lbrace
#    \begin{aligned}
#       \min_{y \in H^1_0, u \in H^1} \quad &  \frac{1}{2} \int_\Omega |y(x) - y_d(x)|^2dx + \frac{\alpha}{2} \int_\Omega |u|^2dx \\
#       \text{s.t.} & -\Delta y = h + u, \quad x \in \Omega, \\
#                   & y = 0, \quad x \in \partial \Omega,
#    \end{aligned}
#    \right.
# ```
# where yd(x) = -x₁² and α = 1e-2.
# The force term is h(x₁, x₂) = - sin(ω x₁)sin(ω x₂) with  ω = π - 1/8.

# We refer to [Gridap.jl](https://github.com/gridap/Gridap.jl) for more details on modeling PDEs and [PDENLPModels.jl](https://github.com/JuliaSmoothOptimizers/PDENLPModels.jl) for PDE-constrained optimization problems.

using Gridap, PDENLPModels

# Definition of the domain and discretization
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

# Overall, we built a GridapPDENLPModel, which implements the [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl) API.
nlp = GridapPDENLPModel(x0, f, trian, Ypde, Ycon, Xpde, Xcon, op, name = "Control elastic membrane")

(nlp.meta.nvar, nlp.meta.ncon)

using RegularizedOptimization, Logging

const global options =
  ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-9, ϵr = 1e-9, verbose = 10, spectral = true)
function J!(nlp::AbstractNLPModel,z::Matrix{Float64},x::Vector{Float64})
    z .= jac(nlp,x)
end
stats_fps_solve = with_logger(NullLogger()) do
  
    Penalization(
        x -> obj(nlp,x),
        (g,x) -> grad!(nlp,x,g),
        (x,c) -> cons!(nlp,x,c),
        (x,j) -> J!(nlp,j,x),
        nlp.meta.ncon,
        options,
        nlp.meta.x0
    ) 
end