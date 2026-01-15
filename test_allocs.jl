using LinearAlgebra: length
using LinearAlgebra, Random
using ProximalOperators
using LinearOperators

using NLPModels
using  NLPModelsModifiers, RegularizedProblems, RegularizedOptimization, SolverCore

macro wrappedallocs(expr)
  kwargs = [a for a in expr.args if isa(a, Expr)]
  args = [a for a in expr.args if isa(a, Symbol)]

  argnames = [gensym() for a in args]
  kwargs_dict = Dict{Symbol, Any}(a.args[1] => a.args[2] for a in kwargs if a.head == :kw)
  quote
    function g($(argnames...); kwargs_dict...)
      $(Expr(expr.head, argnames..., kwargs...)) # Call the function twice to make the allocated macro more stable
      @allocated $(Expr(expr.head, argnames..., kwargs...))
    end
    $(Expr(:call, :g, [esc(a) for a in args]...))
  end
end

Random.seed!(0)
compound = 1
bpdn, _, _ = bpdn_model(compound)
λ = norm(grad(bpdn, zeros(bpdn.meta.nvar)), Inf) / 10

h = NormL1(λ)


reg_nlp = RegularizedNLPModel(LBFGSModel(bpdn), h)
solver = R2NSolver(reg_nlp)
stats = RegularizedExecutionStats(reg_nlp)

@wrappedallocs solve!(solver, reg_nlp, stats, σk = 1.0, atol = 1e-6, rtol = 1e-6, opnorm_maxiter = -1)

