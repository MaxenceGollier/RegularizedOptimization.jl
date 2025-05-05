using LinearAlgebra: length
using LinearAlgebra, Random, Test
using ProximalOperators
using NLPModels, NLPModelsModifiers, RegularizedProblems, RegularizedOptimization, SolverCore, CUTEst, JET

compound = 1
nz = 10 * compound
options = ROSolverOptions(ν = 1.0, β = 1e16, ϵa = 1e-6, ϵr = 1e-6, verbose = 10)
nlp = CUTEstModel("JUDGE")
#bpdn2, bpdn_nls2, sol2 = bpdn_model(compound, bounds = true)
λ = 1.0

macro wrappedallocs(expr)
  kwargs = [a for a in expr.args if isa(a, Expr)]
  args = [a for a in expr.args if isa(a, Symbol)]

  argnames = [gensym() for a in args]
  kwargs_dict = Dict{Symbol, Any}(a.args[1] => a.args[2] for a in kwargs if a.head == :kw)
  quote
    function g($(argnames...); kwargs_dict...)
      @allocated $(Expr(expr.head, argnames..., kwargs...))
    end
    $(Expr(:call, :g, [esc(a) for a in args]...))
  end
end

h = NormL0(λ)
reg_nlp = RegularizedNLPModel(LBFGSModel(nlp), h)
solver = R2NSolver(reg_nlp)#, store_h = true)
stats = RegularizedExecutionStats(reg_nlp)
println(@wrappedallocs(
          solve!(solver, reg_nlp, stats, σk = 1.0, atol = 1e-6, rtol = 1e-6)#, verbose = 1)
        ) )
finalize(nlp)
