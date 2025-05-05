using NLPModels,
  NLPModelsModifiers,
  RegularizedProblems,
  RegularizedOptimization,
  SolverCore, CUTEst, SparseMatricesCOO

nlp = CUTEstModel("MSS1")

#H = hess(nlp, nlp.meta.x0)
#J = jac(nlp, nlp.meta.x0)

H = hess_op(nlp, nlp.meta.x0)
K = SparseKKTCOO(SparseMatrixCOO(Matrix(H)), SparseMatrixCOO(J))
println(SparseMatrixCOO(K.m+K.n, K.m+K.n, K.rows, K.cols, K.vals))
#finalize(nlp)

"""
out = L2Penalty(
  nlp,
  atol = 1e-3,
  rtol = 1e-3,
  ktol = 1e-1,
  max_iter = 100,
  max_time = 10.0,
  verbose = 1,
)"""

error("done")
tol = 1e-3
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

#solver = L2PenaltySolver(nlp)#, subsolver = R2NSolver)
#stats = GenericExecutionStats(nlp)
stats = L2Penalty(
      nlp,
      max_time=100.0,
      subsolver = R2NSolver,#NSolver,
      ktol=100 * tol,
      atol=0.0,
      rtol=0.0,
      neg_tol=sqrt(tol),
      τ=500.0,
      callback=(nlp, solver, stats) -> begin

        if stats.primal_feas < tol && stats.dual_feas < tol
          stats.status = :user          
        end
      end,
      verbose = 1,
      sub_verbose = 1
    )
println(stats)
#println(@wrappedallocs solve!(solver, nlp, stats, atol = 1e-3, rtol = 1e-3, ktol = 1e-1, max_iter = 100, max_time = 10.0))
#solve!(solver, nlp, stats, subsolver = R2NSolver, atol = 1e-3, rtol = 1e-3, ktol = 1e-1, max_iter = 100, max_time = 10.0, verbose = 1)
finalize(nlp)
finalize(nlp.model)
##TODO: - remove allocs with R2
