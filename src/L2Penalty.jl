using DelimitedFiles
export L2Penalty, L2PenaltySolver, solve!, L2_R2N_subsolver

import SolverCore.solve!

mutable struct L2PenaltySolver{
  T <: Real,
  V <: AbstractVector{T},
  S <: AbstractOptimizationSolver,
  PB <: AbstractRegularizedNLPModel,
  G1 <: ShiftedCompositeNormL2{T},
  G2 <: CompositeNormL2{T}
} <: AbstractOptimizationSolver
  x::V
  y::V
  dual_res::V
  s::V
  s0::V
  ψ::G1
  sub_h::G2
  subsolver::S
  subpb::PB
  substats::GenericExecutionStats{T, V, V, T}
end

function L2PenaltySolver(nlp::AbstractNLPModel{T, V}; subsolver = R2Solver) where {T, V}
  x0 = nlp.meta.x0
  x = similar(x0)
  s = similar(x0)
  y = similar(x0, nlp.meta.ncon)
  dual_res = similar(x0)
  s0 = zero(x0)

  # Allocating variables for the ShiftedProximalOperator structure
  (rows, cols) = jac_structure(nlp)
  vals = similar(rows, eltype(x0))
  A = SparseMatrixCOO(nlp.meta.ncon, nlp.meta.nvar, rows, cols, vals)
  b = similar(x0, eltype(x0), nlp.meta.ncon)

  # Allocate ψ = ||c(x) + J(x)s|| to compute θ
  ψ = ShiftedCompositeNormL2(
    one(T),
    (c, x) -> cons!(nlp, x, c),
    (j, x) -> jac_coord!(nlp, x, j.vals),
    A,
    b,
  )

  # Allocate sub_h = ||c(x)|| to solve min f(x) + τ||c(x)||
  sub_h =
    CompositeNormL2(one(T), (c, x) -> cons!(nlp, x, c), (j, x) -> jac_coord!(nlp, x, j.vals), A, b)
  subnlp = RegularizedNLPModel(nlp, sub_h)
  if subsolver == R2NSolver
    solver = subsolver(subnlp, subsolver = L2_R2N_subsolver)
  else
    solver = subsolver(subnlp)
  end
  subpb = RegularizedNLPModel(nlp, sub_h)
  substats = RegularizedExecutionStats(subpb)

  return L2PenaltySolver(x, y, dual_res, s, s0, ψ, sub_h, solver, subpb, substats)
end

"""
    L2Penalty(nlp; kwargs…)

An exact ℓ₂-penalty method for the problem

    min f(x) 	s.t c(x) = 0

where f: ℝⁿ → ℝ and c: ℝⁿ → ℝᵐ respectively have a Lipschitz-continuous gradient and Jacobian.

At each iteration k, an iterate is computed as 

    xₖ ∈ argmin f(x) + τₖ‖c(x)‖₂

where τₖ is some penalty parameter.
This nonsmooth problem is solved using `R2` (see `R2` for more information) with the first order model ψ(s;x) = τₖ‖c(x) + J(x)s‖₂

For advanced usage, first define a solver "L2PenaltySolver" to preallocate the memory used in the algorithm, and then call `solve!`:

    solver = L2PenaltySolver(nlp)
    solve!(solver, nlp)

    stats = GenericExecutionStats(nlp)
    solver = L2PenaltySolver(nlp)
    solve!(solver, nlp, stats)

# Arguments
* `nlp::AbstractNLPModel{T, V}`: the problem to solve, see `RegularizedProblems.jl`, `NLPModels.jl`.

# Keyword arguments 
- `x::V = nlp.meta.x0`: the initial guess;
- `atol::T = √eps(T)`: absolute tolerance;
- `rtol::T = √eps(T)`: relative tolerance;
- `neg_tol::T = eps(T)^(1 / 4)`: negative tolerance
- `ktol::T = eps(T)^(1 / 4)`: the initial tolerance sent to the subsolver
- `max_eval::Int = -1`: maximum number of evaluation of the objective function (negative number means unlimited);
- `sub_max_eval::Int = -1`: maximum number of evaluation for the subsolver (negative number means unlimited);
- `max_time::Float64 = 30.0`: maximum time limit in seconds;
- `max_iter::Int = 10000`: maximum number of iterations;
- `sub_max_iter::Int = 10000`: maximum number of iterations for the subsolver;
- `max_decreas_iter::Int = 10`: maximum number of iteration where ‖c(xₖ)‖₂ does not decrease before calling the problem locally infeasible;
- `verbose::Int = 0`: if > 0, display iteration details every `verbose` iteration;
- `sub_verbose::Int = 0`: if > 0, display subsolver iteration details every `verbose` iteration;
- `τ::T = T(100)`: initial penalty parameter;
- `β1::T = τ`: penalty update parameter: τₖ <- τₖ + β1;	
- `β2::T = T(0.1)`: tolerance decreasing factor, at each iteration, ktol <- β2*ktol;
- `β3::T = 1/τ`: initial regularization parameter σ₀ = β3/τₖ at each iteration;
- `β4::T = eps(T)`: minimal regularization parameter σ for `R2`;
other 'kwargs' are passed to `R2` (see `R2` for more information).

The algorithm stops either when `√θₖ < atol + rtol*√θ₀ ` or `θₖ < 0` and `√(-θₖ) < neg_tol` where θₖ := ‖c(xₖ)‖₂ - ‖c(xₖ) + J(xₖ)sₖ‖₂, and √θₖ is a stationarity measure.

# Output
The value returned is a `GenericExecutionStats`, see `SolverCore.jl`.

# Callback
The callback is called at each iteration.
The expected signature of the callback is `callback(nlp, solver, stats)`, and its output is ignored.
Changing any of the input arguments will affect the subsequent iterations.
In particular, setting `stats.status = :user` will stop the algorithm.
All relevant information should be available in `nlp` and `solver`.
Notably, you can access, and modify, the following:
- `solver.x`: current iterate;
- `solver.subsolver`: a `R2Solver` structure holding relevant information on the subsolver state, see `R2` for more information;
- `stats`: structure holding the output of the algorithm (`GenericExecutionStats`), which contains, among other things:
  - `stats.iter`: current iteration counter;
  - `stats.objective`: current objective function value;
  - `stats.status`: current status of the algorithm. Should be `:unknown` unless the algorithm has attained a stopping criterion. Changing this to anything will stop the algorithm, but you should use `:user` to properly indicate the intention.
  - `stats.elapsed_time`: elapsed time in seconds.
You can also use the `sub_callback` keyword argument which has exactly the same structure and in sent to `R2`.
"""
function L2Penalty(
  nlp::AbstractNLPModel{T, V};
  subsolver = R2Solver,
  kwargs...
) where {T <: Real, V}
  if !equality_constrained(nlp)
    error("L2Penalty: This algorithm only works for equality contrained problems.")
  end
  solver = L2PenaltySolver(nlp, subsolver = subsolver)
  stats = GenericExecutionStats(nlp)
  solve!(solver, nlp, stats; kwargs...)
  return stats
end

function SolverCore.solve!(
  solver::L2PenaltySolver{T, V},
  nlp::AbstractNLPModel{T, V},
  stats::GenericExecutionStats{T, V, V};
  callback = (args...) -> nothing,
  sub_callback::F = (args...) -> nothing,
  x::V = nlp.meta.x0,
  atol::T = √eps(T),
  rtol::T = √eps(T),
  neg_tol = eps(T)^(1/4),
  ktol::T = eps(T)^(1/4),
  max_iter::Int = 10000,
  sub_max_iter::Int = 10000,
  max_time::T = T(30.0),
  max_eval::Int = -1,
  sub_max_eval::Int = -1,
  max_decreas_iter::Int = 10,
  verbose::Int = 0,
  sub_verbose::Int = 0,
  τ::T = T(100),
  β1::T = τ,
  β2::T = T(0.1),
  β3::T = 1/τ,
  β4::T = eps(T),
  kwargs...,
) where {T, V, F <: Function}
  reset!(stats)

  # Retrieve workspace
  ψ = solver.ψ
  sub_h = solver.sub_h
  sub_h.h = NormL2(τ)
  solver.subsolver.ψ.h = NormL2(τ)

  x = solver.x .= x
  s = solver.s
  s0 = solver.s0
  shift!(ψ, x)
  fx = obj(nlp, x)
  hx = norm(ψ.b)

  if verbose > 0
    @info log_header(
      [:iter, :sub_iter, :fx, :hx, :theta, :xi, :epsk, :tau, :normx],
      [Int, Int, Float64, Float64, Float64, Float64, Float64, Float64, Float64],
      hdr_override = Dict{Symbol, String}(   # TODO: Add this as constant dict elsewhere
        :iter => "outer",
        :sub_iter => "inner",
        :fx => "f(x)",
        :hx => "‖c(x)‖₂",
        :theta => "√θ",
        :xi => "√(ξ/ν)",
        :epsk => "ϵₖ",
        :tau => "τ",
        :normx => "‖x‖",
      ),
      colsep = 1,
    )
  end

  set_iter!(stats, 0)
  rem_eval = max_eval
  start_time = time()
  set_time!(stats, 0.0)
  set_objective!(stats, fx)

  local θ::T
  prox!(s, ψ, s0, T(1))
  θ = hx - ψ(s)

  sqrt_θ = θ ≥ 0 ? sqrt(θ) : sqrt(-θ)
  θ < 0 &&
    sqrt_θ ≥ neg_tol &&
    error("L2Penalty: prox-gradient step should produce a decrease but θ = $(θ)")

  atol += rtol * sqrt_θ # make stopping test absolute and relative
  ktol = max(ktol, atol) # Keep ϵ₀ ≥ ϵ
  tol_init = ktol # store value of ϵ₀ 

  done = false

  n_iter_since_decrease = 0
  νsub = 1/max(β4, β3*τ)

  while !done
    if isa(solver.subsolver, R2Solver)
      solve!(
        solver.subsolver,
        solver.subpb,
        solver.substats;
        callback = sub_callback,
        x = x,
        atol = ktol,
        rtol = T(0),
        neg_tol = neg_tol,
        verbose = sub_verbose,
        max_iter = sub_max_iter,
        max_time = max_time - stats.elapsed_time,
        max_eval = min(rem_eval, sub_max_eval),
        σmin = β4,
        ν = νsub
      )
    else
      solve!(
        solver.subsolver,
        solver.subpb,
        solver.substats;
        callback = sub_callback,
        x = x,
        atol = ktol,
        rtol = T(0),
        neg_tol = neg_tol,
        verbose = sub_verbose,
        max_iter = sub_max_iter,
        max_time = max_time - stats.elapsed_time,
        max_eval = min(rem_eval, sub_max_eval),
        σmin = β4,
        σk = 1/νsub,
        sub_kwargs = Dict{Symbol, Any}()
      )
    end

    x .= solver.substats.solution
    fx = solver.substats.solver_specific[:smooth_obj]
    hx_prev = copy(hx)
    hx = solver.substats.solver_specific[:nonsmooth_obj]/τ
    sqrt_ξ_νInv = solver.substats.dual_feas

    shift!(ψ, x)
    prox!(s, ψ, s0, T(1))

    θ = hx - ψ(s)
    sqrt_θ = θ ≥ 0 ? sqrt(θ) : sqrt(-θ)
    θ < 0 &&
      sqrt_θ ≥ neg_tol &&
      error("L2Penalty: prox-gradient step should produce a decrease but θ = $(θ)")

    if sqrt_θ > ktol
      τ = τ + β1
      sub_h.h = NormL2(τ)
      solver.subsolver.ψ.h = NormL2(τ)
      νsub = 1/max(β4, β3*τ)
    else
      n_iter_since_decrease = 0
      ktol = max(β2^(ceil(log(β2, sqrt_ξ_νInv/tol_init)))*ktol, atol) #the β^... allows to directly jump to a sufficiently small ϵₖ
      νsub = 1/solver.substats.solver_specific[:sigma]
    end
    if sqrt_θ > ktol && hx_prev ≥ hx
      n_iter_since_decrease += 1
    else
      n_iter_since_decrease = 0
    end

    solved =
      (sqrt_θ ≤ atol && solver.substats.status == :first_order) ||
      (θ < 0 && sqrt_θ ≤ neg_tol && solver.substats.status == :first_order)
    (θ < 0 && sqrt_θ > neg_tol) &&
      error("L2Penalty: prox-gradient step should produce a decrease but θ = $(θ)")

    verbose > 0 &&
      stats.iter % verbose == 0 &&
      @info log_row(
        Any[stats.iter, solver.substats.iter, fx, hx, sqrt_θ, sqrt_ξ_νInv, ktol, τ, norm(x)],
        colsep = 1,
      )

    set_iter!(stats, stats.iter + 1)
    rem_eval = max_eval - neval_obj(nlp)
    set_time!(stats, time() - start_time)
    set_objective!(stats, fx)

    #@. solver.y = solver.subsolver.ψ.q*solver.substats.solver_specific[:sigma]
    #mul!(solver.dual_res, solver.subsolver.ψ.A', solver.y, -one(T), zero(T))
    #@. solver.dual_res += solver.subsolver.∇fk
    isa(solver.subsolver, R2Solver) && set_residuals!(stats, hx, norm(solver.subsolver.s)*solver.substats.solver_specific[:sigma])
    isa(solver.subsolver, R2NSolver) && set_residuals!(stats, hx, norm(solver.subsolver.s1)*solver.substats.solver_specific[:sigma_cauchy])

    set_status!(
      stats,
      get_status(
        nlp,
        elapsed_time = stats.elapsed_time,
        n_iter_since_decrease = n_iter_since_decrease,
        iter = stats.iter,
        optimal = solved,
        max_eval = max_eval,
        max_time = max_time,
        max_iter = max_iter,
        max_decreas_iter = max_decreas_iter,
      ),
    )

    callback(nlp, solver, stats)

    done = stats.status != :unknown
  end

  set_solution!(stats, x)
  #isa(solver.subsolver, R2Solver) && set_constraint_multipliers!(stats, solver.subsolver.s)
  #isa(solver.subsolver, R2NSolver) && set_constraint_multipliers!(stats, solver.subsolver.s1)
  return stats
end

function get_status(
  nlp::M;
  elapsed_time = 0.0,
  iter = 0,
  optimal = false,
  n_iter_since_decrease = 0,
  max_eval = Inf,
  max_time = Inf,
  max_iter = Inf,
  max_decreas_iter = Inf,
) where {M <: AbstractNLPModel}
  if optimal
    :first_order
  elseif iter > max_iter
    :max_iter
  elseif elapsed_time > max_time
    :max_time
  elseif neval_obj(nlp) > max_eval && max_eval > -1
    :max_eval
  elseif n_iter_since_decrease ≥ max_decreas_iter
    :infeasible
  else
    :unknown
  end
end

mutable struct L2_R2N_subsolver{T <: Real, V <: AbstractVector{T}} <: AbstractOptimizationSolver
  u1::V
  u2::V
end

function L2_R2N_subsolver(reg_nlp::AbstractRegularizedNLPModel{T, V};) where {T, V}
  x0 = reg_nlp.model.meta.x0
  n = reg_nlp.model.meta.nvar
  m = length(reg_nlp.h.b)
  #x = zero(x0)
  u1 = similar(x0, n+m)
  u2 = zeros(eltype(x0), n+m)

  return L2_R2N_subsolver(u1, u2)
end

function SolverCore.solve!( # TODO: optimize this code, make it allocation-free + check tolerances
  solver::L2_R2N_subsolver{T, V},
  reg_nlp::AbstractRegularizedNLPModel{T, V},
  stats::GenericExecutionStats{T, V, V};
  x = reg_nlp.model.meta.x0,
  σk = T(1),
  atol = eps(T)^(0.5),
  max_time = T(30),
  max_iter = 100,
) where {T <: Real, V <: AbstractVector{T}}
  start_time = time()
  set_time!(stats, 0.0)
  set_iter!(stats, 0)

  n = reg_nlp.model.meta.nvar
  m = length(reg_nlp.h.b)
  Δ = reg_nlp.h.h.lambda/reg_nlp.model.σ

  u1 = solver.u1
  u2 = solver.u2

  # Create problem
  @. u1[1:n] = reg_nlp.model.∇f/reg_nlp.model.σ # - mν∇fk
  @. u1[(n + 1):(n + m)] = -reg_nlp.h.b

  full_row_rank = reg_nlp.h.full_row_rank
  αₖ = 0.0
  αmin = eps(T)^(0.9)
  θ = 0.8
  Q = reg_nlp.model.B/reg_nlp.model.σ
  writedlm("Q.txt", Matrix(Q))
  atol = eps(T)^0.3

  H = [[-Q-opEye(n, n) reg_nlp.h.A']; [reg_nlp.h.A αₖ*opEye(m, m)]]
  x1, stats_minres = minres_qlp(H, u1)
  
  solved = stats_minres.solved
  #println("-----------")
  #println(norm(x1[(n + 1):end]) <= Δ && full_row_rank && solved)
  #println(norm(x1[(n + 1):end]) <= Δ)
  #println(full_row_rank)
  #println(solved)
  #println("-----------")
  if norm(x1[(n + 1):end]) <= Δ && solved && !stats_minres.inconsistent
    set_solution!(stats, x1[1:n])
    return
  end
  """
  if reg_nlp.h.h.lambda*norm(reg_nlp.h.b) - obj(reg_nlp, x1[1:n]) < 0 && norm(x1[(n + 1):(n + m)]) <= Δ # The problem is not convex, retreat to Cauchy point.
    set_solution!(stats, x)
    return
  end
  """
  happened = false
  res = norm(H*x1-u1)
  #println(res)

  if !full_row_rank && norm(H*x1-u1) ≤ eps(T)^0.4

    # First compute v = -(AQ^-1 d + b). USE GMRES because this matrix is not hermitian...
    H = [[-Q-opEye(n,n) opZeros(n, m)];
         [reg_nlp.h.A opEye(m, m)]]
    x2, stats_gmres = gmres(H, u1)
    u = zeros(T, 2*n + 2*m)
    u[m+2*n+1:end] .= x2[n+1:end]

    H = [[opEye(m,m) LinearOperator(reg_nlp.h.A) opZeros(m, n) opZeros(m, m)]; 
         [LinearOperator(reg_nlp.h.A') opZeros(n, n) Q+opEye(n,n) opZeros(n, m)];
         [opZeros(n,m) Q+opEye(n,n) opZeros(n, n) -LinearOperator(reg_nlp.h.A')];
         [opZeros(m, m) opZeros(m, n) -LinearOperator(reg_nlp.h.A) opZeros(m, m)]]
    x, stats_minres = minres_qlp(H, u, atol = eps(T)^0.4)
    #println(stats_minres)
    #println(norm(H*x-u))
    if norm(x[1:m]) <= Δ + atol && stats_minres.solved && !stats_minres.inconsistent
      # w = Q^{-1}A^T y = x[m+n+1:m+2*n]
      set_solution!(stats, -x[m+n+1:m+2*n]  + x2[1:n])
      #println("YES STILL !!!")
      println(stats_minres)
      return
    end
  end

  if !full_row_rank || !solved
    αₖ = eps(T)^0.2
    H = [[-Q-opEye(n, n) reg_nlp.h.A']; [reg_nlp.h.A αₖ*opEye(m, m)]]
    x1, stats_minres = minres_qlp(H, u1)
    #println(stats_minres)
  end
  u2[(n + 1):(n + m)] .= x1[(n + 1):(n + m)]
  x2, stats_minres = minres_qlp(H, u2)
  #println(stats_minres)
  #println(αₖ + norm(x1[(n + 1):(n + m)])^2/dot(x1[(n + 1):(n + m)], x2[(n + 1):(n + m)])*(norm(x1[(n + 1):(n + m)])/Δ - 1))

  while abs(norm(x1[(n + 1):(n + m)]) - Δ) > atol && stats.iter < max_iter && stats.elapsed_time < max_time
    α₊ = αₖ + norm(x1[(n + 1):(n + m)])^2/dot(x1[(n + 1):(n + m)], x2[(n + 1):(n + m)])*(norm(x1[(n + 1):(n + m)])/Δ - 1)

    αₖ = α₊ ≤ 0 ? θ*αₖ : α₊
    αₖ = αₖ ≤ αmin ? αmin : αₖ

    H = [[-Q-opEye(n, n) reg_nlp.h.A']; [reg_nlp.h.A αₖ*opEye(m, m)]]
    x1, stats_minres = minres_qlp(H, u1)
    u2[(n + 1):(n + m)] .= x1[(n + 1):(n + m)]
    x2, _ = minres_qlp(H, u2)

    set_iter!(stats, stats.iter + 1)
    set_time!(stats, time()-start_time)
    αₖ == αmin && break
  end
  """
  #println(x1[1:n])
  println("---------")
  println(stats.iter)
  println(αₖ)
  println(full_row_rank)
  println("----------")
  #println(norm(x1[(n + 1):(n + m)]))
  #println(Δ)
  #println(x1[1:n])
  #x, _ = minres_qlp(Q+opEye(n,n), (-reg_nlp.model.∇f/reg_nlp.model.σ + reg_nlp.h.A'*x1[n+1:end]))
  #println(x)
  println("sol 1")
  """
  #println(typeof(Q))
  #error("done")
  if stats.iter > 10 
    println(norm(H*x1-u1))
    println(reg_nlp.h.full_row_rank)
  end
  set_solution!(stats, x1[1:n])
end
