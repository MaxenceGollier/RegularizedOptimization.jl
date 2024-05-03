export L2Penalty

function L2Penalty(nlp::AbstractNLPModel, args...; kwargs...)
  kwargs_dict = Dict(kwargs...)
  x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
  (rows, cols) = jac_structure(nlp)
  vals = similar(rows,eltype(x0))
  J_coo = SparseMatrixCOO(nlp.meta.ncon,nlp.meta.nvar,rows,cols,vals)
  xk, k, outdict = L2Penalty(
  x -> obj(nlp, x),
  (g, x) -> grad!(nlp, x, g),
  (c, x) -> cons!(nlp, x, c),
  (j, x) -> jac_coord!(nlp, x, j.vals),
  J_coo,
  args...,
  x0;
  kwargs_dict...,
  )

  ξ = outdict[:ξ]
  stats = GenericExecutionStats(nlp)
  set_status!(stats, outdict[:status])
  set_solution!(stats, xk)
  set_objective!(stats, outdict[:fk] + outdict[:hk])
  set_residuals!(stats, zero(eltype(xk)), ξ ≥ 0 ? sqrt(ξ) : ξ)
  set_iter!(stats, k)
  set_time!(stats, outdict[:elapsed_time])
  set_solver_specific!(stats, :Fhist, outdict[:Fhist])
  set_solver_specific!(stats, :Hhist, outdict[:Hhist])
  set_solver_specific!(stats, :NonSmooth, outdict[:NonSmooth])
  set_solver_specific!(stats, :SubsolverCounter, outdict[:Chist])
  set_solver_specific!(stats, :SubHist, outdict[:SubHist])
  return stats
end
  
function L2Penalty(
  f::F,
  ∇f!::G,
  c!::C,
  J!::Jac,
  J_coo::SparseMatrixCOO,
  options::ROSolverOptions{R},
  x0::AbstractVector{R};
  subsolver_logger::Logging.AbstractLogger = Logging.NullLogger(),
  subsolver = R2,
  subsolver_options = ROSolverOptions(ϵa = options.ϵa),
  selected::AbstractVector{<:Integer} = 1:length(x0),
  benchmark = false,
  ϵ_benchm = 0.0,
  kwargs...,
) where {F <: Function, G <: Function,C<: Function, Jac <: Function, R <: Real}
  start_time = time()
  elapsed_time = 0.0
  ϵ = options.ϵa

  ϵr = options.ϵr
  verbose = options.verbose
  maxIter = options.maxIter
  maxTime = options.maxTime
  σmin = options.σmin
  ν = options.ν
  iter = 0

  if verbose == 0
    ptf = Inf
  elseif verbose == 1
    ptf = round(maxIter / 10)
  elseif verbose == 2
    ptf = round(maxIter / 100)
  else
    ptf = 1
  end

  # initialize parameters
  τk = 100.0 
  β = 100.0

  h_norm = NormL2(1.0)
  h = NormL2(τk)

  b = zeros(eltype(x0),J_coo.m)
  c!(b,x0[selected])
  hk = h_norm(b)

  xk = copy(x0)
  ∇fk = similar(xk)
  s = similar(xk)
  fk = f(xk)
  ∇f!(∇fk, xk)
  J!(J_coo,xk)

  ψ = CompositeNormL2(1.0,c!,J!,J_coo,b)

  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Complex_hist = zeros(Int, maxIter)
  Subsolver_hist_ξ = zeros(maxIter)
  if verbose > 0
    #! format: off
    @info @sprintf "%6s %6s %8s %8s %7s %8s %7s" "iter" "sub-iter" "f(x)" "h(x)" "√ξ" "τ" "‖x‖"
    #! format: off
  end

  local ξ1
  k = 0
  σk = max(1 / ν, σmin)
  ν = 1 / σk

  optimal = false
  tired = maxIter > 0 && k ≥ maxIter || elapsed_time > maxTime
  while !(optimal || tired)
    k = k + 1
    elapsed_time = time() - start_time


    # define model
    ψ = shifted(1.0,c!,J!,J_coo,b,xk)

    mk(d) = ψ(d)

    prox!(s, ψ, zero(xk), 1.0)
    Complex_hist[k] += 1
    mks = mk(s)

    hk = h_norm(ψ.b)
    fk = f(xk)
    Fobj_hist[k] = fk
    Hobj_hist[k] = hk

    ξ1 = hk - mks
    ξ1 ≥ 0 || error("L2Penalty: prox-gradient step should produce a decrease but ξ = $(ξ1)")

    if benchmark && hk < ϵ_benchm
      println(hk)
      optimal = true
      continue
    end

    if ξ1 ≥ 0 && k == 1
      ϵ_increment = ϵr * sqrt(ξ1)
      ϵ += ϵ_increment  # make stopping test absolute and relative
    end
    if sqrt(ξ1) < ϵ && k > 1
      # the current xk is approximately first-order stationary
      optimal = true
      continue
    end

    s, iter, stats = with_logger(subsolver_logger) do
      subsolver(f, ∇f!, CompositeNormL2(τk,c!,J!,J_coo,b), subsolver_options, xk;benchmark = benchmark, ϵ_benchm = ϵ_benchm)
    end

    Complex_hist[k] = iter 
    Subsolver_hist_ξ[k+1] = stats[:ξ]

    xk .=  s

    if (verbose > 0) && (k % ptf == 0)
      #! format: off
      @info @sprintf "%6d %6d %8.1e %8.1e %7.1e %8.1e %7.1e" k iter fk hk sqrt(ξ1) τk norm(xk)
      #! format: on
    end

    tired = k ≥ maxIter || elapsed_time > maxTime
    τk = τk + β
    h = eval(:NormL2)(τk)
  end

  if verbose > 0
    if k == 1
      @info @sprintf "%6d %8.1e %8.1e" k fk hk
      @info "L2Penalty: √ξ = $(sqrt(ξ1)) for initial point"
    elseif optimal
      #! format: off
      @info @sprintf "%6d %6d %8.1e %8.1e %7.1e %7.1e %7.1e" k iter fk hk sqrt(ξ1) τk norm(xk)
      #! format: on
      @info "L2Penalty: terminating with √ξ = $(sqrt(ξ1))"
    end
  end
  status = if optimal
    :first_order
  elseif elapsed_time > maxTime
    :max_time
  elseif tired
    :max_iter
  else
    :exception
  end
  outdict = Dict(
    :Fhist => Fobj_hist[1:k],
    :Hhist => Hobj_hist[1:k],
    :Chist => Complex_hist[1:k],
    :SubHist => Subsolver_hist_ξ[1:k],
    :NonSmooth => h,
    :status => status,
    :fk => fk,
    :hk => hk,
    :ξ => ξ1,
    :elapsed_time => elapsed_time,
  )

  return xk, k, outdict
end
