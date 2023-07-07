export Penalization

function Penalization(nlp::AbstractNLPModel, args...; kwargs...)
  kwargs_dict = Dict(kwargs...)
  x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
  function J!(nlp::AbstractNLPModel,x::Vector{Float64},z::SparseMatrixCSC{Float64})
    (rows,cols) = jac_structure(nlp)
    z .= sparse(rows,cols,jac_coord(nlp,x))
  end
  xk, k, outdict = Penalization(
    x -> obj(nlp, x),
    (g, x) -> grad!(nlp, x, g),
    (x, c) -> cons!(nlp, x, c),
    (x, j) -> J!(nlp, x, j),
    nlp.meta.ncon,
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
  return stats
end

function Penalization(
  f::F,
  ∇f!::G,
  c!::C,
  J!::Jac,
  m::Int,
  options::ROSolverOptions{R},
  x0::AbstractVector{R};
  subsolver_logger::Logging.AbstractLogger = Logging.NullLogger(),
  subsolver = R2,
  subsolver_options = ROSolverOptions(ϵa = options.ϵa),
  selected::AbstractVector{<:Integer} = 1:length(x0),
  kwargs...,
) where {F <: Function, G <: Function,C<: Function, Jac <: Function, R <: Real}
  start_time = time()
  elapsed_time = 0.0
  ϵ = options.ϵa
  ϵ_subsolver = subsolver_options.ϵa
  ϵ_subsolver_init = subsolver_options.ϵa
  ϵ_subsolver = copy(ϵ_subsolver_init)
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
  h_norm = eval(:NormL2)(1.0)
  xk = copy(x0)
  z = zeros(eltype(xk),m)
  ck = c!(xk[selected],z)
  hk = h_norm(ck)

  τk = 100.0 + hk 
  nu_opt = 1.0
  β = 100.0

  h = eval(:NormL2)(τk)


  Fobj_hist = zeros(maxIter)
  Hobj_hist = zeros(maxIter)
  Complex_hist = zeros(Int, maxIter)
  if verbose > 0
    #! format: off
    @info @sprintf "%6s %6s %8s %8s %7s %8s %7s" "iter" "sub-iter" "f(x)" "h(x)" "√ξ" "τ" "‖x‖"
    #! format: off
  end

  local ξ1
  k = 0
  σk = max(1 / ν, σmin)
  ν = 1 / σk

  fk = f(xk)
  ∇fk = similar(xk)
  ∇f!(∇fk, xk)
  s = zero(xk)
  Jk = spzeros(eltype(xk),m,length(xk))
  J!(xk,Jk)
  optimal = false
  tired = maxIter > 0 && k ≥ maxIter || elapsed_time > maxTime
    
  while !(optimal || tired)

    k = k + 1
    elapsed_time = time() - start_time
    Fobj_hist[k] = fk
    Hobj_hist[k] = hk

    # define model
    ψ = shifted(h_norm,c!,J!,Jk,z,xk)

    mk(d) = ψ(d)
    #display(Array(ψ.A))
    prox!(s, ψ, zero(xk), nu_opt)
    Complex_hist[k] += 1
    mks = mk(s)
    ξ1 = hk - mks
    ξ1 ≥ 0 || error("Penalization: prox-gradient step should produce a decrease but ξ = $(ξ1)")

    if ξ1 ≥ 0 && k == 1
      ϵ_increment = ϵr * sqrt(ξ1)
      ϵ += ϵ_increment  # make stopping test absolute and relative
      ϵ_subsolver += ϵ_increment
    end
    if sqrt(ξ1) < ϵ && k > 1
      # the current xk is approximately first-order stationary
      optimal = true
      continue
    end

    subsolver_options.ϵa = 1e-6
    subsolver_options.ν = (1-subsolver_options.η2)/(2*subsolver_options.γ)

    @debug "setting inner stopping tolerance to" subsolver_options.optTol
    s, iter, _ = with_logger(subsolver_logger) do
      subsolver(f, ∇f!, shifted(h,c!,J!,Jk,z), subsolver_options, xk)
    end
    # restore initial subsolver_options.ϵa here so that subsolver_options.ϵa
    # is not modified if there is an error
    subsolver_options.ϵa = ϵ_subsolver_init

    Complex_hist[k] = iter 

    xk .=  s
    fk = f(xk)
    ck = c!(xk[selected],z)
    hk = h_norm(ck)
    hk == -Inf && error("nonsmooth term is not proper")

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
      @info "Penalization: √ξ = $(sqrt(ξ1)) for initial point"
    elseif optimal
      #! format: off
      @info @sprintf "%6d %6d %8.1e %8.1e %7.1e %7.1e %7.1e" k iter fk hk sqrt(ξ1) τk norm(xk)
      #! format: on
      @info "Penalization: terminating with √ξ = $(sqrt(ξ1))"
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
    :NonSmooth => h,
    :status => status,
    :fk => fk,
    :hk => hk,
    :ξ => ξ1,
    :elapsed_time => elapsed_time,
  )

  return xk, k, outdict
end

