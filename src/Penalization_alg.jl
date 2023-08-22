export Penalization

## Minimize 0.5*uQu - d*u + λ||Au+b||
function QuadraticSolver(
  f::F,
  ∇f!::G,
  ψ::S,
  options::ROSolverOptions{R},
  x0::AbstractVector{R};
  selected::AbstractVector{<:Integer} = 1:length(x0),
  Bk_inv = nothing,
  d::AbstractVector{R} = zeros(length(x0)),
  kwargs...
) where {F <: Function, G <: Function, S, R <: Real}
  if Bk_inv === nothing
    error("The inverse of the hessian should be provided.")
  end

  α = 0.0
  g = ψ.A*Bk_inv*d + ψ.b
  H = Symmetric(ψ.A*Bk_inv*ψ.A')

  Δ = ψ.h.lambda
  s = zero(g)
  m = length(g)
  max_iter = 10000
  tol = options.ϵa
  k = 0

  try
    C = cholesky(H)
    s .=  C\(-g)
    if norm(s) <= Δ
      return Bk_inv*(d + ψ.A'*s),1
    end

    w = C.L\s
    α += ((norm(s)/norm(w))^2)*(norm(s)-Δ)/Δ

  catch ex 
    if isa(ex,LinearAlgebra.SingularException) || isa(ex,PosDefException)
      α_opt = 10.0*sqrt(tol)
      while α <= 0 
        α_opt /= 10.0
        C = cholesky(H+α_opt*I(m))
        s .=  C\(-g)
        w = C.L\s
        α = α_opt + ((norm(s)/norm(w))^2)*(norm(s)-Δ)/Δ
      end
    else  
      rethrow()
    end

  end
  
  # Cf Algorithm 7.3.1 in Conn-Gould-Toint
  while abs(norm(s)-Δ)>tol && k < max_iter

    k = k + 1 

    C = cholesky(H+α*I(m))
    s .=  C\(-g)
    w = C.L\s

    αn = ((norm(s)/norm(w))^2)*(norm(s)-Δ)/Δ
    α += αn

  end

  if k > max_iter && abs(norm(s)-Δ)>sqrt(tol)
    error("QuadraticSolver : Newton Method did not converge.")
  end 

  return Bk_inv*(d + ψ.A'*s),1
end


function Penalization(nlp::AbstractNLPModel, args...; kwargs...)
  kwargs_dict = Dict(kwargs...)
  x0 = pop!(kwargs_dict, :x0, nlp.meta.x0)
  
  function J!(nlp::AbstractNLPModel,z::SparseMatrixCSC{Float64},x::Vector{Float64})
    (rows,cols) = jac_structure(nlp)
    z .= sparse(rows,cols,jac_coord(nlp,x))
  end
  if isa(nlp,QuasiNewtonModel)
    xk, k, outdict = Penalization(
      x -> obj(nlp, x),
      (g, x) -> grad!(nlp, x, g),
      (c, x) -> cons!(nlp, x, c),
      (j, x) -> J!(nlp, j, x),
      nlp.meta.ncon,
      subsolver = R2N,
      args...,
      x0;
      B = x -> hess_op(nlp,x),
      B_inv =  InverseLBFGSOperator(length(x0)),
      kwargs_dict...,
    )
  else
    xk, k, outdict = Penalization(
      x -> obj(nlp, x),
      (g, x) -> grad!(nlp, x, g),
      (c, x) -> cons!(nlp, x, c),
      (j, x) -> J!(nlp, j, x),
      nlp.meta.ncon,
      args...,
      x0;
      kwargs_dict...,
    )
  end

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
  B = nothing,
  B_inv = nothing,
  Lj = nothing,
  Lg = nothing,
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
  ck = c!(z,xk[selected])
  hk = h_norm(ck)

  τk = 100.0 + hk 
  #τk = 0.05
  nu_opt = 1.0
  β = 100.0
  #β = 1.0

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
  J!(Jk,xk)
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

    subsolver_options.ν = (1-subsolver_options.η2)/(2*subsolver_options.γ)
    subsolver_options.ϵa = k == 1 ? 1.0e-1 : max(ϵ_subsolver, min(1.0e-1, ξ1 / 10))
    if Lj !== nothing && Lg !== nothing
      κm = (τk*Lj+Lg)/2
      subsolver_options.σmin = max(1/subsolver_options.ν,subsolver_options.γ*2*κm/(1-subsolver_options.η2))
    end

    if subsolver == R2N
      s, iter, _ = with_logger(subsolver_logger) do
        subsolver(f, ∇f!,B, CompositeNormL2(h,c!,J!,Jk,z), subsolver_options, xk,B_inv = B_inv,subsolver = QuadraticSolver)
      end
    else
      s, iter, _ = with_logger(subsolver_logger) do
        subsolver(f, ∇f!, CompositeNormL2(h,c!,J!,Jk,z), subsolver_options, xk)
      end
    end

    Complex_hist[k] = iter 

    xk .=  s
    fk = f(xk)
    ck = c!(z,xk[selected])
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

