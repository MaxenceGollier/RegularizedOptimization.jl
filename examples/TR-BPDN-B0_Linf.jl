using Random, LinearAlgebra, TRNC 
using ProximalOperators, ShiftedProximalOperators 
using NLPModels, NLPModelsModifiers, ADNLPModels

# min_x 1/2||Ax - b||^2 + λ||x||₀; ΔB_∞
function B0Binf(compound=1)
  m, n = compound * 200, compound * 512 # if you want to rapidly change problem size 
  k = compound * 10 # 10 signals 
  α = .01 # noise level 

  # start bpdn stuff 
  x0 = zeros(n)
  p   = randperm(n)[1:k]
  x0 = zeros(n, )
  x0[p[1:k]] = sign.(randn(k)) # create sparse signal 

  A, _ = qr(randn(n, m))
  B = Array(A)'
  A = Array(B)

  b0 = A * x0
  b = b0 + α * randn(m, )

  λ = 1.0 
  # put in your initial guesses
  xi = zeros(size(x0))
  function grad!(g, x)
    g .= A'*(A*x - b)
    return g
  end
  ϕ = LSR1Model(SmoothObj((x) -> .5*norm(A*x - b)^2, grad!, xi))

  h = IndBallL0(k)

  ϵ = 1e-6
  ν = opnorm(A)^2
  χ = NormLinf(1.0)
  parameters = TRNCoptions(; ν = ν, β = 1e16, ϵ=ϵ, verbose = 10)

  # input initial guess, parameters, options 
  xtr, ktr, Fhisttr, Hhisttr, Comp_pgtr = TR(ϕ, h, χ, parameters; s_alg = QRalg)


  # input initial guess, parameters, options 
  paramsQR = TRNCoptions(; ϵ=ϵ, verbose = 10)
  xi .= 0 

  xqr, kqr, Fhistqr, Hhistqr, Comp_pgqr = QRalg(ϕ, h, paramsQR; x0 = xi)

  @info "TR relative error" norm(xtr - x0) / norm(x0)
  @info "QR relative error" norm(xqr - x0) / norm(x0)
  @info "monotonicity" findall(>(0), diff(Fhisttr + Hhisttr))
end