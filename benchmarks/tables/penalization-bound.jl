include("regulopt-tables.jl")
using OptimizationProblems, ADNLPModels, OptimizationProblems.ADNLPProblems
using Plots,Polynomials

function prob_test_1(n::Int)
    function f(x)
      return sum(x)
    end
    x0 = 30*rand(Float64, n)
  
    function c!(cx, x)
      cx[1] = 0.5*norm(x)^2
      return cx
    end
    return ADNLPModels.ADNLPModel!(
      f,
      x0,
      c!,
      zeros(Float64, 1),
      zeros(Float64, 1),
      minimize = true,
      name = "prob_test_1";
    )
  end

function prob_test_2(n::Int;k::Int=2)
  function f(x)
    return sum(x)
  end
  x0 = ones(Float64, n)
  function c!(cx, x)
    for i in 1:n
      cx[i] = (x[i]^k)/k
    end
    return cx
  end
  return ADNLPModels.ADNLPModel!(
      f,
      x0,
      c!,
      zeros(Float64, n),
      zeros(Float64, n),
      minimize = true,
      name = "prob_test_2";
    )
end

function prob_test_3()
  function f(x)
    return x[1]^2 + x[2]^2
  end
  x0 = ones(Float64, 2)
  function c!(cx, x)
    cx[1] = (x[2]-1)^3-x[1]^2
    return cx
  end
  return ADNLPModels.ADNLPModel!(
      f,
      x0,
      c!,
      zeros(Float64, 1),
      zeros(Float64, 1),
      minimize = true,
      name = "prob_test_4";
    )
end


function prob_test_4(α::Float64)
  function f(x)
    return sin(α*x[1])
  end
  x0 = ones(Float64,1)
  function c!(cx,x)
    cx[1] = 0.5*x[1]^2
    return cx
  end
return ADNLPModels.ADNLPModel!(
    f,
    x0,
    c!,
    zeros(Float64, 1),
    zeros(Float64, 1),
    minimize = true,
    name = "prob_test_4";
  )
end

function plot_conv_pen(problem::AbstractNLPModel,α::Float64;k::Int = 11,start::Int = 1)
    tol = [10.0^(-i) for i in start:(start+k)]
    iter = zeros(Int,k+1)

    ν = 1.0
    maxIter = 1000000
    maxIter_inner = 10000000
    verbose = 10
    for (idx,ϵ) in enumerate(tol)
        @info("Solving for tolerance : $(ϵ)")
        options =
            ROSolverOptions(ν = ν, ϵa = ϵ, ϵr = 0.0, verbose = verbose, maxIter = maxIter, spectral = true)
        options2 = ROSolverOptions(spectral = false, psb = true,verbose = 10, ϵa = ϵ, ϵr = ϵ, maxIter = maxIter_inner)
        solver_out = Penalization(problem,options,subsolver_options = options2,Lg = α^2,Lj = 1.0)
        iter[idx] = sum(solver_out.solver_specific[:SubsolverCounter])
    end
    slope_2 = -7
    slope_3 = -8
    slope_4 = -9
    slope = round(-log10(iter[length(iter)]/iter[1]),digits = 2)
    plot(tol,iter,label=latexstring("k(ϵ):ϵ^{$(slope)}"),xlabel = L"ϵ",ylabel = L"k")
    plot!(tol,tol.^(slope_2).*(iter[k+1]*tol[k+1]^(-slope_2)),linestyle = :dash,label = L"ϵ^{-7}")
    plot!(tol,tol.^(slope_3).*(iter[k+1]*tol[k+1]^(-slope_3)),linestyle = :dash,label = L"ϵ^{-8}")
    plot!(tol,tol.^(slope_4).*(iter[k+1]*tol[k+1]^(-slope_4)),linestyle = :dash,label = L"ϵ^{-9}")

    plot!(xscale=:log10,yscale =:log10)

end


α = 1e5

plot_conv_pen(prob_test_4(α),α;k=1,start = 0)