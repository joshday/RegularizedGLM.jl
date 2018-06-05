module RegularizedGLM

using Reexport
@reexport using PenaltyFunctions


# linreg: μ = identity,                 μ′ = x -> 1
# logreg: μ = x -> 1 / (1 + exp(-x)),   μ′ = x -> exp(-x) / (1 + exp(-x))^2
# poisreg: μ = exp,                     μ′ = exp

function glmfit(x, y, μ::Function, μ′::Function)
    n, p = size(x)
    β = zeros(p)
    xβ = zeros(n)
    r = zeros(n)
    xtr = zeros(p)
    x2 = vec(sum(abs2, x, 1))
    for iter in 1:100
        for i in 1:n 
            r[i] = y[i] - μ(xβ[i])
        end
        At_mul_B!(xtr, x, r)
        for j in 1:p 
            β[j] = β[j] + p * xtr[j] / x2[j]
        end
        A_mul_B!(xβ, x, β)
    end
    β
end

function logreg(x, y)
    n, p = size(x)
    β = zeros(p)
    buffer_n = zeros(n)
    buffer_p = zeros(p)
    buffer_p2 = zeros(p)
    xtx_inv = 4inv(Symmetric(x'x))
    for iter in 1:50 
        for i in 1:n 
            buffer_n[i] = y[i] - 1 / (1 + exp(-buffer_n[i]))
        end
        At_mul_B!(buffer_p, x, buffer_n)
        A_mul_B!(buffer_p2, xtx_inv, buffer_p)
        β[:] = β + buffer_p2
        A_mul_B!(buffer_n, x, β)
    end
    β
end


# (exp(x) / (1 + exp(x))) / (1 + exp(x))



end # module
