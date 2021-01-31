# Nadia Lucas and Fern Ramoutar
# February 3, 2021
#
# We would like to thank Chase Abram, Jordan Rosenthal-Kay, and Jeanne Sorin 
# for helpful comments
# 
# running on Julia version 1.5.2
################################################################################

using CSV
using DataFrames
using DataFramesMeta
using Random
using Distributions
using NLsolve
using LinearAlgebra
using Statistics
using Optim
using ForwardDiff


path = "/Users/nadialucas/Dropbox/Second year/IO 2/pset1/"
data = CSV.read(string(path, "psetOne.csv"), DataFrame)

# Part 8: Market t=17 (data setup is a bit janky for now)
data = @where(data, :Market .== 17)

dist = Normal()
data = @transform(data, xi = rand(dist, nrow(data)))

# creates the ownership matrix
justbrands = @select(data, :Brand2, :Brand3)
justbrands = convert(Matrix, justbrands)

samebrand2 = [x1 == x2  ? 1 : 0 for x1 in justbrands[:,1], x2 in justbrands[:,1]  ]
samebrand3 = [y1 == y2 ? 1 : 0 for y1 in justbrands[:,2], y2 in justbrands[:,2]]
ownership_mat = samebrand2 .* samebrand3

# now compute all shares taking price
theta = [-3, 1, 1, 2, -1, 1, 1]


data_mat = @select(data, :Constant, :EngineSize, :SportsBike, :Brand2, :Brand3, :xi)
prices_mat = @select(data, :Price)
prices = convert(Matrix, prices_mat)
xs = convert(Matrix, data_mat)

beta = theta[2:7]
alpha = -1*theta[1]


function get_shares(p, beta, alpha, xs)
    s = exp.(-alpha*p + xs*beta)
    D = s./(1+sum(s))
    return D
end

function get_jacobian(p, beta, alpha, xs)
    D = get_shares(p, beta, alpha, xs)
    J = alpha .* D * D'
    for i in eachindex(p)
        J[i,i] = -alpha * D[i] * (1-D[i])
    end
    return J
end

function fixed_pt(p, beta, alpha, xs, ownership_mat)
    D = get_shares(p, beta, alpha, xs)
    J = get_jacobian(p, beta, alpha, xs)
    O = ownership_mat.*J
    return -p - O\D
end

function fp_solver(prices, beta, alpha, xs, ownership_mat, tol = 1e-14, maxiter = 10000)
    p = prices
    iter = 0
    diff = 100

    while diff > tol && iter < maxiter
        H = ForwardDiff.jacobian(x -> fixed_pt(x, beta, alpha, xs, ownership_mat), p)
        p_next = p - H \ fp(p, beta, alpha, xs, ownership_mat)
        diff = maximum(abs.(p.-p_next))
        iter+=1
        p = p_next
    end
    return p
end

homogenous_prices = fp_solver(prices, beta, alpha, xs, ownership_mat)
println(homogenous_prices)

# we might need this later, who knows?

function hessian(p, beta, alpha, xs)
    H = zeros(length(p), length(p), length(p))
    D, J = shares_jacobian(p, beta, alpha, xs)
    for i in eachindex(p)
        for j in eachindex(p)
            for k in eachindex(p)
                if i==j | j==k
                    H[i,j,k] = alpha * J[i,i] * (2*D[i].-1)
                else
                    H[i,j,k] = 2* alpha * J[i,k] * D[j]
                end
            end
        end
    end
    return(H)
end

#### Time for problem 9 (CURRENTLY NOT WORKING, still unsure about the prob setup)

function get_numerator(delta_j, sigma, X_j, zeta_i)
    return exp.(delta_j+ sigma*X_j*zeta_i)
end

function sHat(delta, X, sigma, zeta, I, J)
    for i in 1:I
        zeta_i = zeta[i,:]
        denominator = 1
        numerator_vec = zeros(J)
        for j in 1:J
            X_j = X[j,:]
            delta_j = delta[j,:]
            num = get_numerator(delta_j, sigma, X_j, zeta_i)
            denominator +=1





        end
    end
end


