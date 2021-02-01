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
using LinearAlgebra
using Statistics
using ForwardDiff


current_path = pwd()
if pwd() == "/Users/nadialucas"
    data_path = "/Users/nadialucas/Dropbox/Second year/IO 2/pset1/"
elseif pwd() == "/home/nrlucas"
    data_path = "/home/nrlucas/IO2Data/"
end
data = CSV.read(string(data_path, "psetOne.csv"), DataFrame)

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
        # I attempted to solve this by hand with the hessian but was pointed to
        # ForwardDiff by a friend
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
    vec_probs = zeros(J)
    for i in 1:I
        zeta_i = zeta[i,:]
        denominator = 1
        numerator_vec = zeros(J)
        for j in 1:J
            X_j = X[j,:]
            delta_j = delta[j]
            num = exp(delta_j + X_j'* sigma * zeta_i)
            denominator += num
            numerator_vec[j] = num
        end
        numerator_vec = numerator_vec./denominator
        vec_probs .+= numerator_vec
    end
    return vec_probs ./ I
end


J = 3
I = 20
numChars = 4
delta = zeros(J)
sigma = zeros(numChars, numChars)
X = zeros(J, numChars)
zeta = rand(dist, I, numChars)

hi = sHat(delta, X, sigma, zeta, I, J)

J = 3
I = 20
numChars = 4
delta = zeros(J)
delta[1] = 40
delta[J] = 20
sigma = zeros(numChars, numChars)
X = zeros(J, numChars)
zeta = rand(dist, I, numChars)

hello = sHat(delta, X, sigma, zeta, I, J)

J = 3
I = 20
numChars = 4
delta = zeros(J)
sigma = .1 * (zeros(numChars, numChars) + Diagonal(ones(numChars)))
X = zeros(J, numChars)
zeta = rand(dist, I, numChars)

hey = sHat(delta, X, sigma, zeta, I, J)

### Problem 10: Inverse share function (this is from contraction mapping)
function sHat_inverse(s, sigma, X, zeta, I, J, tol = 1e-14, maxiter = 10000)
    # initialize delta
    delta = zeros(J)
    diff = 100
    iter = 0
    while diff > tol && iter < maxiter
        shat = sHat(delta, X, sigma, zeta, I, J)
        delta_next = delta + log.(s) - log.(shat)
        diff = maximum(abs.(delta.-delta_next))
        iter+=1
        delta = delta_next
    end
    return delta


end

J = 3
I = 20
numChars = 4
delta = ones(J)
delta[1] = 1.01
sigma = zeros(numChars, numChars)
X = zeros(J, numChars)
zeta = rand(dist, I, numChars)

shares = sHat(delta, X, sigma, zeta, I, J)
# woooo it works!!
delts = sHat_inverse(shares, sigma, X, zeta, I, J)


# Part 12



