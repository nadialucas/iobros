# Nadia Lucas and Fern Ramoutar
# February 3, 2021
#
# We would like to thank Chase Abram, Jordan Rosenthal-Kay, Jeanne Sorin, 
# George Vojta 
# for helpful comments
# 
# running on Julia version 1.5.2
################################################################################
# ENV["PYTHON"] = "/path/to/python3"
# Pkg.build("PyCall")
# restart Julia
# PyCall.python # should show new path to python packages

using CSV
using DataFrames
using DataFramesMeta
using Random
using Distributions
using LinearAlgebra
using Statistics
using ForwardDiff
using PyCall
using FiniteDiff


current_path = pwd()
if pwd() == "/Users/nadialucas"
    data_path = "/Users/nadialucas/Dropbox/Second year/IO 2/pset1/"
elseif pwd() == "/home/nrlucas"
    data_path = "/home/nrlucas/IO2Data/"
end
main_data = CSV.read(string(data_path, "psetOne.csv"), DataFrame)

## Helper function
# take in dataset and market and return matrix of instruments, Xs, prices, shares, and ownership matrix
function clean_data(df, mkt)
    market = @where(df, :Market .== mkt)
    # creates the ownership matrix
    justbrands = @select(market, :Brand2, :Brand3)
    justbrands = convert(Matrix, justbrands)

    samebrand2 = [x1 == x2  ? 1 : 0 for x1 in justbrands[:,1], x2 in justbrands[:,1]  ]
    samebrand3 = [y1 == y2 ? 1 : 0 for y1 in justbrands[:,2], y2 in justbrands[:,2]]
    O = samebrand2 .* samebrand3

    X_mat = @select(market, :Constant, :EngineSize, :SportsBike, :Brand2, :Brand3)
    X = convert(Matrix, X_mat)
    P_mat = @select(market, :Price)
    P = convert(Matrix, P_mat)
    S_mat = @select(market, :shares)
    S = convert(Matrix, S_mat)
    Z_mat = @select(market, :z1, :z2, :z3, :z4)
    Z = convert(Matrix, Z_mat)
    
    return O, X, P, S, Z
end

# Part 8
O_17, X_17, P_17, S_17, Z_17 = clean_data(main_data, 17)

dist = Normal()
xi = rand(dist, length(X_17[:,1]))
xi = reshape(xi, (length(xi), 1))
X_17_xi = [X_17'; xi']'

# now compute all shares taking price
theta_xi = [-3, 1, 1, 2, -1, 1, 1]
theta = [-3, 1, 1, 2, -1, 1]
beta_xi = theta_xi[2:7]
alpha = -1*theta[1]
beta = theta[2:6]


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

function fp_solver(prices, beta, alpha, xs, ownership_mat)
    tol = 1e-14
    maxiter = 10000
    p = prices
    iter = 0
    diff = 100
    while diff > tol && iter < maxiter
        # I attempted to solve this by hand with the hessian but was pointed to
        # ForwardDiff by a friend
        H = ForwardDiff.jacobian(x -> fixed_pt(x, beta, alpha, xs, ownership_mat), p)
        p_next = p - H \ fixed_pt(p, beta, alpha, xs, ownership_mat)
        diff = maximum(abs.(p.-p_next))
        iter+=1
        p = p_next
    end
    return p
end

homogenous_prices = fp_solver(P_17, beta_xi, alpha, X_17_xi, O_17)
println("Part 8 Solution")
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

#### Problem 9

function sHat(delta, X, sigma, zeta, I, J)
    vec_probs = zeros(J)
    for i in 1:I
        zeta_i = zeta[i,:]
        denominator = 1
        numerator_vec = zeros(J)
        for j in 1:J
            X_j = X[j,:]
            delta_j = delta[j]
            num = exp(delta_j + X_j'* (sigma * zeta_i))
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
function sHat_inverse(s, sigma, X, zeta, I, J)
    tol = 1e-14
    maxiter = 10000
    # initialize delta
    delta = zeros(J)
    diff = 100
    iter = 0
    while diff > tol && iter < maxiter
        shat = sHat(delta, X, sigma, zeta, I, J)
        delta_next = delta .+ log.(s) - log.(shat)
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
delta = [4, 2, 2]
sigma = zeros(numChars, numChars)
X = zeros(J, numChars)
zeta = rand(dist, I, numChars)

shares = sHat(delta, X, sigma, zeta, I, J)
# woooo it works!!
delts = sHat_inverse(shares, sigma, X, zeta, I, J)

# Part 11 The Objective function
# take in a sigma, the data (X, Z, S), the weighting matrix, and zeta draws
# note that for now, just works in one market, could loop over multiple later
function objective(s, X, Z, S, W, zeta, M)
    # sigma has to be a vector input for built-in AD 
    K = size(X,2)
    I = size(zeta, 1)
    J = size(X,1)
    sigma = zeros(K, K)
    sigma[1] = s[1]
    sigma[3,3] = s[2]
    # calculate delta
    deltas = []
    for m in unique(M)
        mask = findall(M .== m)
        X_m = X[mask,:]
        j = size(X_m, 1)
        S_m = S[mask]
        deltas = [deltas;sHat_inverse(S_m, sigma, X_m, zeta, I, j)]
    end

    deltas = reshape(deltas, (length(deltas), 1))

    # calculate theta
    bread = X' * Z * W * Z'
    theta = (bread * X) \ bread * deltas

    # calculate xi
    xi = deltas - X*theta

    result = xi' * Z * W * Z' * xi
    return result[1]
end

# only two of the characteristics have random coefficients
numChars = 6
sigma = .1 * ones(2)
I = 50
zeta = rand(dist, I, numChars)

cov = @select(main_data, :Constant, :EngineSize, :SportsBike, :Brand2, :Brand3)
inst = @select(main_data, :z1, :z2, :z3, :z4)
cov = convert(Matrix, cov)
inst = convert(Matrix, inst)
market = @select(main_data, :Market)
market = convert(Matrix, market)
M = reshape(market, length(market))
prices = @select(main_data, :Price)
prices = convert(Matrix, prices)

shares = @select(main_data, :shares)
S = convert(Matrix, shares)
X = [prices'; cov']'
Z = [cov'; inst']'

W = inv(Z'*Z)
objective(sigma, X, Z, S, W, zeta, M)

#sigma = convert(Array{Real}, sigma)

# Part 12 The gradient


# rework the share equation to get heterogeneous shares (with sigma and zeta)
# and keep it at the individual level to feed into the gradient helper functions
function get_shares_het(theta_bar, X, sigma, zeta)
    I = size(zeta, 1)
    J = size(X, 1)
    vec_shares = zeros(I, J)
    for i in 1:I
        num = exp.(X * theta_bar + X * (sigma * zeta[i,:]))
        vec_shares[i,:] = num./(1+sum(num))
    end
    return vec_shares
end

# this jacobian is individual level, returns J x J x I
# gets collapsed in the other jacobian
# collapsed down this is the full on market-level share jacobian
function jacobian_xi_i(theta_bar, sigma, X, zeta)
    J = size(X, 1)
    I = size(zeta, 1)
    # get the shares from the heterogeneous share equation
    shares = get_shares_het(theta_bar, X, sigma, zeta)
    jac = zeros(J, J, I)
    # for each individual, same kind of computation as homogeneous case
    for i in 1:I
        shares_i = shares[i,:]
        jac[:,:,i] = -shares_i * shares_i'
        for j in 1:J
            jac[j, j, i] = shares_i[j] * (1-shares_i[j])
        end
    end
    return jac
end

# now we collapse all down to a bigger jacobian which will be JxK
# this is from the bottom of page 11 in the notes
function jacobian_theta(theta_bar, sigma, X, zeta)
    I = size(zeta, 1)
    J = size(X, 1)
    K = size(X, 2)
    jac_xi_i = jacobian_xi_i(theta_bar, sigma, X, zeta)
    jac_xi_summed = sum(jac_xi_i, dims = 3)[:,:] 
    jac_xi_all = jac_xi_summed./I
    
    jac_theta = zeros(J, K)
    for i in 1:I
        jac_theta .+= jac_xi_i[:,:,i] * X * Diagonal(zeta[i,:])
    end
    jac_theta = jac_theta./I
    return -inv(jac_xi_all) * jac_theta

end

# combine all the jacobians to get the gradient
function gradient(s, X, Z, S, W, zeta, M)
    sigma = zeros(numChars, numChars)
    sigma[1] = s[1]
    sigma[3,3] = s[2]
    I = size(zeta, 1)
    J = size(X,1)
    K = size(X,2)

    # calculate delta
    deltas = []
    for m in unique(M)
        mask = findall(M .== m)
        X_m = X[mask,:]
        j = size(X_m, 1)
        S_m = S[mask]
        deltas = [deltas;sHat_inverse(S_m, sigma, X_m, zeta, I, j)]
    end

    # calculate theta_bar
    bread = X' * Z * W * Z'
    theta_bar = (bread * X) \ bread * deltas

    
    # calculate xi
    xi = deltas - X*theta_bar


    # get jacobian_theta for each market
    jacobians = zeros(J, K)
    for m in unique(M)
        mask = findall(M .== m)
        X_m = X[mask,:]
        jacobians[mask,:] = jacobian_theta(theta_bar, sigma, X_m, zeta)
    end
    return (2 * jacobians' * Z * W * Z' * xi)

end


sigma = .1 * ones(2)
grad = gradient(sigma, X, Z, S, W, zeta, M)
println(grad)
# the gradient we want is the first and third elements
final_gradient = [grad[1], grad[3]]

# Check with AD
# Forward diff is being finicky
# AD_grad = ForwardDiff.gradient(x -> objective(x, X, Z, S, W, zeta, M), sigma)
AD_finite_grad = Calculus.gradient(x -> objective(x, X, Z, S, W, zeta, M), sigma)
#0.10181718614028794
#0.8471135214997823
# This thing is not really even close lol

# GMM Time



