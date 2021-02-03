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

import Pkg; Pkg.add("KNITRO")
import Pkg; Pkg.add("JuMP")
using JuMP
using KNITRO
using CSV
using DataFrames
using DataFramesMeta
using Random
using Distributions
using LinearAlgebra
using Statistics



current_path = pwd()
if pwd() == "/Users/nadialucas"
    data_path = "/Users/nadialucas/Dropbox/Second year/IO 2/pset1/"
elseif pwd() == "/home/nrlucas"
    data_path = "/home/nrlucas/IO2Data/"
elseif pwd() == "/home/cschneier"
    data_path = "/home/cschneier/IO/"
    out_path = "/home/cschneier/nadia/"
end
main_data = CSV.read("/home/cschneier/IO/psetOne.csv", DataFrame)



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
    total_grad = (2 * jacobians' * Z * W * Z' * xi)
    return [total_grad[1], total_grad[3]]

end

dist = Normal()
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
#objective(sigma, X, Z, S, W, zeta, M)



sigma = .1 * ones(2)
#grad = gradient(sigma, X, Z, S, W, zeta, M)
#println(grad)
# the gradient we want is the first and third elements
#final_gradient = [grad[1], grad[3]]

# This thing is not really even close lol

# GMM Time

# BABY MARKET for testing
global X_10 = X[1:100,:]
global Z_10 = Z[1:100,:]
global W_10 = inv(Z'*Z)
global M_10 = M[1:100,:]
global S_10 = S[1:100,:]
global numChars = 6
sigma = .1 * ones(2)
global I = 10
global zeta_10 = rand(dist, I, numChars)

function knitro_objective(sig)
    return objective(sig, X_10, Z_10, S_10, W_10, zeta_10, M_10)
end
function knitro_gradient(sig)
    return gradient(sig, X_10, Z_10, S_10, W_10, zeta_10, M_10)
end

function callbackEvalF(kc, cb, evalRequest, evalResult, userParams)
    x = evalRequest.x
    evalResult.obj[1] = knitro_objective(x)
    return 0
end

function callbackEvalG!(kc, cb, evalRequest, evalResult, userParams)
    x = evalRequest.x
    grad = knitro_gradient(x)
    # Evaluate gradient of nonlinear objective
    for i in 1:length(grad)
      evalResult.objGrad[i] = grad[i]
    end
    return 0
end




kc = KNITRO.KN_new()
KNITRO.KN_add_vars(kc, n)
KNITRO.KN_set_var_lobnds(kc, x_L)
KNITRO.KN_set_var_upbnds(kc, x_U)
KNITRO.KN_set_var_primal_init_values(kc, sigma )
KNITRO.KN_set_obj_goal(kc, KNITRO.KN_OBJGOAL_MINIMIZE)

cb = KNITRO.KN_add_objective_callback(kc, callbackEvalF)
KNITRO.KN_set_cb_grad(kc, cb, callbackEvalG!)
nStatus = KNITRO.KN_solve(kc)
nStatus, objSol, x, lambda_ = KNITRO.KN_get_solution(kc)

