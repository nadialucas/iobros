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
    Z_mat = @select(market, :z1, :z2, :z3)
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

function sHat(delta, X, sigma, zeta, I, J)
    vec_probs = zeros(J)
    for i in 1:I
        zeta_i = zeta[i,:]
        denominator = 1
        numerator_vec = zeros(J)
        for j in 1:J
            X_j = X[j,:]
            delta_j = delta[j]
            num = exp(delta_j + X_j'* (sigma .* zeta_i))
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
        delta_next = delta + log.(s) - log.(shat)
        diff = maximum(abs.(delta.-delta_next))
        iter+=1
        delta = delta_next
    end
    return delta


end

function objective(sigma, X, Z, S, W, zeta)
    # sigma has to be a vector input for built-in AD 
    I = size(zeta, 1)
    J = size(X,1)
    # calculate delta
    delta = sHat_inverse(S, sigma, X, zeta, I, J)
    

    # calculate theta
    bread = X' * Z * W * Z'
    theta = (bread * X) \ bread * delta

    # calculate xi
    xi = delta - X*theta

    result = (Z'*xi)' * W * Z' * xi
    return result[1]
end

numChars = 6
sigma = zeros(numChars)
sigma[1] = .1
sigma[3] = .1
zeta = rand(dist, I, numChars)

X = [P_17'; X_17']'
Z = [X_17'; Z_17']'
W_17 = inv(Z'*Z)
objective(sigma, X, Z, S_17, W_17, zeta)


#Jeanne's gradient
# Similar to shatJ_f, but does not sum over ζ
# i = 1
# ζi = ζ[i,:]
function shatJ_i_f(δ, Σ, X, ζi)
    NK = size(X)[2]                     # nb of good attributes
    NJ = length(δ)                      # nb of goods
    ## Get proba & share
    prob = IndProba_f(δ, Σ, ζi, X)
    Gradfi = zeros(NJ, NJ)               # nb_goods * nb_goods
    for j = 1:NJ
        for k = 1:NJ
            if j==k
                Gradfi[j,k] = prob[j,:]' * (1 .- prob[j,:])
            else
                Gradfi[j,k] = -1  * prob[k,:]' * prob[j,:]
            end
        end
    end
    # at that point = Gradient f_j^i
    # now needs do get [∂ Pr(i→j)] / [∂ σ] =  [Gradient f_j^i * Xt ; Gradient f_j^i * Xt * diag(ζi)]
    Jacob = Gradfi * X * hcat(ones(NK), Diagonal(ζi))
    return Jacob
end
# Jacobian total share = average over Jacob_i
function sHatJ_f(δ, Σ, X, ζ)
    NI = size(ζ)[1]
    NK = size(X)[2]                     # nb of good attributes
    NJ = length(δ)                      # nb of goods
    Jacob = zeros(NJ, NJ)
    for i = 1:NI
        ζi = ζ[i,:]
        Jacob = Jacob .+ shatJ_i_f(δ, Σ, X, ζi)
    end
    Jacob = Jacob / NI
    return Jacob
end
# Jσ_f = n * 6
function Jσ_f(θbar, Σ, δ, ξ, X, ζ)
    NJ = size(X)[1]
    NI = size(ζ)[1]
    # Compute Jξs_i for each i (jacobian of individual proba)
    Jξs_iall = zeros(size(X)[1], size(X)[1], NI)
    for i in 1:NI
        Jξs_iall[:,:,i] = shatJ_i_f(δ, Σ, X, ζ[i,:])
    end
    # Compute Jξs (jacobian of share equation)
    Jξs = sHatJ_f(δ, Σ, X, ζ)
    # Compute baby
    Jσ = zeros(NJ, NK)
    for i = 1:NI
        Jσ = Jσ .+ Jξs_iall[:,:,i] * X * Diagonal(ζ[i,:])
    end
    Jσ = - inv(Jξs) * Jσ / NI
    return Jσ
end
# The Jacobian / Gradient of the Objective Function G_σ
function demandobjG_F(θbar, σA, σBCC, δ, ξ, X, ζ, W, Z)
    Σ = fullΣ_f(σA, σBCC)
    Jσ = Jσ_f(θbar, Σ, δ, ξ, X, ζ)
    # See slidesJan28
    Grad = 2 .* (Z' * Jσ)' * W * (Z' * ξ)       # (6*1)
    return Grad
end