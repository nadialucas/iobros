# Nadia Lucas and Fern Ramoutar
# February 26, 2021
#
# We would like to thank Jordan Rosenthal-Kay, Jeanne Sorin, and George Vojta for helpful comments
# 
# running on Julia version 1.5.2
################################################################################
# Parts 2

# load packages
using CSV
using DataFrames
using DataFramesMeta
using Query
using LinearAlgebra
using StatsBase
using Optim
using ForwardDiff

# get mileage data
current_path = pwd()
if pwd()[1:17] == "/Users/nadialucas"
    data_path = "/Users/nadialucas/Documents/iobros/pset2/"
end
raw_data = CSV.read(string(data_path, "psetTwo.csv"), DataFrame)

################################################################################
###### 2.3 Questions
###### 1. Recover engine replacement and clean mileage data
################################################################################

# set up the number of bins of the state space
K = 10
mileage = raw_data[!,1]

# how to recover engine replacement - mileage will decrease suddenly

# first do some data manipulation to get all the mileage lags,
# differences, replacement times, and then bin everything
# accordingly
# first get lags 
data_size = size(raw_data)[1]
col2 = zeros(data_size)
for i in 1:data_size 
    if i == 1
        # we pretend the first observation was after driving
        col2[i] = 0
    else
        col2[i] = raw_data[i-1, 1]
    end
end
main_data = insertcols!(raw_data, 2, :mileage_lag => col2)
# using query to get at mileage intervals
main_data = @from i in main_data begin
    @select {
        i.milage, i.mileage_lag,
        replace = i.milage < i.mileage_lag ? 1 : 0,
        interval = i.milage < i.mileage_lag ? i.milage : i.milage - i.mileage_lag
    } 
    @collect DataFrame
end
# get the maximum step size
max_interval = maximum(filter(!isnan, main_data[!,4]))
# use this to get the proper bins
bin_size = maximum((mileage .+ max_interval) ./ K)
# now get a binned lag and bin intervals using query
main_data = @transform(main_data, mileage_binned = ceil.(Integer, (:milage .+ .01) ./ bin_size))
main_data = @transform(main_data, mileage_binned_lag = ceil.(Integer, (:mileage_lag .+ .01) ./ bin_size))

main_data = @from i in main_data begin
    @select {
        i.milage, i.mileage_lag, i.replace, i.interval, i.mileage_binned, i.mileage_binned_lag,
        new_binned_mileage_lag = i.milage < i.mileage_lag ? 0 : i.mileage_binned_lag
    } 
    @collect DataFrame
end
# this should be the cleaned data for use on the rest of the problem set
main_data = @transform(main_data, binned_difference = :mileage_binned - :new_binned_mileage_lag)

################################################################################
###### 2.3 Questions
###### 3. Discretized domain and estimate Markov Transition probability
################################################################################

# get the columns that we want to be easily accessible
binned_miles = main_data[!,5]
mile_intervals = main_data[!, 4]


mile_intervals = mile_intervals[mile_intervals.>=0]
# initialize the probability distribution, g
g = countmap(mile_intervals)
# get at the maximum state space reached
max_miles = Int(maximum(keys(g)))
# create a dictionary for easy accessibility
g_dict = Dict()
for i in 0:max_miles
    g_dict[i] = g[i]/data_size
end
#get array from dict
g_array = [g_dict[i] for i in 0:max_miles]
# max number of miles in state space
K_miles = maximum(main_data[!,1]) + Integer(max_interval) + 1

# create the transition matrix from raw data first
# initializze transition probabilities
F0_raw = zeros(K_miles,K_miles)
F1_raw = zeros(K_miles,K_miles)
# construct both transition probabilities
for i in 1:K_miles
    if i < K_miles - max_miles
        F0_raw[i, i:i+max_miles] .= g_array
    else
        F0_raw[i, i:K_miles] .= g_array[1:K_miles-i+1]
    end
    F1_raw[i, 1:max_miles+1] .= g_array
end

# then bin the transition probabilities
F0 = zeros(K, K)
xk_map = Dict()
for i in 1:K_miles
    # first figure out what Xs are bins
    xk_map[i] = ceil(Integer, i / bin_size)
end
# use the map between miles and bins to aggregate up
for i in 1:K_miles-1
    for j in 1:K_miles-1
        F0[xk_map[i], xk_map[j]] += (F0_raw[i,j]/bin_size)
    end
end

# correction for the bottom right corner of the transition probability matrix
for i in 1:K
    summed = sum(F0[i,:])
    if summed > 1e-10
        F0[i, K] += (1-summed)
    end
end
# transpose the matrix so we can left-multiply to match syntax on the slides
F0 = F0'

################################################################################
###### 2.3.1 Nested Fixed Point
###### 1,2,3 Working up to the Rust poly-algorithm
################################################################################

# set up Zurcher's utility to handle vector inputs
function utility_vec(d, x, θ)
    if d == 0
        utility = -θ[1] .* x .- (θ[2] .* (x./100).^2)
    elseif d == 1
        utility = -θ[3] .+ zeros(length(x))
    end
    return utility
end

# the fixed point equation as derived in the writeup
function gamma(EV, x, θ, F0, β=.999)
    EV0 = EV[1]
    V0 =  utility_vec(0,x,θ) .+ β .* EV 
    V1 = utility_vec(1,x, θ) .+ β .* EV0 
    # to avoid numerical instability from TA session slide 43
    Vmax = [max(V0[i], V1[i]) for i in 1:length(V0)]
    return F0 * ( Vmax .+ log.( exp.( V0 .- Vmax ) .+  exp.( V1 .- Vmax ) ) )
end
# we want to evaluate utility right at the midpoint of each bin
states = [1:K;] .* bin_size .- (bin_size ./ 2)

# from slide 48, pk is probability of NOT replacing the engine
function get_pks(EV, x, θ, β=.999)
    EV0 = EV[1]
    V0 = utility_vec(0, x, θ) .+ β .* EV
    V1 = utility_vec(1, x, θ) .+ β .* EV0
    denominator = 1 .+ exp.(V1 .- V0)
    return 1 ./ denominator
end

EV1 = [1:K;]
θ1 = [.1, .1, .1]
hi = get_pks(EV1, states, θ1)

# Jacobian of gamma
# functional form on slide 47
function gamma_prime(EV, x, θ, F0, β = .999)
    EV0 = EV[1]
    u0 = utility_vec(0, x, θ)
    u1 = utility_vec(1, x, θ)
    pks = get_pks(EV, x, θ)
    pks_diag = Diagonal(pks)
    EV0term = zeros(K, K)
    EV0term[:,1] = 1 .- pks
    return β .* F0 * (pks_diag + EV0term)
end
# they are identical!!
J_test = ForwardDiff.jacobian(x -> gamma(x, states, θ1, F0), EV1)
J_gamma = gamma_prime(EV1, states, θ1, F0)

function gamma_prime(EV, x, θ, F0, β = .999)
    return ForwardDiff.jacobian(x -> gamma(x, states, θ, F0), EV)
end


function newton_kantarovich(EV_start, x, θ, F0, tol = 1e-14, maxiter = 10000) 
    EV_old = EV_start
    # initialize delta
    diff = 100
    tol_old = 100
    iter = 0
    # run the contraction
    x_k = EV_old
    y_k = EV_old
    while diff > tol && iter < maxiter
        # employ the King-Warner method
        midpoint = (x_k .+ y_k) ./ 2
        Gamma_prime = gamma_prime(midpoint, x, θ, F0)
        Gammak = gamma(x_k, x, θ, F0)
        xk1 = x_k - (I - Gamma_prime) \ (x_k - Gammak)
        Gammak1 = gamma(xk1, x, θ, F0)
        yk1 = xk1 - (I - Gamma_prime) \ (xk1 - Gammak)
        diff = maximum(abs.(xk1 - x_k))
        iter += 1
        x_k = xk1
        y_k = yk1

    end
    #return EV_old
    return y_k
end

function solve_fixed_pt(EV_start, x, θ, F0, β = .999, tol = 1e-14, maxiter = 1000)
    EV_old = EV_start
    # initialize delta
    diff_old = 100
    diff_ratio = 100
    iter = 0
    ID = Matrix{Float64}(I, K, K)
    # run the contraction
    while abs(diff_ratio - β)>tol && iter < maxiter
        EV_new = gamma(EV_old, x, θ, F0)
        diff_new = maximum(abs.(EV_new.-EV_old))
        EV_old = EV_new
        diff_ratio = diff_new/diff_old
        diff_old = diff_new
        iter += 1
    end
    final_EV = newton_kantarovich(EV_old, x, θ, F0)
end

@time EV = solve_fixed_pt(EV1, states, θ1, F0)

################################################################################
###### 2.3.1 Nested Fixed Point
###### 4. Likelihood function for any θ
################################################################################

function likelihood(θ, d, x, states, β=.999) 
    EV1 = zeros(K)
    EV = solve_fixed_pt(EV1, states, θ, F0)
    pk = get_pks(EV, states, θ)
    likelihood = 0
    for i in x
        pxt = pk[i]
        dt = d[i]
        likelihood += dt * log(1-pxt) + (1-dt) * log(pxt)
    end
    return -likelihood
end

miles = main_data[!,1]
x = binned_miles
d = main_data[!,3]

hi = likelihood(θ1, d, x, states)

function f(θ)
    return likelihood(θ, d, x, states)
end

# this returns something!!
#@time opt = optimize(f, θ1, LBFGS())
#params = Optim.minimizer(opt)


################################################################################
###### 2.3.1 Nested Fixed Point
###### 4. Likelihood gradient function
################################################################################

function du0(θ, x)
    return [-x -(x./100).^2 zeros(length(x))]
end

function du1(θ, x)
    return [zeros(length(x)) zeros(length(x)) -ones(length(x))]
end

# get ∂Γ/∂θ for ∂EV/∂θ, derivation in writeup
function dG(EV, θ, x, F0, β=.999)
    # Kx1
    u0 = utility_vec(0, x, θ)
    u1 = utility_vec(1, x, θ)
    V0 = u0 .+ β .* EV
    V1 = u1 .+ β .* EV[1]
    # Kx3
    du1_dtheta = du1(θ, x)
    du0_dtheta = du0(θ, x)
    # this should give a Kx3
    numerator = exp.(V1) .* du1_dtheta .+ exp.(V0) .* du0_dtheta
    # this is Kx1
    denominator = exp.(V1) .+ exp.(V0)
    # should return a Kx3
    return F0 * (numerator ./ denominator)
end

# checks out - they match! uncomment to verify
#my_dG = dG(EV1, θ1, states, F0)
#check_dG = ForwardDiff.jacobian(y -> gamma(EV1, states, y, F0), θ1)

# get ∂EV/∂θ for ∂p_k/∂θ, derivation in the writeup
function dEV(EV, θ, x)
    # KxK
    Gprime = gamma_prime(EV, x, θ, F0)
    # Kx3
    dG_dtheta = dG(EV, θ, x, F0)
    # Kx3
    return (I - Gprime) \ dG_dtheta
end

# get ∂p_k/∂θ, derivation in writeup
function dpk(EV, states, θ, β = 0.999)
    # these should be Kx3
    pk = get_pks(EV, states, θ)
    du1_dtheta = du1(θ, states)
    du0_dtheta = du0(θ, states)
    dEV_dtheta = dEV(EV, θ, states)
    dEV0 = repeat(dEV_dtheta[1,:]', outer = [length(states),1])
    dV1 = du1_dtheta .+ β .* dEV0
    dV0 = du0_dtheta .+ β .* dEV_dtheta
    # should be Kx3
    return -pk .* (1 .- pk) .* (dV1 .- dV0)
end

# matches! uncomment to verify
#my_dpk = dpk(EV1, states, θ1)
#check_dpk = ForwardDiff.jacobian(y -> get_pks(EV1, states, y), θ1)

β = .999

# using ∂p_k/∂θ, we can put together the likelihood gradient
# summing over the observed data
function likelihood_grad(θ)
    EV1 = [1:K;]
    EV = solve_fixed_pt(EV1, states, θ, F0)
    pk = get_pks(EV, states, θ)
    # this should return a Kx3
    dpk_dtheta = dpk(EV, states, θ)
    # get a Tx1
    pxt_vec = [pk[i,:][1] for i in x]
    # Tx3
    dpxt_vec = [dpk_dtheta[i,j] for i in x for j in 1:length(θ1)]
    dpxt_vec = reshape(dpxt_vec, (length(θ1), length(x)))
    dpxt_vec = dpxt_vec'
    # check dimensions to make sure this returns a 3x1
    result_vec = -((d ./ (1 .- pxt_vec)) .* (dpxt_vec) .+ ((1 .- d) ./ pxt_vec) .* dpxt_vec)
    result = sum(result_vec, dims = 1)
    return result[:]
end

# they match!! uncomment to verify
my_dldtheta = likelihood_grad(θ1)
check_dldtheta = ForwardDiff.gradient(y -> likelihood(y, d, x, states), θ1)


function gg!(storage, θ)
    hi = likelihood_grad(θ)
    storage .= hi
end

# oh man why is this not working argh
@time final_opt = optimize(f, gg!, θ1, BFGS())
theta = Optim.minimizer(final_opt)
