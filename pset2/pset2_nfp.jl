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
elseif pwd() == "/home/nrlucas"
    data_path = "/home/nrlucas/IO2Data/"
end
raw_data = CSV.read(string(data_path, "psetTwo.csv"), DataFrame)

# set up Zurcher's utility
function utility(d, x, θ)
    if d == 0
        utility = -θ[1] * x - θ[2] * (x/100)^2 + ϵ[1]
    elseif d == 1
        utility = -θ[3] + ϵ[2]
    end
    return utility
end

# set up Zurcher's utility in vector form

function utility_vec(d, x, θ)
    if d == 0
        utility = -θ[1] .* x .- (θ[2] .* (x./100).^2)
    elseif d == 1
        utility = -θ[3] .+ zeros(length(x))
    end
    return utility
end




# how to recover engine replacement - mileage will decrease suddenly
# first bin all the miles
K = 10
mileage = raw_data[!,1]
bin_size = maximum(mileage ./ K)



main_data = @transform(raw_data, mileage_binned = ceil.(Integer, :milage ./ bin_size))
binned_miles = main_data[!,2]


data_size = size(raw_data)[1]
col3 = zeros(data_size)
col4 = zeros(data_size)
for i in 1:data_size 
    if i == 1
        # should we ignore this or count?
        col3[i] = 0
        col4[i] = 0
    else
        col3[i] = main_data[i-1, 1]
        col4[i] = main_data[i-1, 2]
    end
end
# get lags and binned lags
main_data = insertcols!(main_data, 3, :mileage_lag => col3)
main_data = insertcols!(main_data, 4, :binned_mileage_lag => col4)

main_data = @from i in main_data begin
    @select {
        i.milage, i.mileage_binned, i.mileage_lag, i.binned_mileage_lag,
        replace = i.milage < i.mileage_lag ? 1 : 0,
        new_binned_mileage_lag = i.milage < i.mileage_lag ? 0 : i.binned_mileage_lag,
        difference = i.milage < i.mileage_lag ? i.milage : i.milage - i.mileage_lag
    } 
    @collect DataFrame
end

main_data = @transform(main_data, binned_difference = :mileage_binned - :new_binned_mileage_lag)


# ok transform mileage into mileage bins first of all



increments = main_data[!,8]
hi = increments[increments.>=0]
g = countmap(hi)
# get at the maximum state space reached
max_bin = Int(maximum(keys(g)))

g_dict = Dict()


for i in 0:max_bin
    g_dict[i] = g[i]/data_size
end
#get array from dict
g_array = [g_dict[i] for i in 0:max_bin]



difference = main_data[!, 7]
hi2 = difference[difference.>=0]
g2 = countmap(hi2)
# get at the maximum state space reached
max_miles = Int(maximum(keys(g2)))

g_dict2 = Dict()


for i in 0:max_miles
    g_dict2[i] = g2[i]/data_size
end
#get array from dict
g_array2 = [g_dict2[i] for i in 0:max_miles]


mileage = main_data[!,1]
mile_k = main_data[!,2]
bin_size = maximum(mileage ./ K)


K_miles = maximum(main_data[!,1]) +1

# initializze transition probabilities
F0_raw = zeros(K_miles,K_miles)
F1_raw = zeros(K_miles,K_miles)

# construct both transition probabilities
for i in 1:K_miles
    if i < K_miles - max_miles
        F0_raw[i, i:i+max_miles] .= g_array2
    else
        F0_raw[i, i:K_miles] .= g_array2[1:K_miles-i+1]
    end
    F1_raw[i, 1:max_miles+1] .= g_array2
end


F0 = zeros(K, K)

mile_k = main_data[!,2]
bin_size = maximum((mileage .+ .1) ./ K)
new_dict = Dict()

for i in 1:K_miles
    # first figure out what Xs are bins
    new_dict[i] = ceil(Integer, i / bin_size)
end

for i in 1:K_miles-1
    for j in 1:K_miles-1
        F0[new_dict[i], new_dict[j]] += (F0_raw[i,j]/bin_size)
    end
end

# 
for i in 1:K
    summed = sum(F0[i,:])
    F0[i, K] += (1-summed)
end

F0 = F0'


function gamma(EV, x, θ, F0, β=.999)
    EV0 = EV[1]
    V0 =  utility_vec(0,x,θ) .+ β .* EV 
    V1 = utility_vec(1,x, θ) .+ β .* EV0 
    # to avoid numerical instability
    Vmax = [max(V0[i], V1[i]) for i in 1:length(V0)]
    return F0 * ( Vmax .+ log.( exp.( V0 .- Vmax ) .+  exp.( V1 .- Vmax ) ) )
end

states = [1:K;] .* bin_size .- (bin_size ./ 2)
θ1 = [.01, .01, .01]
EV1 = zeros(K)
G1 = gamma(EV1, states, θ1, F0)
G2 = gamma(G1, states, θ1, F0)

function get_pks(EV, x, θ, β=.999)
    EV0 = EV[1]
    V0 = utility_vec(0, x, θ) .+ β .* EV
    V1 = utility_vec(1, x, θ) .+ β .* EV0
    Vmax = [max(V0[i], V1[i]) for i in 1:length(V0)]
    #Vmax = maximum(hcat(V0, V1))
    #println(Vmax)
    denominator = exp.(V0 .- Vmax) .+ exp.(V1 .- Vmax)
    numerator = exp.(V1 .- Vmax)
    #denominator = 1 .+ exp.(V0 .- V1)
    #pk = 1 ./ denominator
    #pk[pk .< 1e-90] .= 1e-90
    #pk[pk .> 1-1e-90] .= 1-1e-90
    return numerator ./ denominator
end

hi = get_pks(EV1, states, θ1)

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

gamma_prime(EV1, states, θ1, F0)
get_pks(EV1, states, θ1)


function newton_kantarovich(EV_start, x, θ, F0, tol = 1e-14, maxiter = 10000)
    EV_old = EV_start
    # initialize delta
    diff = 100
    tol_old = 100
    iter = 0
    ID = Matrix{Float64}(I, K, K)
    # run the contraction
    x_k = EV_old
    y_k = EV_old
    while diff > tol && iter < maxiter
        midpoint = (x_k .+ y_k) ./ 2
        Gamma_prime = gamma_prime(midpoint, x, θ, F0)
        #Gamma_prime = gamma_prime(EV_old, θ, F0)
        #Gamma = gamma(EV_old, θ, F0)
        #EV_next = EV_old - (ID - Gamma_prime) \ (EV_old - Gamma)
        #diff = maximum(abs.(EV_next.-EV_old))
        #iter += 1
        #EV_old = EV_next
        Gammak = gamma(x_k, x, θ, F0)
        xk1 = x_k - (ID - Gamma_prime) \ (x_k - Gammak)
        Gammak1 = gamma(xk1, x, θ, F0)
        yk1 = xk1 - (ID - Gamma_prime) \ (xk1 - Gammak)
        diff = maximum(abs.(xk1 - x_k))
        #diff = abs.(tol_new - tol_old)
        #tol_old = tol_new
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



# likelihood time 

function likelihood(θ, d, x, miles, β=.999)
    states = [1:K;] .* bin_size .- (bin_size ./ 2)
    EV1 = [1:K;]
    EV = solve_fixed_pt(EV1, states, θ, F0)
    u0 = utility_vec(0, miles, θ)
    u1 = utility_vec(1, miles, θ)
    EVvec = [EV[i] for i in x]
    V0 = u0 .+ β .* EVvec
    V1 = u1 .+ β .* EV[1]
    pk = 1 ./ (1 .+ exp.(V1 - V0))
    # get rid of the log(0) problems
    #pk[pk .< 1e-90] .= 1e-90
    #pk[pk .> 1-1e-90] .= 1-1e-90
    likelihood = 0
    for i in x
        pxt = pk[i]
        dt = d[i]
        likelihood += dt * log(1-pxt) + (1-dt) * log(pxt)
    end
    return -likelihood
end

miles = main_data[!,1]
x = main_data[!,2]
d = main_data[!,5]

θ1 = [.01,.01,.01]

hi = likelihood(θ1, d, x, miles)

function f(θ)
    return likelihood(θ, d, x, miles)
end

optimize(f, θ1)

function du1(θ, x)
    return [-x -(x./100).^2 zeros(length(x))]
end


function du0(θ, x)
    return [zeros(length(x)) zeros(length(x)) -ones(length(x))]
end


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

hi = dG(EV1, θ1, states, F0)

function dEV(EV, θ, x)
    # KxK
    Gprime = gamma_prime(EV, x, θ, F0)
    # Kx3
    dG_dtheta = dG(EV, θ, x, F0)
    # Kx3
    return (I - Gprime) \ dG_dtheta
end


function dpk(EV, states, θ, β = 0.999)
    # these should be Kx1
    u0 = utility_vec(0, states, θ)
    u1 = utility_vec(1, states, θ)
    V0 = u0 .+ β .* EV
    V1 = u1 .+ β .* EV[1]
    # these should be Kx3
    pk = get_pks(EV, states, θ)
    du1_dtheta = du1(θ, states)
    du0_dtheta = du0(θ, states)
    dEV_dtheta = dEV(EV, θ, states)
    dEV0 = dEV_dtheta[1]
    coef = (du1_dtheta .+ β .* dEV0 - du0_dtheta .- β .* dEV_dtheta)
    # should be Kx3
    return -pk .* (1 .- pk) .* coef 
end

β = .999


function likelihood_grad!(θ, β=.999)
    EV1 = [1:K;]
    EV = solve_fixed_pt(EV1, states, θ, F0)
    pk = get_pks(EV, states, θ)
    # this should return a Kx3
    dpk_dtheta = dpk(EV, states, θ)
    # get a Tx1
    pxt_vec = [pk[i,:][1] for i in x]
    # Tx3
    dp_old = [dpk_dtheta[i,:] for i in x]
    dpxt_vec = [dpk_dtheta[i,j] for i in x for j in 1:length(θ1)]
    dpxt_vec = reshape(dpxt_vec, (length(θ1), length(x)))
    dpxt_vec = dpxt_vec'
    # check dimensions to make sure this returns a 3x1
    result_vec = (d ./ (1 .- pxt_vec)) .* (dpxt_vec) .+ ((1 .- d) ./ pxt_vec) .* dpxt_vec
    return sum(result_vec, dims = 1)
end



optimize(f, likelihood_grad!, θ1)