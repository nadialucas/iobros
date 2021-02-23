# Nadia Lucas and Fern Ramoutar
# February 26, 2021
#
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

# get mileage data
current_path = pwd()
if (pwd() == "/Users/nadialucas") | (pwd() == "/Users/nadialucas/Documents/iobros")
    data_path = "/Users/nadialucas/Documents/iobros/pset2/"
elseif pwd() == "/home/nrlucas"
    data_path = "/home/nrlucas/IO2Data/"
elseif pwd() == "/home/cschneier"
    data_path = "/home/cschneier/IO/"
    out_path = "/home/cschneier/nadia/"
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
function utility_vec(d, θ)
    x = [0:K-1;]
    if d == 0
        utility = -θ[1] .* x .- θ[2] .* (x./100).^2
    elseif d == 1
        utility = -θ[3]
    end
    return utility
end

function utility(d, x, θ)
    if d == 0
        utility = -θ[1] * x - θ[2] * (x/100)^2 + ϵ[1]
    elseif d == 1
        utility = -θ[3] + ϵ[2]
    end
    return utility
end


# how to recover engine replacement - mileage will decrease suddenly

data_size = size(raw_data)[1]
col2 = zeros(data_size)
for i in 1:data_size 
    if i == 1
        # should we ignore this or count?
        col2[i] = 0
    else
        col2[i] = main_data[i-1, 1]
    end
end

main_data = insertcols!(raw_data, 2, :mileage_lag => col2)

main_data = @from i in main_data begin
    @select {
        i.milage, i.mileage_lag,
        replace = i.milage < i.mileage_lag ? 1 : 0,
        new_mileage_lag = i.milage < i.mileage_lag ? 0 : i.mileage_lag
    } 
    @collect DataFrame
end

main_data = @transform(main_data, difference = :milage - :new_mileage_lag)

miles = main_data[!,5]
hi = miles[miles.>=0]
g = countmap(hi)
# get at the maximum state space reached
max_miles = Int(maximum(keys(g)))

g_dict = Dict()


for i in 0:max_miles
    g_dict[i] = g[i]/data_size
end
#get array from dict
g_array = [g_dict[i] for i in 0:max_miles]

K = Int(maximum(main_data[!,1])+1 + max_miles)
# initializze transition probabilities
F0 = zeros(K,K)
F1 = zeros(K,K)

# construct both transition probabilities
for i in 1:K
    if i < K-max_miles
        F0[i, i:i+max_miles] .= g_array
    else
        F0[i, i:K] .= g_array[1:K-i+1]
    end
    F1[i, 1:max_miles+1] .= g_array
end


function gamma(EV,F0,θ)
    EV0 = EV[1]
    term1 =  uvec(0,θ) .+ β .* EV 
    term2 = uvec(1,θ) .+ β .* EV0 
    hc = hcat(term1,term2)
    v0 = [ maximum( hc[ii,:] ) for ii in 1:size(hc,1) ]
    return F0 * ( v0 + log.( exp.( term1 .- v0 ) .+  exp.( term2 .- v0 ) ) )
end

function gamma(EV, θ, F0, β = .999)
    EV0 = EV[1]
    # must be evaluated at a vector for all states
    u0 = utility_vec(0, θ)
    # does not matter what x is
    u1 = utility_vec(1, θ)
    println(size(u0))
    println(size(F0))
    Gamma = F0 * (log.(exp.(u0 .+ β .* EV ) .+ exp.(u1 .+ β*EV0)))
    return Gamma
end

θ1 = [1,2,3]
EV1 = zeros(K)
gamma(EV1, θ1, F0)

function get_pks(EV, θ, β=.999)
    EV0 = EV[1]
    u0 = utility_vec(0, θ)
    u1 = utility_vec(1, θ)
    denominator = 1 .+ exp.(u1 .- β .* EV0 - u0 - β .* EV)
    return 1 ./ denominator
end

function gamma_prime(EV, θ, F0, β = .999)
    EV0 = EV[1]
    u0 = utility_vec(0, θ)
    u1 = utility_vec(1, θ)
    pks = get_pks(EV, θ)
    numerator = exp.(u0 .+ β .* EV) 
    return β .* F0 * (numerator ./ denom)
end

θ1 = [1,2,3]
EV1 = zeros(K)
yo = gamma_prime(EV1, θ1, F0)


function newton_kantarovich(EV_start, θ, F0, β = .999, tol = 1e-14, maxiter = 1000)
    EV_old = EV_start
    # initialize delta
    diff = 100
    iter = 0
    ID = Matrix{Float64}(I, K, K)
    # run the contraction
    while diff > tol && iter < maxiter
        Gamma_prime = gamma_prime(EV_old, θ, F0)
        println(size(Gamma_prime))
        println(ID - Gamma_prime)
        Gamma = gamma(EV_old, θ, F0)
        println(ID - Gamma)
        println("hai")
        EV_next = EV_old - (ID - Gamma_prime) \ (ID - Gamma) * EV_old
        diff = maximum(abs.(EV_next.-EV_old))
        iter += 1
        EV_old = EV_next
    end
    return EV_next
end


θ1 = [1,2,3]
EV1 = zeros(K)
newton_kantarovich(EV1, θ1, F0)


#check tol ratio every step and then switch to newton kantorivich