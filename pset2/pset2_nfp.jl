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
using StatsBase

# get mileage data
current_path = pwd()
if pwd()[1:17] == "/Users/nadialucas"
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
#function utility_vec(d, θ)
#    max_miles
#    x = [0:K;]
#    if d == 0
#        utility = -θ[1] .* x .- θ[2] .* (x./100).^2
#    elseif d == 1
#        utility = -θ[3] .+ zeros(K+1)
#    end
#    return utility
#end
# utility inner func
function u(x,d,θ)
    if d == 0
            return -θ[1].* x - θ[2] .* ( x ./ 100).^2 
    end
    if d == 1 
            return -θ[3] .+ 0 .* x
    end
end
function get_xk_mid(xvec,kvec,f)
    K = maximum(kvec) - 1
    fmids = zeros(K+1)
    for kk in 2:(K+1)
            inds = findall(x->x==kk,kvec)
            mid = (maximum(xvec[inds]) + minimum(xvec[inds]) )/2
            fmids[kk] = f(mid)
    end
    fmids[1] = f(0)
    return fmids
end
# utility vector function
function utility_vec(d,θ)
    return get_xk_mid(milage_t_plus_1,mile_k, xx->u(xx,d,θ))
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
        col2[i] = raw_data[i-1, 1]
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

function x_to_k(x,band)
    return ceil.(Integer, (x .+ .1) ./ band)
end


function make_markov(xt1,d,K)
    # start with transitions 
    xt = vcat(0, xt1[1:end - 1])
    mat = zeros(K,K)
    replace_t = xt .> xt1
    band = (maximum(xt1) + .1) / K
    kt = x_to_k(xt,band)
    kt1 = x_to_k(xt1,band)
    for ii in 1:length(kt)
            mat[kt[ii],kt1[ii]] += 1-replace_t[ii]
    end
    colsums = mat * ones(K,1) 
    out = diagm( 1 ./ colsums[:] ) * mat 
    out[isnan.(out)] .= 0
    # want to append zero
    zero_row = zeros(1,K+1)
    for ii in 1:length(kt)
        zero_row[1,kt1[ii]+1] += replace_t[ii]
    end
    zero_row = zero_row ./ sum(zero_row)
    out = hcat(zeros(K,1),out)
    out = vcat(zero_row,out)
    return (1-d).*out + d.*repeat(out[1,:]',K+1,1) , kt1 .+ 1
end

miles = main_data[!,1]
milage_t_plus_1 = miles


K = 30



#milage_t_plus_1 = [1,2,4,6,1,4,6,7,1,2,3,4,1,2,1,4,6,7,4]
#miles = milage_t_plus_1
#K = 3
F0 = make_markov(miles, 0, K)[1]
F0 = make_markov(miles, 0, K)[1]
mile_k = make_markov(miles, 0, K)[2]
F1 = make_markov(miles, 1, K)[1]


#K_max = Int(maximum(main_data[!,1])+1 + max_miles)
#K = 10
# initializze transition probabilities
#F0 = zeros(K,K)
#F1 = zeros(K,K)

# construct both transition probabilities
#for i in 1:K
#    if i < K-max_miles
#        F0[i, i:i+max_miles] .= g_array
#    else
#        F0[i, i:K] .= g_array[1:K-i+1]
#    end
#    F1[i, 1:max_miles+1] .= g_array
#end

#xt1 = [1,2,4,6,1,4,6,7,1,2,3,4,1,2,1,4,6,7,4]
#K = 3
#F0 = make_markov(xt1, 0, K)[1]


function gamma(EV, θ, F0, β=.999)
    EV0 = EV[1]
    V0 =  utility_vec(0,θ) .+ β .* EV 
    V1 = utility_vec(1,θ) .+ β .* EV0 
    # to avoid numerical instability
    Vmax = [max(V0[i], V1[i]) for i in 1:length(V0)]
    return F0 * ( Vmax .+ log.( exp.( V0 .- Vmax ) .+  exp.( V1 .- Vmax ) ) )
end

θ1 = [1,0,1]
EV1 = ones(K+1)
gamma(EV1, θ1, F0)

function get_pks(EV, θ, β=.999)
    EV0 = EV[1]
    u0 = utility_vec(0, θ)
    u1 = utility_vec(1, θ)
    denominator = 1 .+ exp.(u1 .- β .* EV0 .- u0 - β .* EV)
    return 1 ./ denominator
end

function gamma_prime(EV, θ, F0, β = .999)
    EV0 = EV[1]
    u0 = utility_vec(0, θ)
    u1 = utility_vec(1, θ)
    pks = get_pks(EV, θ)
    pks_diag = Diagonal(pks)
    EV0term = zeros(K+1, K+1)
    EV0term[:,1] = 1 .- pks
    return β .* F0 * (pks_diag + EV0term)
end

EV1 = ones(K+1)
gamma_prime(EV1, θ1, F0)
get_pks(EV1, θ1)


function newton_kantarovich(EV_start, θ, F0, tol = 1e-14, maxiter = 10000)
    EV_old = EV_start
    # initialize delta
    diff = 100
    iter = 0
    ID = Matrix{Float64}(I, K+1, K+1)
    # run the contraction
    x_k = EV_old
    y_k = EV_old
    while diff > tol && iter < maxiter
        midpoint = (x_k .+ y_k) ./ 2
        Gamma_prime = gamma_prime(midpoint, θ, F0)
        #Gamma_prime = gamma_prime(EV_old, θ, F0)
        #Gamma = gamma(EV_old, θ, F0)
        #EV_next = EV_old - (ID - Gamma_prime) \ (EV_old - Gamma)
        #diff = maximum(abs.(EV_next.-EV_old))
        #iter += 1
        #EV_old = EV_next
        Gammak = gamma(x_k, θ, F0)
        xk1 = x_k - (ID - Gamma_prime) \ (x_k - Gammak)
        Gammak1 = gamma(xk1, θ, F0)
        yk1 = xk1 - (ID - Gamma_prime) \ (xk1 - Gammak)
        diff = maximum(abs.(xk1 - x_k))
        iter += 1
        x_k = xk1
        y_k = yk1

    end
    println("iterr")
    println(iter)
    #return EV_old
    return y_k
end

function solve_fixed_pt(EV_start, θ, F0, β = .999, tol = 1e-14, maxiter = 1000)
    EV_old = EV_start
    # initialize delta
    diff_old = 100
    diff_ratio = 100
    iter = 0
    ID = Matrix{Float64}(I, K+1, K+1)
    # run the contraction
    while abs(diff_ratio - β)>tol && iter < maxiter
        EV_new = gamma(EV_old, θ, F0)
        diff_new = maximum(abs.(EV_new.-EV_old))
        EV_old = EV_new
        diff_ratio = diff_new/diff_old
        diff_old = diff_new
        iter += 1
    end
    println(EV_old)
    println("yo")
    println(diff)
    println(iter)
    println("frick")
    final_EV = newton_kantarovich(EV_old, θ, F0)
end

EV1 = ones(K+1)
@time yvec = solve_fixed_pt(EV1, θ1, F0)
xvec = get_xk_mid(milage_t_plus_1,mile_k, xx->xx)
using Plots
plot(xvec,yvec, label="EV(x|θ)")
xlabel!("miles driven")
ylabel!("continuation value")


#check tol ratio every step and then switch to newton kantorivich