# Nadia Lucas and Fern Ramoutar
# February 12, 2021
#
# We would like to thank Jordan Rosenthal-Kay, Jeanne Sorin, and George Vojta for helpful comments
# 
# running on Julia version 1.5.2
################################################################################
# Part 2

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
    data_path = "/Users/nadialucas/Documents/iobros/pset3/"
end
raw_data = CSV.read(string(data_path, "ps3.csv"), DataFrame, header = 2)


print(names(raw_data))
# start cleaning

data = rename!(raw_data, "realisation in final auAtion" => "target_winbid")
data = rename!(data, "Data NuEer" => "rowid")


data = @transform(groupby(data, [:lot, :house]), auction_id = maximum(:rowid))
data = @transform(groupby(data, :auction_id), max_ring_bid = maximum(:bid))

sum_data = select(data, [:house, :auction_id, :target_winbid, :max_ring_bid])
sum_data = unique(sum_data)
colwins = [dat:target_winbid > :max_ring_bid ? 0 : 1)



summary_data = @select(groupby(data, :lot), max_ring_bid = maximum(:bid), win_bid = maximum(:target_winbid), ring_prof = maximum(:profit))
