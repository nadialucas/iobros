import Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("DataFramesMeta")
Pkg.add("Random")
Pkg.add("Distributions")
Pkg.add("LinearAlgebra")
Pkg.add("Statistics")
Pkg.add("ForwardDiff")
Pkg.add("PyCall")
Pkg.add("FiniteDiff")

# sets python environment to path 
# run "which python" in terminal to get this
ENV["PYTHON"] = "/usr/bin/python2.7"
Pkg.build("PyCall")

# then restart julia and you're good to run PyCall
pyblp = pyimport("pyblp")


current_path = pwd()
if pwd() == "/Users/nadialucas"
    data_path = "/Users/nadialucas/Dropbox/Second year/IO 2/pset1/"
elseif pwd() == "/home/nrlucas"
    data_path = "/home/nrlucas/IO2Data/"
end
open(string(data_path, "knitro_out.txt"), "a") do io
    println(io, data_path)
    close(io)
end