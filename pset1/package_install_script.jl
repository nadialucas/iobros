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

# sets python environment to path 
# run "which python" in terminal to get this
ENV["PYTHON"] = "/usr/bin/python2.7"
Pkg.build("PyCall")

# then restart julia and you're good to run PyCall
pyblp = pyimport("pyblp")