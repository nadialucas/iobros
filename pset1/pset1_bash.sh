#!/bin/bash
#PBS -N TestJob
#PBS -l nodes=1:ppn=10,mem=10gb
#PBS -j oe
module load julia
module load knitro
# execute program
julia $HOME/nadia/iobros/pset1/pset1.jl