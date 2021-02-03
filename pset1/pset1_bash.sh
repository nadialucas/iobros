#!/bin/bash
#PBS -N TestJob
#PBS -l nodes=1:ppn=10,mem=10gb
#PBS -j oe
module load knitro/12.1.1-z
module load julia
# execute program
/home/nrlucas/iobros/pset1/pset1_knitro.jl