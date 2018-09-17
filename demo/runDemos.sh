#!/bin/bash

if [ ! -d "output" ]; then
  mkdir output
fi

echo Running demos

set -x

julia -O3 simpleDemoVisualize.jl > output/simpleDemoVisualize.txt &
julia -O3 simpleDemoBoundaries.jl &
julia -O3 lgDemoVisualize.jl > output/lgDemoVisualize.txt &
julia -O3 lgDemoEstimates.jl > output/lgDemoEstimates.txt &
julia -O3 lgDemoBoundaries.jl &
julia -O3 lgDemoEstimateComparison.jl > output/lgDemoEstimateComparison.txt &

wait
