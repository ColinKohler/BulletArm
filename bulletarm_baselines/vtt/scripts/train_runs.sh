#!/bin/bash

main() {
  local env=$1
  local vision_size=$2
  local encoder=$3
  local results_path=$4

  for j in $(seq ${5} ${6}); do
    sbatch -J ${env}_${results_path}_${j} scripts/train.sbatch $env $vision_size $encoder ${results_path}_${j}
  done
}

main $1 $2 $3 $4 $5 $6
