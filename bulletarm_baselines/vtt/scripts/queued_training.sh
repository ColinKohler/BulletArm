#!/bin/bash

train() {
  local i=$1
  local j=$2
  local env=$3
  local vision_size=$4
  local encoder=$5
  local results_path=$6

  if [ $j -eq 1 ]; then
    ret_val=$(sbatch --parsable -J ${env}_${results_path}_${i}_${j} scripts/train.sbatch $env $vision_size $encoder ${results_path}_${i}_${j})
  else
    ret_val=$(sbatch --parsable -J ${env}_${results_path}_${i}_${j} --dependency=afterany:${ret_val} scripts/train.sbatch $env $vision_size $encoder ${results_path}_${i}_${j} ${results_path}_${i}_$((j-1)) ${results_path}_${i}_$((j-1)) ${results_path}_${i}_$((j-1)))
  fi
}


main() {
  local env=$1
  local vision_size=$2
  local encoder=$3
  local results_path=$4

  for i in $(seq ${5} ${6}); do
    local ret_val
    for j in {1..3}; do
      train $i $j $env $vision_size $encoder $results_path
    done
  done
}

main $1 $2 $3 $4 $5 $6
