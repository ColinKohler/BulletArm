#!/bin/bash

#SBATCH --output=scripts/outputs/%j.out
#SBATCH --error=scripts/outputs/%j.err

#SBATCH --nodes=1

#SBATCH --partition=gpu
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=60GB
#SBATCH --time=08:00:00

#SBATCH --mail-type=FAIL
#SBATCH --mail-user=kohler.c@northeastern.edu

#SBATCH --exclude=c[2160,2162-2175,2176-2183],d1008,d1011,d1026

echo "node: $SLURM_NODELIST"
if [ $# -eq 3 ]; then
  python -u scripts/train.py $1 --vision_size=$2 --encoder=l3
elif [ $# -eq 4 ]; then
  python -u scripts/train.py $1 --vision_size=$2 --encoder=$3 --results_path=$4
else
  python -u scripts/train.py $1 --vision_size=$2 --encoder=$3 --results_path=$4 --checkpoint=$5 --buffer=$6
fi
