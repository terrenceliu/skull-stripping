#!/bin/bash

#SBATCH -J skstrp
#SBATCH -o ./output/skstrp.%j.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:01:00
#SBATCH --mail-user=yliu30@mdanderson.org
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

module load python3/3.5.2
module load cuda
module load tensorflow-gpu

python ./main.py

