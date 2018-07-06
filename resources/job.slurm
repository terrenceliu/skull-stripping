#!/bin/bash

#SBATCH -J skstrp
#SBATCH -o skstrp.%j.out
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:05:00
#SBATCH --mail-user=yliu30@mdanderson.org
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

module load python3
module load cuda

source ./venv/bin/activate

python ./main.py

