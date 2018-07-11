#!/bin/bash

#SBATCH -J skstrp                               # Job Name
#SBATCH -o ./output/skstrp.%j.out               # Output file (%j is the job id)
#SBATCH -p gpu-long                             # Queue Name
#SBATCH -N 1                                    # Total number of nodes requested
#SBATCH -n 1                                    # Total number of mpi tasks
#SBATCH -t 48:00:00                             # Requested run time (hh:mm:ss)
#SBATCH --mail-user=yliu30@mdanderson.org       # Email
#SBATCH --mail-type=begin
#SBATCH --mail-type=end

# Loading dependencies into module
module load gcc/4.9.3 cuda/8.0 cudnn/5.1
module load python3/3.5.2
module load tensorflow-gpu

# Activate virtualenv
source ./venv/bin/activate

# Start the job
python ./main.py

