# Environment Set Up

## Set up Python 3 on log-in node

Loading dependent modules

```
module load gcc/4.9.3 cuda/8.0 cudnn/5.1
module load python3/3.5.2
```

## Save and Restore Modules Dependencies
Since Maverick refreshes the modules everytime we log in, we could save the current
environment into a module collection. Next time when we log in, we coudl simply restore the module from the list.

Save the current modules into the collection `py3`
```
module save py3
```

Restore the modules from the colleciton `py3`
```
module restore py3
```

## Set up virtual environment for the project

Under the project directory `$WORK\maverick\my-project`, create an virtual environment named `venv` with python 3 

```
virtualenv -p python3 venv
```

To activate the virtual environment `venv`, run
```
source ./venv/bin/activate
```

## Install tensorflow

Due to driver issues, installing `tensorflow` with `pip` doesn't work on Maverick. We need to load the provided module to get it working.

```
module load tensorflow-gpu
```

## Install Keras

After activating the virutalenv, using `pip` to install Keras.
```
pip install keras
```



# SLURM Config

## Create SLURM file
Under the project directory, create a file `myjob.slurm`
```
touch myjob.slurm
```

Now the structure of the project directory is
```
- my-project
    - venv
        - ...
    - main.py
    - myjob.slurm   

``` 

Edit `myjob.slurm`
(Note: the path in `main.py` is also relative to `myjob.slurm`)

```
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

```

## Submit job to the queue

Run
```
sbatch ./myjob.slurm
```

To check the status of the job in queue,
```
squeue -u username
```