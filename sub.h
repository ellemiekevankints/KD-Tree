#!/bin/bash
#SBATCH --job-name=KDTree.cu          # Job name
#SBATCH --partition=gpu_p             # Partition (queue) name
#SBATCH --gres=gpu:1                  # Requests one GPU device 
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=2             # Number of CPU cores per task
#SBATCH --mem=4gb                     # Job memory request
#SBATCH --time=10:00:00               # Time limit hrs:min:sec
#SBATCH --output=log/KDTree.%j.out         
#SBATCH --error=log/KDTree.%j.err 

#SBATCH --mail-type=BEGIN,END,FAIL    # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ejv88036@uga.edu  # Where to send mail

cd $SLURM_SUBMIT_DIR

ml CUDA/10.0.130
ml GCCcore/6.4.0

g++ -Wall -pedantic-errors -g -O0 -c src/Logger.hpp -g 
g++ -Wall -pedantic-errors -g -O0 -c src/Logger.cpp -g -o bin/Logger.o
nvcc src/Feature.cu -g -include src/Feature.cuh -dc -o bin/Feature.o 
nvcc src/KDTree.cu -g -include src/KDTree.cuh -o bin/KDTree.cu.o -dc 
nvcc bin/Feature.o bin/KDTree.cu.o bin/Logger.o -o KDTree

./KDTree

rm bin/Logger.o
rm bin/Feature.o
rm bin/KDTree.cu.o