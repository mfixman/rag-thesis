#!/bin/bash
#SBATCH --job-name rag
#SBATCH --partition=prigpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72GB
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH -e results/%x_%j.e
#SBATCH -o results/%x_%j.o
#SBATCH --error run.err
#SBATCH --output run.out

source /opt/flight/etc/setup.sh
flight env activate gridware
module load libs/nvidia-cuda/11.2.0/bin
module load gnu
module add compilers/gcc gnu

export http_proxy=http://hpc-proxy00.city.ac.uk:3128
export https_proxy=http://hpc-proxy00.city.ac.uk:3128
export HTTP_PROXY=http://hpc-proxy00.city.ac.uk:3128
export HTTPS_PROXY=http://hpc-proxy00.city.ac.uk:3128
export TORCH_HOME=/mnt/data/public/torch

job=$1
shift

python ${job} "$@"
