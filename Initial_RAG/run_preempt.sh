#!/bin/bash
#SBATCH --job-name rag_preempt
#SBATCH --partition=preemptgpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72GB
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:1
#SBATCH -e results/%x_%j.e
#SBATCH -o results/%x_%j.o
#SBATCH --error run_preempt.err
#SBATCH --output run_preempt.out
#SBATCH --constraint=cuda12

# source /opt/flight/etc/setup.sh
# flight env activate conda
# conda init
# conda activate conda
# 
# 
# export NVIDIA_CUDADIR=/opt/gridware/depots/b26e1471/el7/pkg/libs/nvidia-cuda/11.2.0/bin 
# export NVIDIA_CUDABIN=/opt/gridware/depots/b26e1471/el7/pkg/libs/nvidia-cuda/11.2.0/bin/bin 
# export NVIDIA_CUDASAMPLES=/opt/gridware/depots/b26e1471/el7/pkg/libs/nvidia-cuda/11.2.0/bin/samples 
# export NVIDIA_CUDADRIVER_INSTALLER=/opt/gridware/depots/b26e1471/el7/pkg/libs/nvidia-cuda/11.2.0/bin/installers/NVIDIA-Linux-x86_64-460.27.04.run 
# export CUDA_DIR=/opt/gridware/depots/b26e1471/el7/pkg/libs/nvidia-cuda/11.2.0/bin 
# export CUDA_HOME=/opt/gridware/depots/b26e1471/el7/pkg/libs/nvidia-cuda/11.2.0/bin 
# export CUDA_SAMPLES=/opt/gridware/depots/b26e1471/el7/pkg/libs/nvidia-cuda/11.2.0/bin/samples 
# export PATH="/opt/gridware/depots/b26e1471/el7/pkg/libs/nvidia-cuda/11.2.0/bin/bin:$PATH" 
# export MANPATH="/opt/gridware/depots/b26e1471/el7/pkg/libs/nvidia-cuda/11.2.0/bin/doc/man:$MANPATH" 
# export LD_LIBRARY_PATH="/opt/gridware/depots/b26e1471/el7/pkg/libs/nvidia-cuda/11.2.0/bin/lib64:$LD_LIBRARY_PATH" 

flight env activate gridware
module load libs/nvidia-cuda/11.2.0/bin
module load gnu
module add compilers/gcc gnu

export http_proxy=http://hpc-proxy00.city.ac.uk:3128
export https_proxy=http://hpc-proxy00.city.ac.uk:3128
export HTTP_PROXY=http://hpc-proxy00.city.ac.uk:3128
export HTTPS_PROXY=http://hpc-proxy00.city.ac.uk:3128

export WANDB_API_KEY=9692ff12f6990a08e1a75d22ddd651d0f3de3e95
export TORCH_HOME=/mnt/data/public/torch

job=$1
shift

python ${job} "$@"
