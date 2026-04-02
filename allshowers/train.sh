#!/bin/bash
#SBATCH --job-name=train
#SBATCH --mem=100G
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH -p arguelles_delgado_gpu_mixed
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --output=/n/home04/hhanif/AllShowers/allshowers/logs/train_%j.out


module load python
eval "$(mamba shell hook --shell bash)"
mamba config set changeps1 False
mamba activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tamboOpt_env/



python /n/home04/hhanif/AllShowers/allshowers/train.py /n/home04/hhanif/AllShowers/conf/transformer.yaml