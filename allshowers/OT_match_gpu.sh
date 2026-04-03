#!/bin/bash
#SBATCH --job-name=ot_full
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH -p arguelles_delgado_gpu_mixed
#SBATCH --output=/n/home04/hhanif/AllShowers/logs/ot_full_%j.out
#SBATCH --error=/n/home04/hhanif/AllShowers/logs/ot_full_%j.err
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1

module load python
module load cuda
eval "$(mamba shell hook --shell bash)"
mamba config set changeps1 False
mamba activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tamboOpt_env/



python /n/home04/hhanif/AllShowers/allshowers/OT_match.py /n/home04/hhanif/AllShowers/conf/transformer_time.yaml --with-time