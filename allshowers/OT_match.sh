#!/bin/bash
#SBATCH --job-name=ot_full
#SBATCH --mem=100G
#SBATCH --cpus-per-task=20
#SBATCH --time=1:00:00
#SBATCH -p arguelles_delgado,sapphire
#SBATCH --output=/n/home04/hhanif/AllShowers/allshowers/logs/ot_full_%j.out
#SBATCH --error=/n/home04/hhanif/AllShowers/allshowers/logs/ot_full_%j.err

module load python
eval "$(mamba shell hook --shell bash)"
mamba config set changeps1 False
mamba activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tamboOpt_env/



python /n/home04/hhanif/AllShowers/allshowers/OT_match.py /n/home04/hhanif/AllShowers/conf/transformer_time_small.yaml --with-time