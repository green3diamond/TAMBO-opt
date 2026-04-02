#!/bin/bash
#SBATCH --job-name=ot_part
#SBATCH --mem=100G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH -p shared,sapphire
#SBATCH --output=/n/home04/hhanif/AllShowers/allshowers/logs/ot_part_%j.out
#SBATCH --error=/n/home04/hhanif/AllShowers/allshowers/logs/ot_part_%j.err

module load python
eval "$(mamba shell hook --shell bash)"
mamba config set changeps1 False
mamba activate /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tamboOpt_env/

CONFIG=/n/home04/hhanif/AllShowers/conf/transformer.yaml
SCRIPT=/n/home04/hhanif/AllShowers/allshowers/OT_matchv2.py

# FILE is passed in by sbatch --export
: "${FILE:?FILE not set}"

python "${SCRIPT}" \
  --config "${CONFIG}" \
  --file "${FILE}" \
  --batch-size 64 \
  --num-proc 4 \
  --emd-numitermax 300000 \
  --fit-file /n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_processed_step1_v3/subset_40k_per_pdg.h5 \
  --fit-stop 100000