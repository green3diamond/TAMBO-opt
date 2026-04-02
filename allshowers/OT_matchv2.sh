#!/bin/bash
set -euo pipefail

PART_DIR=/n/holylfs05/LABS/arguelles_delgado_lab/Everyone/hhanif/tambo_simulations/all_shower_ot_matched/
LOG_DIR=/n/home04/hhanif/AllShowers/allshowers/logs
mkdir -p "${LOG_DIR}"

# Find all part files
FILES=( "${PART_DIR}"/merged_all_showers_part*.h5 )

if [ ${#FILES[@]} -eq 0 ]; then
  echo "No files found matching ${PART_DIR}/merged_all_showers_part*.h5"
  exit 1
fi

echo "Found ${#FILES[@]} part files."

for f in "${FILES[@]}"; do
  base=$(basename "$f" .h5)   # e.g. merged_all_showers_part007
  part_id=${base##*part}      # e.g. 007

  echo "Submitting OT job for ${f}"
  sbatch \
    --job-name="ot_p${part_id}" \
    --output="${LOG_DIR}/ot_p${part_id}_%j.out" \
    --export=ALL,FILE="${f}" \
    /n/home04/hhanif/AllShowers/allshowers/OT_matchv2_submission.sh
done

echo "Done submitting all OT jobs."