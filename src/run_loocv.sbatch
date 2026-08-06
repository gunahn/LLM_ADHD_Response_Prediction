#!/bin/bash
#SBATCH --job-name=responder_loocv
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# --- environment ---
source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda activate responder
export PYTHONNOUSERSITE=1                                   # ignore ~/.local (CUDA torch)
export HF_HOME=/orcd/data/gabrieli/001/guna23/hf_cache      # pre-cached model weights
export HF_HUB_OFFLINE=1                                     # use cache, no network on node
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export TOKENIZERS_PARALLELISM=false

PROJ=/orcd/data/gabrieli/001/guna23
DATA="$PROJ/data/Prediction Data for GA (02.13.26).xls"
OUT="$PROJ/results/loocv_$(date +%Y%m%d_%H%M%S)"

# MODELS: override with `sbatch --export=ALL,MODELS=...`. Default omits TabPFN
# (license-gated). Set TABPFN_TOKEN and MODELS=... to include it.
MODELS="${MODELS:-LR,RF,XGBoost,TabNet,TabFM}"

echo "host=$(hostname)  job=$SLURM_JOB_ID  $(date)"
echo "models: $MODELS"
python --version
python "$PROJ/scripts/responder_loocv.py" --data "$DATA" --out "$OUT" --only "$MODELS"
echo "results in: $OUT"
