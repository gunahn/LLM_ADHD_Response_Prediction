#!/bin/bash
#SBATCH --job-name=adhd_fewshot
#SBATCH --partition=ou_bcs_normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
set -euo pipefail
source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh
conda activate responder
export PYTHONNOUSERSITE=1 HF_HOME=/orcd/data/gabrieli/001/guna23/hf_cache HF_HUB_OFFLINE=1
cd /orcd/data/gabrieli/001/guna23
nvidia-smi --query-gpu=name --format=csv,noheader || true
python scripts/adhd_fewshot_curve.py \
  --data "data/Prediction Data for GA (02.13.26).xls" \
  --out results/fewshot --repeats 10 --models "LR,RF,XGBoost,TabNet,TabFM"
