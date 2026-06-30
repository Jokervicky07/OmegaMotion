#!/bin/bash
set -e

# =========================
# Paths
# =========================
GT_ROOT="/home/jhshen/Experiment/Easi3R/data/davis/DAVIS/Annotations/480p"

PRED_ROOT="/home/jhshen/Experiment/MyMotion/outputs/vggt_omega_v1/davis_sam/final"
EVAL_OUTPUT_DIR="/home/jhshen/Experiment/MyMotion/results/vggt_omega_v1/davis_sam/eval_final/2016"

# =========================
# Evaluation config
# =========================
THRESHOLD=0.5
BOUND_TH=0.008
MISSING_POLICY="zero"

python evaluate.py \
  --pred_root "${PRED_ROOT}" \
  --gt_root "${GT_ROOT}" \
  --output_dir "${EVAL_OUTPUT_DIR}" \
  --threshold "${THRESHOLD}" \
  --bound_th "${BOUND_TH}" \
  --missing_policy "${MISSING_POLICY}" \
  --sequence_list splits/davis2016.txt