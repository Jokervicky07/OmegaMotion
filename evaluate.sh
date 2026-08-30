#!/bin/bash
set -e

# =========================
# Paths
# =========================
# GT_ROOT="/home/jhshen/Experiment/Easi3R/data/davis/DAVIS/Annotations/480p"
# GT_ROOT="/data2/jhshen/dataset/SegTrackv2/Annotations"
GT_ROOT="/data2/jhshen/dataset/FBMS59/Annotations"

# PRED_ROOT="/home/jhshen/Experiment/MyMotion/outputs/vggt_omega_v2_geo_cam_depth/davis_sam/final"
# EVAL_OUTPUT_DIR="/home/jhshen/Experiment/MyMotion/results/vggt_omega_v2_geo_cam_depth/davis_sam/eval_final/2016"

# PRED_ROOT="/home/jhshen/Experiment/VGGT4D/result_clean/"
# EVAL_OUTPUT_DIR="/home/jhshen/Experiment/VGGT4D/result_clean/2016"


PRED_ROOT="/home/jhshen/Experiment/MyMotion/evaluate/FlowP-SAM/fbms"
EVAL_OUTPUT_DIR="/home/jhshen/Experiment/MyMotion/evaluate/FlowP-SAM/fbms"

# PRED_ROOT="/data2/jhshen/dataset/SegTrackv2/vggt_omega_v2/final"
# EVAL_OUTPUT_DIR="/home/jhshen/Experiment/MyMotion/results/vggt_omega_v2/segtrack_v2_sam/eval_final"


# PRED_ROOT="/home/jhshen/Experiment/SegAnyMo/result_seg/sam2/initial_preds"
# EVAL_OUTPUT_DIR="/home/jhshen/Experiment/SegAnyMo/result_seg/sam2/segtrack_v2"


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
  --missing_policy "${MISSING_POLICY}" 
# --sequence_list splits/davis2016.txt