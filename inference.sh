#!/bin/bash
set -e

# =========================
# Paths
# =========================

MODEL_PATH="/home/jhshen/Experiment/MyMotion/checkpoints/vggt_omega_v2.pth"
RAFT_MODEL_PATH="/home/jhshen/Experiment/MyMotion/raft_large_C_T_SKHT_V2-ff5fadd5.pth"

SAM2_CONFIG_PATH="/home/jhshen/Experiment/MyMotion/sam2-main/sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT_PATH="/home/jhshen/Experiment/MyMotion/sam2-main/checkpoints/sam2.1_hiera_large.pt"

IMAGE_ROOT="/home/jhshen/Experiment/Easi3R/data/davis/DAVIS/JPEGImages/480p"
# IMAGE_ROOT="/data2/jhshen/dataset/SegTrackv2/JPEGImages"
OUTPUT_ROOT="/home/jhshen/Experiment/MyMotion/outputs/vggt_omega_v2/davis_sam/"
#OUTPUT_ROOT="/data2/jhshen/dataset/SegTrackv2/vggt_omega_v4"

INPUT_DIR="/home/jhshen/Experiment/Easi3R/data/davis/DAVIS/JPEGImages/480p/kite-surf/"
OUTPUT_DIR="/home/jhshen/Experiment/MyMotion/outputs/vggt_omega_v6/davis_sam/"

# =========================
# Inference config
# =========================
SEQUENCE_LENGTH=32
IMG_SIZE=512
DEVICE="cuda"

# CUDA_VISIBLE_DEVICES=1 python infer_vggt_omega_v1.py \
#   --model_path "${MODEL_PATH}" \
#   --raft_model_path "${RAFT_MODEL_PATH}" \
#   --image_root "${IMAGE_ROOT}" \
#   --output_root "${OUTPUT_ROOT}" \
#   --img_size "${IMG_SIZE}" \
#   --sequence_length "${SEQUENCE_LENGTH}" \
#   --device "${DEVICE}" \
#   --use_geo_feature True \
#   --use_cam_feature True \
#   --use_flow_feature False \
#   --use_depth_feature True \
#   --use_sam_refine True 
  
  
# --use_flow_residual True

# CUDA_VISIBLE_DEVICES=2 python infer_vggt_omega.py \
#   --model_path "${MODEL_PATH}" \
#   --raft_model_path "${RAFT_MODEL_PATH}" \
#   --input_dir "${INPUT_DIR}" \
#   --output_root "${OUTPUT_DIR}" \
#   --img_size "${IMG_SIZE}" \
#   --sequence_length "${SEQUENCE_LENGTH}" \
#   --device "${DEVICE}" \
#   --mixed_precision True \
#   --use_sam_refine True \
#   --sam_seed_threshold 0.8 \
#   --save_raw True \
#   --save_sam True \
#   --save_final True \
#   --save_vis True \
#   --save_per_frame_vis True \
#   --save_masks_at_original_size False

# CUDA_VISIBLE_DEVICES=0 python infer_vggt_omega.py \
#   --model_path "${MODEL_PATH}" \
#   --raft_model_path "${RAFT_MODEL_PATH}" \
#   --image_root "${IMAGE_ROOT}" \
#   --output_root "${OUTPUT_ROOT}" \
#   --img_size "${IMG_SIZE}" \
#   --sequence_length "${SEQUENCE_LENGTH}" \
#   --device "${DEVICE}" \
#   --use_sam_refine True \
#   --save_masks_at_original_size true \
#   --threshold 0.5





# DEFAULT_IMAGE_ROOT = "/home/jhshen/Experiment/Easi3R/data/davis/DAVIS/JPEGImages/480p/seq_name"
# DEFAULT_GT_ROOT = "/home/jhshen/Experiment/Easi3R/data/davis/DAVIS/Annotations/480p/seq_name"
# DEFAULT_OUR_ROOT = "/home/jhshen/Experiment/MyMotion/outputs/vggt_omega_v2/davis_sam/seq_name/final"
# DEFAULT_SEGANYMO_ROOT = "/home/jhshen/Experiment/SegAnyMo/result_v2/sam2/initial_preds/seq_name"
# DEFAULT_FLOWP_ROOT = "/home/jhshen/Experiment/MyMotion/evaluate/FlowP-SAM/dvs17/seq_name"

# CUDA_VISIBLE_DEVICES=2 python infer_vggt_omega_save_features.py \
#     --model_path "${MODEL_PATH}" \
#     --input_dir "${INPUT_DIR}" \
#     --output_root ./outputs_feature_vis \
#     --sequence_length 12 \
#     --save_features true \
#     --save_feature_pt true \
#     --save_feature_png true \
#     --save_feature_summary true \
#     --feature_summary_frames 3 \
#     --use_sam_refine false



CUDA_VISIBLE_DEVICES=1 python infer_vggt_omega.py \
    --model_path "${MODEL_PATH}" \
    --raft_model_path "${RAFT_MODEL_PATH}" \
    --device cuda \
    --params_only true