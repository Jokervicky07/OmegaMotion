#!/bin/bash
# set -euo pipefail

CUDA_VISIBLE_DEVICES=1,2 torchrun --master-port=29501 --nproc_per_node=2 train_vggt_omega_v1.py configs/vggt_motion_fusion_6layers_12frame.yaml
