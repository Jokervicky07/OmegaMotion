#!/bin/bash
# set -euo pipefail

CUDA_VISIBLE_DEVICES=1 torchrun --master-port=29502 --nproc_per_node=1 train_vggt_omega.py configs/vggt_motion_fusion_6layers_100frame.yaml
