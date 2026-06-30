#!/bin/bash
# set -euo pipefail

CUDA_VISIBLE_DEVICES=2,3 torchrun --master-port=29502 --nproc_per_node=2 train_vggt_omega.py configs/vggt_motion_fusion_6layers_100frame.yaml
