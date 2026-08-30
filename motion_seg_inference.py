import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
import argparse
from glob import glob
import tempfile
import shutil
from typing import List, Dict, Tuple
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from core.utils.geometry import process_video_with_improved_sliding_window
from sam2.build_sam import build_sam2_video_predictor

DEFAULT_RAFT_MODEL_PATH = "./raft_large_C_T_SKHT_V2-ff5fadd5.pth"
DEFAULT_SAM2_CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_SAM2_CHECKPOINT_PATH = "./sam2-main/checkpoints/sam2.1_hiera_large.pt"


def initialize_raft_model(device, raft_model_path=None):
    """Initialize the RAFT optical-flow model."""
    global RAFT_MODEL, RAFT_TRANSFORMS
    if raft_model_path is None:
        raft_model_path = os.environ.get("RAFT_MODEL_PATH", DEFAULT_RAFT_MODEL_PATH)
    
    # Check for torchvision and optical flow models
    try:
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
    except ImportError:
        print("Torchvision optical flow models not found (requires torchvision >= 0.13).")
        print("Will fallback to OpenCV optical flow.")
        return False
        
    try:
        # If a specific path is provided, load from it
        if raft_model_path and os.path.exists(raft_model_path):
            print(f"Loading RAFT model from: {raft_model_path}")
            RAFT_MODEL = raft_large(weights=None, progress=False)
            state_dict = torch.load(raft_model_path, map_location=device)
            RAFT_MODEL.load_state_dict(state_dict)
            # When loading custom weights, transforms are often not included, so we create a basic one
            RAFT_TRANSFORMS = lambda img1, img2: (img1 * 2.0 - 1.0, img2 * 2.0 - 1.0)
        else:
            # Otherwise, load the default pre-trained model from torchvision
            print("Loading RAFT model with default torchvision weights...")
            weights = Raft_Large_Weights.DEFAULT
            RAFT_MODEL = raft_large(weights=weights, progress=False)
            RAFT_TRANSFORMS = weights.transforms()
        
        RAFT_MODEL = RAFT_MODEL.to(device).eval()
        print("RAFT model initialized successfully!")
        return True
        
    except Exception as e:
        print(f"Failed to initialize RAFT model: {e}")
        print("Will fallback to OpenCV optical flow.")
        RAFT_MODEL = None # Ensure model is None on failure
        return False

def compute_optical_flow_raft(frame1: torch.Tensor, frame2: torch.Tensor) -> torch.Tensor:
    """
    使用RAFT计算光流 (输入为 [0,1] 范围的Tensor)
    """
    global RAFT_MODEL, RAFT_TRANSFORMS
    import torchvision.transforms.functional as F
    
    # This function assumes RAFT_MODEL is not None
    original_H, original_W = frame1.shape[1], frame1.shape[2]
    
    if frame1.dim() == 3: frame1 = frame1.unsqueeze(0)
    if frame2.dim() == 3: frame2 = frame2.unsqueeze(0)
    
    H, W = frame1.shape[2], frame1.shape[3]
    new_H = ((H + 7) // 8) * 8
    new_W = ((W + 7) // 8) * 8
    
    img1_resized = F.resize(frame1, size=[new_H, new_W], antialias=False)
    img2_resized = F.resize(frame2, size=[new_H, new_W], antialias=False)
    
    img1_processed, img2_processed = RAFT_TRANSFORMS(img1_resized, img2_resized)
    
    with torch.no_grad():
        flow_predictions = RAFT_MODEL(img1_processed, img2_processed)
        flow_pred = flow_predictions[-1] 
    
    if flow_pred.shape[2] != original_H or flow_pred.shape[3] != original_W:
        
        flow_pred = F.resize(flow_pred, size=[original_H, original_W], antialias=False)
                

        scale_W = original_W / new_W
        scale_H = original_H / new_H
        flow_pred[:, 0] *= scale_W
        flow_pred[:, 1] *= scale_H


    return flow_pred.squeeze(0)

def denormalize_imagenet(tensor: torch.Tensor) -> torch.Tensor:
    """
    将ImageNet标准化的tensor恢复到[0,1]范围
    Args:
        tensor: [..., C, H, W] 标准化后的tensor
    Returns:
        denormalized tensor in [0,1] range
    """
    # ImageNet标准化参数
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    
    # 反标准化：x = x_norm * std + mean
    denormalized = tensor * std + mean
    
    # 确保在[0,1]范围内
    denormalized = torch.clamp(denormalized, 0.0, 1.0)
    
    return denormalized


def batch_fundamental_matrix_ransac(x1_batch, x2_batch, confidence=0.99, max_iters=2000):
    """
    批量RANSAC估计基础矩阵 - 使用OpenCV但优化调用
    Args:
        x1_batch: list of [N, 2] numpy arrays
        x2_batch: list of [N, 2] numpy arrays
    Returns:
        F_batch: [B, 3, 3] tensor
        inlier_ratios: [B] tensor
        valid_mask: [B] bool tensor
    """
    B = len(x1_batch)
    F_list = []
    ratio_list = []
    valid_list = []
    

    for x1, x2 in zip(x1_batch, x2_batch):
        if x1.shape[0] < 8:
            F_list.append(np.eye(3, dtype=np.float32))
            ratio_list.append(0.0)
            valid_list.append(False)
            continue
        
        F, mask = cv2.findFundamentalMat(
            x1, x2, 
            cv2.FM_RANSAC,  
            ransacReprojThreshold=3.0,
            confidence=confidence,
            maxIters=max_iters
        )
        
        if F is None or mask is None:
            F_list.append(np.eye(3, dtype=np.float32))
            ratio_list.append(0.0)
            valid_list.append(False)
        else:
            F_list.append(F.astype(np.float32))
            ratio_list.append(float(np.sum(mask)) / len(mask))
            valid_list.append(True)
    
    F_batch = torch.from_numpy(np.stack(F_list, axis=0))  # [B, 3, 3]
    ratios = torch.tensor(ratio_list)
    valid = torch.tensor(valid_list)
    
    return F_batch, ratios, valid


def compute_optical_flow_magnitude_batch_with_camera_comp(
    video_tensor: torch.Tensor, 
    device,
    pred_masks: torch.Tensor = None,
    ratio_thresh: float = 0.5,
    error_thresh_high_factor: float = 2.0,
    error_thresh_low_factor: float = 0.01,
    subsample_for_F: int = 4  
) -> tuple:
    """
    
    Args:
        video_tensor: [B, N, C, H, W] 视频tensor
        device: 设备
        pred_masks: [B, N, H, W] 可选的预测mask
        ratio_thresh: 基础矩阵内点比例阈值
        error_thresh_high_factor: 高误差阈值因子
        error_thresh_low_factor: 低误差阈值因子
        subsample_for_F: 点采样步长,越大越快但精度略降
        
    Returns:
        flow_magnitudes: [B, N, H, W] 原始光流幅度
        motion_masks_high: [B, N, H, W] 高置信度物体运动mask
        motion_masks_low: [B, N, H, W] 低误差(静态/相机运动)mask
        trusted_frames: list of (batch_idx, frame_idx) 可信任帧的索引
    """
    global RAFT_MODEL
    
    if RAFT_MODEL is None:
        print("RAFT model not initialized")
        B, N, _, H, W = video_tensor.shape
        return (torch.zeros(B, N, H, W, device=device),
                torch.zeros(B, N, H, W, device=device),
                torch.zeros(B, N, H, W, device=device),
                [])
    
    B, N, C, H, W = video_tensor.shape
    
    # 预分配所有输出张量
    flow_magnitudes = torch.zeros(B, N, H, W, device=device)
    motion_masks_high = torch.zeros(B, N, H, W, device=device)
    motion_masks_low = torch.zeros(B, N, H, W, device=device)
    trusted_frames = []
    
    if pred_masks is None:
        pred_masks = torch.ones(B, N, H, W, device=device)
    

    video_denorm = denormalize_imagenet(video_tensor)

    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device), 
        torch.arange(W, device=device), 
        indexing='ij'
    )
    grid = torch.stack([grid_y, grid_x], dim=0).float()  # [2, H, W]
    grid_normalized = torch.stack([
        2.0 * (grid[0] + 0.5) / H - 1,
        2.0 * (grid[1] + 0.5) / W - 1
    ], dim=0)  # [2, H, W]


    if subsample_for_F > 1:
        grid_sub = grid_normalized[:, ::subsample_for_F, ::subsample_for_F]  # [2, H', W']
        H_sub, W_sub = grid_sub.shape[1], grid_sub.shape[2]
    else:
        grid_sub = grid_normalized
        H_sub, W_sub = H, W
    

    print("Computing optical flows...")
    all_flows_fwd = {}  # {(b, t): flow}
    all_flows_bwd = {}
    
    
    for b in range(B):
        for t in range(N-1):
            frame_curr = video_denorm[b, t]
            frame_next = video_denorm[b, t+1]
            flow = compute_optical_flow_raft(frame_curr, frame_next)
            all_flows_fwd[(b, t)] = flow
    

    for b in range(B):
        for t in range(1, N):
            frame_curr = video_denorm[b, t]
            frame_prev = video_denorm[b, t-1]
            flow = compute_optical_flow_raft(frame_curr, frame_prev)
            all_flows_bwd[(b, t)] = flow
    
    print("Computing epipolar geometry constraints...")

    for t in range(N):
       
        batch_flows = []
        batch_has_fwd = []
        batch_has_bwd = []
        
        for b in range(B):
            flows = []
            has_fwd = (b, t) in all_flows_fwd
            has_bwd = (b, t) in all_flows_bwd
            
            if has_fwd:
                flows.append(all_flows_fwd[(b, t)])
            if has_bwd:
                flows.append(all_flows_bwd[(b, t)])
            
            batch_flows.append(flows)
            batch_has_fwd.append(has_fwd)
            batch_has_bwd.append(has_bwd)
        

        x1_batch_for_F = []
        x2_batch_for_F = []
        flow_idx_map = []  
        
        for batch_idx in range(B):
            if len(batch_flows[batch_idx]) == 0:
                continue
            
            for flow_idx, flow in enumerate(batch_flows[batch_idx]):
                
                normalized_flow = torch.stack([
                    2.0 * flow[0] / (H - 1),
                    2.0 * flow[1] / (W - 1)
                ], dim=0).flip(0)  # [2, H, W], flip to (x, y)
                
                
                if subsample_for_F > 1:
                    flow_sub = F.interpolate(
                        normalized_flow.unsqueeze(0), 
                        size=(H_sub, W_sub), 
                        mode='bilinear', 
                        align_corners=True
                    ).squeeze(0)
                else:
                    flow_sub = normalized_flow
                
                x2 = grid_sub + flow_sub  # [2, H', W']
                
                
                mask = pred_masks[batch_idx, t]
                if subsample_for_F > 1:
                    mask_sub = F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0).float(),
                        size=(H_sub, W_sub),
                        mode='nearest'
                    ).squeeze().bool()
                else:
                    mask_sub = mask.bool()
                
                x1_pts = grid_sub.permute(1, 2, 0).flip(-1).reshape(-1, 2)  # [H'*W', 2]
                x2_pts = x2.permute(1, 2, 0).flip(-1).reshape(-1, 2)
                
                valid = mask_sub.reshape(-1)
                x1_pts = x1_pts[valid].cpu().numpy()
                x2_pts = x2_pts[valid].cpu().numpy()
                
                x1_batch_for_F.append(x1_pts)
                x2_batch_for_F.append(x2_pts)
                flow_idx_map.append((batch_idx, flow_idx))
        

        if len(x1_batch_for_F) > 0:
            F_batch, ratios, valid = batch_fundamental_matrix_ransac(
                x1_batch_for_F, x2_batch_for_F
            )
            F_batch = F_batch.to(device)
            ratios = ratios.to(device)
            valid = valid.to(device)
            
            
            all_x1 = []
            all_x2 = []
            
            for (batch_idx, flow_idx) in flow_idx_map:
                flow = batch_flows[batch_idx][flow_idx]
                normalized_flow = torch.stack([
                    2.0 * flow[0] / (H - 1),
                    2.0 * flow[1] / (W - 1)
                ], dim=0).flip(0)
                
                x2 = grid_normalized + normalized_flow
                x1_flat = grid_normalized.permute(1, 2, 0).flip(-1).reshape(-1, 2)
                x2_flat = x2.permute(1, 2, 0).flip(-1).reshape(-1, 2)
                
                all_x1.append(x1_flat)
                all_x2.append(x2_flat)
            
            
            x1_stacked = torch.stack(all_x1, dim=0)  # [num_flows, H*W, 2]
            x2_stacked = torch.stack(all_x2, dim=0)
            
            errors_batch = compute_sampson_error_batch(
                x1_stacked, x2_stacked, F_batch
            )  # [num_flows, H*W]
            
            
            for idx, (batch_idx, flow_idx) in enumerate(flow_idx_map):
                if not valid[idx]:
                    continue
                
                err = errors_batch[idx].reshape(H, W)
                
                
                fac = (H + W) / 2
                err = err * fac**2
                err = torch.clamp(err, min=0)
                
                
                flow = batch_flows[batch_idx][flow_idx]
                flow_mag = torch.sqrt(flow[0]**2 + flow[1]**2)
                avg_flow_norm = flow_mag.mean()
                
                
                if flow_idx == 0:
                    flow_magnitudes[batch_idx, t] = flow_mag
                
                
                if flow_idx == 0:
                    err_combined = err
                else:
                    err_combined = torch.max(err_combined, err)
                
                
                if flow_idx == len(batch_flows[batch_idx]) - 1:
                    thresh_high = torch.clamp(
                        error_thresh_high_factor * avg_flow_norm, min=0.5
                    )
                    motion_masks_high[batch_idx, t] = (err_combined > thresh_high).float()
                    
                    
                    thresh_low = torch.clamp(
                        avg_flow_norm * error_thresh_low_factor, max=0.01
                    )
                    motion_masks_low[batch_idx, t] = (err_combined <= thresh_low).float()
                    
                    
                    all_valid = True
                    for fi in range(len(batch_flows[batch_idx])):
                        map_idx = flow_idx_map.index((batch_idx, fi))
                        if ratios[map_idx] <= ratio_thresh:
                            all_valid = False
                            break
                    
                    if all_valid and len(batch_flows[batch_idx]) > 0:
                        trusted_frames.append((batch_idx, t))
    
    return flow_magnitudes, motion_masks_high, motion_masks_low, trusted_frames


def compute_optical_flow_magnitude_batch(video_tensor: torch.Tensor, device, 
                                        bidirectional: bool = True,
                                        fusion_method: str = 'max',
                                        return_flow: bool = False):
    """
    为整个视频批次计算光流（支持返回向量）
    
    Args:
        video_tensor: [B, N, C, H, W] 视频tensor
        device: 设备
        bidirectional: 是否使用双向光流
        fusion_method: 双向光流融合方法
        return_flow: 是否返回光流向量
    
    Returns:
        如果 return_flow=False:
            flow_magnitudes: [B, N, H, W]
        如果 return_flow=True:
            (flow_vectors, flow_magnitudes)
            flow_vectors: [B, N, 2, H, W] 或 [B, N, H, W, 2]
            flow_magnitudes: [B, N, H, W]
    """
    global RAFT_MODEL
    
    if RAFT_MODEL is None:
        print("RAFT model not initialized, using zero flow")
        B, N, _, H, W = video_tensor.shape
        flow_magnitudes = torch.zeros(B, N, H, W, device=device)
        if return_flow:
            flow_vectors = torch.zeros(B, N, 2, H, W, device=device)
            return flow_vectors, flow_magnitudes
        return flow_magnitudes
    
    B, N, C, H, W = video_tensor.shape
    flow_magnitudes = torch.zeros(B, N, H, W, device=device)
    flow_vectors = torch.zeros(B, N, 2, H, W, device=device) if return_flow else None
    

    video_denorm = denormalize_imagenet(video_tensor)
    
    if not bidirectional:
        for b in range(B):
            for t in range(1, N):
                frame1 = video_denorm[b, t-1]
                frame2 = video_denorm[b, t]
                
                flow = compute_optical_flow_raft(frame1, frame2)  # [2, H, W]
                flow_magnitude = torch.sqrt(flow[0]**2 + flow[1]**2)
                
                flow_magnitudes[b, t] = flow_magnitude
                if return_flow:
                    flow_vectors[b, t] = flow  # [2, H, W]
    else:
        for b in range(B):
            forward_flows = []
            backward_flows = []
            forward_flow_vecs = [] if return_flow else None
            backward_flow_vecs = [] if return_flow else None
            
            for t in range(N-1):
                frame1 = video_denorm[b, t]
                frame2 = video_denorm[b, t+1]
                
                
                forward_flow = compute_optical_flow_raft(frame1, frame2)
                forward_magnitude = torch.sqrt(forward_flow[0]**2 + forward_flow[1]**2)
                forward_flows.append(forward_magnitude)
                if return_flow:
                    forward_flow_vecs.append(forward_flow)
                
                
                backward_flow = compute_optical_flow_raft(frame2, frame1)
                backward_magnitude = torch.sqrt(backward_flow[0]**2 + backward_flow[1]**2)
                backward_flows.append(backward_magnitude)
                if return_flow:
                    backward_flow_vecs.append(backward_flow)
            
            
            for t in range(N):
                flow_values = []
                flow_vecs = [] if return_flow else None
                
                if t > 0:
                    flow_values.append(backward_flows[t-1])
                    if return_flow:
                        flow_vecs.append(backward_flow_vecs[t-1])
                
                if t < N-1:
                    flow_values.append(forward_flows[t])
                    if return_flow:
                        flow_vecs.append(forward_flow_vecs[t])
                
                if len(flow_values) > 0:
                    flow_magnitudes[b, t] = fuse_flow_magnitudes(flow_values, fusion_method)
                    if return_flow:
                        # 融合向量：用 magnitude 做加权平均
                        if fusion_method == 'max':
                            max_idx = torch.argmax(torch.stack([v.max() for v in flow_values]))
                            flow_vectors[b, t] = flow_vecs[max_idx]
                        else:  # mean
                            flow_vectors[b, t] = torch.stack(flow_vecs).mean(dim=0)
    
    if return_flow:
        return flow_vectors, flow_magnitudes
    return flow_magnitudes


def compute_optical_flow_batch_xy(video_tensor: torch.Tensor, device, 
                                  bidirectional: bool = True,
                                  fusion_method: str = 'max') -> torch.Tensor:
    """
    为整个视频批次计算光流返回完整的x,y分量
    
    Args:
        video_tensor: [B, N, C, H, W] 视频tensor
        device: 设备
        bidirectional: 是否使用双向光流
        fusion_method: 双向光流融合方法
    
    Returns:
        flow_xy: [B, N, H, W, 2] 光流x,y分量
    """
    global RAFT_MODEL
    
    if RAFT_MODEL is None:
        print("RAFT model not initialized")
        B, N, _, H, W = video_tensor.shape
        return torch.zeros(B, N, H, W, 2, device=device)
    
    B, N, C, H, W = video_tensor.shape
    flow_xy = torch.zeros(B, N, H, W, 2, device=device)
    
    # 反标准化
    video_denorm = denormalize_imagenet(video_tensor)
    
    if not bidirectional:
        for b in range(B):
            for t in range(1, N):
                frame1 = video_denorm[b, t-1]
                frame2 = video_denorm[b, t]
                
                flow = compute_optical_flow_raft(frame1, frame2)
                flow_xy[b, t] = flow.permute(1, 2, 0)  # [H, W, 2]
    else:
        for b in range(B):
            forward_flows = []  # list of [H, W, 2]
            backward_flows = []
            
            for t in range(N-1):
                frame1 = video_denorm[b, t]
                frame2 = video_denorm[b, t+1]
                
                
                forward_flow = compute_optical_flow_raft(frame1, frame2)  # [2, H, W]
                forward_flows.append(forward_flow.permute(1, 2, 0))  # [H, W, 2]
                
            
                backward_flow = compute_optical_flow_raft(frame2, frame1)
                backward_flows.append(backward_flow.permute(1, 2, 0))  # [H, W, 2]
            
            
            for t in range(N):
                flow_values = []
                
                if t > 0:
                    flow_values.append(backward_flows[t-1])
                
                if t < N-1:
                    flow_values.append(forward_flows[t])
                
                if len(flow_values) > 0:
                    flow_xy[b, t] = fuse_flow_xy(flow_values, fusion_method)
    
    return flow_xy

@torch.no_grad()
def compute_optical_flow_xy_batch(
    video_tensor: torch.Tensor,
    device,
    bidirectional: bool = False,
    fusion_method: str = "max",
) -> torch.Tensor:
    """
    Wrapper for residual-flow training.

    This function returns RAFT vector flow with the shape and temporal alignment
    expected by vggt_omega_seg.py.

    Args:
        video_tensor:
            [B, N, C, H, W]

        device:
            CUDA / torch device

        bidirectional:
            For residual flow, use False.
            The residual needs a consistent forward direction t -> t+1.

        fusion_method:
            Kept for compatibility. Only relevant if bidirectional=True.

    Returns:
        flow_xy:
            [B, N, 2, H, W]

        flow_xy[:, t] is the forward flow from frame t -> frame t+1.
        The last frame copies the previous flow.
    """
    if bidirectional:
        print(
            "[Warning] compute_optical_flow_xy_batch was called with "
            "bidirectional=True. For camera-flow residual, bidirectional=False "
            "is recommended because residual subtraction needs a fixed direction."
        )

    # Existing function returns:
    #     [B, N, H, W, 2]
    #
    # In your current implementation with bidirectional=False:
    #     old_flow_xy[:, t] stores flow from frame t-1 -> t.
    old_flow_xy = compute_optical_flow_batch_xy(
        video_tensor=video_tensor,
        device=device,
        bidirectional=bidirectional,
        fusion_method=fusion_method,
    )
    # [B, N, H, W, 2]

    B, N, H, W, _ = old_flow_xy.shape

    old_flow_xy = old_flow_xy.permute(0, 1, 4, 2, 3).contiguous()
    # [B, N, 2, H, W]

    if not bidirectional:

        new_flow_xy = torch.zeros_like(old_flow_xy)

        if N > 1:
            new_flow_xy[:, :-1] = old_flow_xy[:, 1:]
            new_flow_xy[:, -1] = new_flow_xy[:, -2]
        else:
            new_flow_xy[:, 0] = old_flow_xy[:, 0]

        return new_flow_xy

    else:
        return old_flow_xy

def fuse_flow_xy(flow_list, method='max'):
    """
    融合多个光流x,y分量
    
    Args:
        flow_list: list of [H, W, 2]
        method: 融合方法
    
    Returns:
        fused_flow: [H, W, 2]
    """
    flow_stack = torch.stack(flow_list, dim=0)  # [N_flows, H, W, 2]
    flow_magnitude = torch.norm(flow_stack, dim=-1)  # [N_flows, H, W]
    
    if method == 'max':
        max_idx = torch.argmax(flow_magnitude, dim=0)  # [H, W]
        fused = flow_stack[max_idx, torch.arange(flow_stack.shape[1])[:, None], 
                          torch.arange(flow_stack.shape[2])]
    elif method == 'mean':
        fused = flow_stack.mean(dim=0)
    elif method == 'sum':
        fused = flow_stack.sum(dim=0)
    
    return fused

def bilinear_sample(tensor, h_coords, w_coords):
    """
    双线性插值采样
    
    Args:
        tensor: [H, W, C] 或 [B, H, W, C]
        h_coords: [H, W] 或 [B, H, W] 高度坐标（可以是浮点数）
        w_coords: [H, W] 或 [B, H, W] 宽度坐标
    
    Returns:
        sampled: [H, W, C] 或 [B, H, W, C]
    """
    # 处理维度
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
        h_coords = h_coords.unsqueeze(0)
        w_coords = w_coords.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    B, H, W, C = tensor.shape
    
    # 归一化坐标到[-1, 1]范围
    h_norm = 2.0 * h_coords / (H - 1) - 1.0
    w_norm = 2.0 * w_coords / (W - 1) - 1.0
    
    # 创建网格
    grid = torch.stack([w_norm, h_norm], dim=-1)  # [B, H, W, 2]
    
    # 使用grid_sample进行采样
    sampled = F.grid_sample(tensor.permute(0, 3, 1, 2),  # [B, C, H, W]
                           grid,
                           mode='bilinear',
                           padding_mode='border',
                           align_corners=True)
    
    sampled = sampled.permute(0, 2, 3, 1)  # [B, H, W, C]
    
    if squeeze_output:
        sampled = sampled.squeeze(0)
    
    return sampled

def compute_flow_consistency(flow_forward, flow_backward, H, W, threshold=1.0):
    """
    计算前后向光流的一致性掩码
    
    前向t -> t+1
    后向t+1 -> t
    
    如果像素(h,w)通过前向光流追踪到(h',w')
    然后从(h',w')通过后向光流追踪回来，
    如果回到的位置接近(h,w)，说明光流一致且可信
    
    Args:
        flow_forward: [H, W, 2] 前向光流 (t -> t+1)
        flow_backward: [H, W, 2] 后向光流 (t+1 -> t)
        H, W: 图像高宽
        threshold: float 一致性阈值
    
    Returns:
        consistency_mask: [H, W] 布尔掩码，True表示光流可信
        occlusion_mask: [H, W] 布尔掩码，True表示被遮挡的区域
    """
    device = flow_forward.device
    
    # 创建像素坐标网格
    h_grid, w_grid = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=device),
                                    torch.arange(W, dtype=torch.float32, device=device),
                                    indexing='ij')
    
    # 前向追踪
    h_forward = h_grid + flow_forward[..., 1]
    w_forward = w_grid + flow_forward[..., 0]
    
    # 限制边界
    h_forward = torch.clamp(h_forward, 0, H - 1)
    w_forward = torch.clamp(w_forward, 0, W - 1)
    
    # 采样后向光流
    flow_backward_tracked = bilinear_sample(flow_backward, h_forward, w_forward)  # [H, W, 2]
    
    # 后向追踪
    h_backward = h_forward + flow_backward_tracked[..., 1]
    w_backward = w_forward + flow_backward_tracked[..., 0]
    
    # 计算回到原始位置的偏差
    h_error = torch.abs(h_backward - h_grid)
    w_error = torch.abs(w_backward - w_grid)
    reprojection_error = torch.sqrt(h_error**2 + w_error**2)  # [H, W]
    
    # 一致性掩码：误差小于阈值的像素
    consistency_mask = reprojection_error < threshold  # [H, W]
    
    # 遮挡检测：边界外的像素视为被遮挡
    boundary_mask = (h_forward < 0) | (h_forward >= H) | (w_forward < 0) | (w_forward >= W)
    occlusion_mask = boundary_mask
    
    return consistency_mask, occlusion_mask, reprojection_error

def compute_motion_from_tracked_points(flow_xy, points_world_t, points_world_t1, 
                                       flow_backward=None, H=None, W=None,
                                       consistency_threshold=1.0):
    """
    用光流追踪像素，比较世界坐标的3D点运动
    仅在光流一致的区域计算运动
    
    Args:
        flow_xy: [H, W, 2] 或 [B, H, W, 2] 前向光流的x,y分量
        points_world_t: [H, W, 3] 或 [B, H, W, 3] t时刻的世界坐标
        points_world_t1: [H, W, 3] 或 [B, H, W, 3] t+1时刻的世界坐标
        flow_backward: [H, W, 2] 可选的后向光流，用于一致性检验
        H, W: 图像高宽
        consistency_threshold: float 一致性阈值
    
    Returns:
        point_motion: [H, W] 或 [B, H, W] 3D点的运动量（不可信区域为0）
        tracked_points: [H, W, 3] 或 [B, H, W, 3] 追踪到的t+1时刻的世界坐标
        confidence_mask: [H, W] 或 [B, H, W] 光流可信性掩码
    """
    device = flow_xy.device
    
    # 处理batch维度
    if flow_xy.dim() == 3:
        flow_xy = flow_xy.unsqueeze(0)
        points_world_t = points_world_t.unsqueeze(0)
        points_world_t1 = points_world_t1.unsqueeze(0)
        if flow_backward is not None:
            flow_backward = flow_backward.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    B = flow_xy.shape[0]
    if H is None or W is None:
        H, W = flow_xy.shape[1:3]
    
    # 创建像素坐标网格
    h_grid, w_grid = torch.meshgrid(torch.arange(H, dtype=torch.float32, device=device),
                                    torch.arange(W, dtype=torch.float32, device=device),
                                    indexing='ij')
    
    # 扩展到batch维度
    h_grid = h_grid.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    w_grid = w_grid.unsqueeze(0).expand(B, -1, -1)  # [B, H, W]
    
    # ===== 计算光流一致性 =====
    consistency_mask = torch.ones(B, H, W, dtype=torch.bool, device=device)
    occlusion_mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)
    
    if flow_backward is not None:
        for b in range(B):
            cons_mask, occ_mask, _ = compute_flow_consistency(
                flow_xy[b], flow_backward[b], H, W, 
                threshold=consistency_threshold
            )
            consistency_mask[b] = cons_mask
            occlusion_mask[b] = occ_mask
    
    # ===== 追踪位置 =====
    h_tracked = h_grid + flow_xy[..., 1]  # [B, H, W]
    w_tracked = w_grid + flow_xy[..., 0]  # [B, H, W]
    
    # 限制在图像边界内
    h_tracked = torch.clamp(h_tracked, 0, H - 1)
    w_tracked = torch.clamp(w_tracked, 0, W - 1)
    
    # 双线性插值获取追踪位置的世界坐标
    tracked_world_t1 = bilinear_sample(points_world_t1, h_tracked, w_tracked)  # [B, H, W, 3]
    
    # 计算3D点的运动向量
    point_motion_vec = tracked_world_t1 - points_world_t  # [B, H, W, 3]
    
    # 计算运动量（欧氏距离）
    point_motion = torch.norm(point_motion_vec, dim=-1)  # [B, H, W]
    
    # ===== 仅在可信区域保留运动信号 =====
    # 不可信的区域（遮挡、光流不一致）设为0
    confidence_mask = consistency_mask & (~occlusion_mask)  # [B, H, W]
    point_motion = point_motion * confidence_mask.float()
    
    if squeeze_output:
        point_motion = point_motion.squeeze(0)
        tracked_world_t1 = tracked_world_t1.squeeze(0)
        confidence_mask = confidence_mask.squeeze(0)
    
    return point_motion, tracked_world_t1, confidence_mask

def fuse_flow_magnitudes(flow_values: list, method: str = 'max') -> torch.Tensor:
    """
    融合多个光流幅度值
    Args:
        flow_values: 光流幅度值列表
        method: 融合方法
    Returns:
        融合后的光流幅度
    """
    if len(flow_values) == 1:
        return flow_values[0]
    
    if method == 'max':
        # 取最大值
        return torch.max(torch.stack(flow_values), dim=0)[0]
    elif method == 'mean':
        # 取平均值
        return torch.mean(torch.stack(flow_values), dim=0)
    elif method == 'sum':
        # 求和
        return torch.sum(torch.stack(flow_values), dim=0)
    elif method == 'weighted_mean':
        # 加权平均（给予时间上更近的帧更高权重）
        weights = torch.tensor([1.0] * len(flow_values), device=flow_values[0].device)
        weights = weights / weights.sum()
        weighted_flows = torch.stack([w * f for w, f in zip(weights, flow_values)])
        return torch.sum(weighted_flows, dim=0)
    else:
        raise ValueError(f"Unknown fusion method: {method}")

def compute_bidirectional_flow_with_temporal_consistency(video_tensor: torch.Tensor, device,
                                                       temporal_weight: float = 0.8) -> torch.Tensor:
    """
    计算具有时间一致性的双向光流
    Args:
        video_tensor: [B, N, C, H, W] 视频tensor
        device: 设备
        temporal_weight: 时间一致性权重，越大越平滑
    Returns:
        flow_magnitudes: [B, N, H, W] 光流幅度
    """
    # 先获得基础的双向光流
    base_flows = compute_optical_flow_magnitude_batch(
        video_tensor, device, bidirectional=True, fusion_method='mean'
    )
    
    B, N, H, W = base_flows.shape
    smoothed_flows = base_flows.clone()
    
    # 应用时间平滑
    for b in range(B):
        for t in range(1, N-1):  # 跳过边界帧
            # 时间平滑：当前帧 = (1-w) * 当前帧 + w * (前一帧 + 后一帧) / 2
            neighbor_mean = (smoothed_flows[b, t-1] + smoothed_flows[b, t+1]) / 2
            smoothed_flows[b, t] = (1 - temporal_weight) * base_flows[b, t] + \
                                 temporal_weight * neighbor_mean
    
    return smoothed_flows

def normalize_flow_magnitudes(flow_magnitudes: torch.Tensor, 
                            method='percentile', 
                            percentile=95, 
                            min_val=0.0, 
                            max_val=1.0) -> torch.Tensor:
    """
    归一化光流幅度
    Args:
        flow_magnitudes: [B, N, H, W] 光流幅度
        method: 归一化方法 ('minmax', 'percentile', 'zscore')
        percentile: 百分位数（method='percentile'时使用）
        min_val, max_val: 输出范围
    Returns:
        normalized_flow: [B, N, H, W] 归一化后的光流幅度
    """
    if method == 'minmax':
        # Min-Max归一化
        flow_min = flow_magnitudes.min()
        flow_max = flow_magnitudes.max()
        if flow_max > flow_min:
            normalized = (flow_magnitudes - flow_min) / (flow_max - flow_min)
        else:
            normalized = torch.zeros_like(flow_magnitudes)
            
    elif method == 'percentile':
        # 基于百分位数的归一化
        flow_percentile = torch.quantile(flow_magnitudes, percentile/100.0)
        normalized = torch.clamp(flow_magnitudes / (flow_percentile + 1e-8), 0, 1)
        
    elif method == 'zscore':
        # Z-score归一化后sigmoid映射到[0,1]
        flow_mean = flow_magnitudes.mean()
        flow_std = flow_magnitudes.std()
        if flow_std > 1e-8:
            zscore = (flow_magnitudes - flow_mean) / flow_std
            normalized = torch.sigmoid(zscore)
        else:
            normalized = torch.ones_like(flow_magnitudes) * 0.5
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # 缩放到指定范围
    normalized = normalized * (max_val - min_val) + min_val
    
    return normalized

def visualize_optical_flow(flow_magnitudes: torch.Tensor, 
                           video_tensor: torch.Tensor, 
                           device, 
                           alpha: float = 0.5, 
                           colormap: str = 'jet',
                           percentile_clip: float = 95.0,
                           gamma: float = 0.5) -> torch.Tensor:
    """
    为视频批次中的每一帧生成光流的可视化图像,并将其返回。
    
    Args:
        flow_magnitudes: [B, N, H, W] 光流幅度
        video_tensor: [B, N, C, H, W] 视频tensor (经过ImageNet标准化的)
        device: 设备
        alpha: 透明度值,范围 [0, 1]
        colormap: 使用的颜色映射方式
        percentile_clip: 百分位裁剪值,用于压缩动态范围 (建议85-98)
        gamma: gamma校正值,用于非线性压缩 (建议0.3-0.7,越小压缩越多)
    Returns:
        vis_images: [B, N, H, W, 3] 可视化的图像
    """
    B, N, H, W = flow_magnitudes.shape
    vis_images = torch.zeros(B, N, H, W, 3, device=device)  # 存储可视化图像,RGB通道

    # 反标准化视频
    video_denorm = denormalize_imagenet(video_tensor)  # [B, N, C, H, W]
    
    for b in range(B):
        for t in range(N):
            flow_magnitude = flow_magnitudes[b, t]  # [H, W]
            frame = video_denorm[b, t]  # [C, H, W], [0,1]

            flow_min = flow_magnitude.min()
            flow_max = torch.quantile(flow_magnitude.flatten(), percentile_clip / 100.0)
            
            # 裁剪并归一化
            flow_magnitude = torch.clamp(flow_magnitude, flow_min, flow_max)
            flow_magnitude = (flow_magnitude - flow_min) / (flow_max - flow_min + 1e-8)
            

            flow_magnitude = torch.pow(flow_magnitude, gamma)
            
            # 映射到0-255
            flow_magnitude = (flow_magnitude * 255).byte().cpu().numpy()  # [H, W]
            
            # 应用颜色映射
            flow_color = cv2.applyColorMap(flow_magnitude, cv2.COLORMAP_JET)  # [H, W, 3]
            
            # 将颜色映射与视频帧叠加
            frame = (frame.permute(1, 2, 0) * 255).byte().cpu().numpy()  # [H, W, C]
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 转为BGR格式
            
            # 使用透明度将光流可视化和原视频帧进行融合
            vis_frame = cv2.addWeighted(frame, 1 - alpha, flow_color, alpha, 0)
            
            # 将结果保存到vis_images
            vis_images[b, t] = torch.from_numpy(vis_frame).to(device)
    
    return vis_images
