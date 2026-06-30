import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple
from vggt.models.vggt import VGGT
from vggt.layers.block import Block

class FlowEncoder(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, flow, target_h, target_w):
        feat = self.net(flow)
        return F.interpolate(feat, size=(target_h, target_w), mode='bilinear', align_corners=False)

class MotionSegmentationVGGT(nn.Module):
    def __init__(
        self, 
        vggt_model_path: Optional[str] = None, 
        img_size: int = 518, 
        patch_size: int = 14, 
        freeze_backbone: bool = True,
        selected_layers: List[int] = [0, 5, 15, 20, 46, 47],
        motion_decoder_depth: int = 5
    ):
        super().__init__()
        self.patch_size = patch_size
        self.selected_layers = selected_layers
        self.freeze_backbone = freeze_backbone
        
        # ================= 核心修改：开启 VGGT 的 Camera 和 Point 头 =================
        self.vggt_backbone = VGGT(
            img_size=img_size, patch_size=patch_size, 
            enable_camera=True,    # 开启位姿预测 (用于 Loss)
            enable_point=True,     # 开启3D点预测 (用于 Loss)
            enable_depth=False, enable_track=False
        )
        if vggt_model_path:
            self.vggt_backbone.load_state_dict(torch.load(vggt_model_path, map_location='cpu'), strict=False)
            
        if freeze_backbone:
            # 冻结 VGGT 所有的参数（包含新开启的 Camera Head 和 Point Head）
            for param in self.vggt_backbone.parameters():
                param.requires_grad = False
                
        self.layer_dim = 2 * self.vggt_backbone.aggregator.patch_embed.embed_dim
        # print("Layer dimension:", self.layer_dim) 2048
        
        # 你的 Motion 头架构
        self.feature_proj = nn.Sequential(
            nn.Linear(len(selected_layers) * self.layer_dim, 2048),
            nn.ReLU(inplace=True)
        )
        
        self.special_proj = nn.Sequential(nn.Linear(2 * self.layer_dim, 512), nn.GELU(), nn.Linear(512, 512))

        self.patch_proj_for_cam = nn.Sequential(nn.Linear(2 * self.layer_dim, 512), nn.GELU(), nn.Linear(512, 512))

        self.special_to_patch_attn = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)

        self.special_to_patch_norm = nn.LayerNorm(512)
        
        self.flow_encoder = FlowEncoder(output_dim=128)
        self.fusion_proj = nn.Linear(2048 + 512 + 128, 2048)
        
        self.decoder_embed_dim = 2048
        self.motion_decoder = nn.ModuleList([
            Block(dim=self.decoder_embed_dim, num_heads=16, mlp_ratio=4.0, qkv_bias=True, qk_norm=True) 
            for _ in range(motion_decoder_depth)
        ])
        self.motion_head = nn.Linear(self.decoder_embed_dim, 1)

    def forward(self, images, flows=None):
        B, S, C, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        
        # 1. 前向传播提取特征
        if self.freeze_backbone:
            with torch.no_grad():
                aggregated_tokens_list, patch_start_idx = self.vggt_backbone.aggregator(images)
                
                # 获取 Loss 需要的 3D 和 Camera 信息
                pose_enc_list = self.vggt_backbone.camera_head(aggregated_tokens_list)
                pose_enc = pose_enc_list[-1]
                
                world_points, world_points_conf = self.vggt_backbone.point_head(
                    aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
                )
        else:
            aggregated_tokens_list, patch_start_idx = self.vggt_backbone.aggregator(images)
            pose_enc_list = self.vggt_backbone.camera_head(aggregated_tokens_list)
            pose_enc = pose_enc_list[-1]
            world_points, world_points_conf = self.vggt_backbone.point_head(
                aggregated_tokens_list, images=images, patch_start_idx=patch_start_idx
            )

        # 2. 提取并融合特征
        selected_feats = [aggregated_tokens_list[min(i, len(aggregated_tokens_list)-1)] for i in self.selected_layers]
        
        # 将特征转为 float32 处理，防止混合精度导致特征投影不稳定
        multi_layer_feat = torch.cat(selected_feats, dim=-1).squeeze(0).float()  #[B*S, N, D]
        patch_token = multi_layer_feat[:, patch_start_idx:, :]
        special_token = multi_layer_feat[:, :patch_start_idx, :]
        
        special_tokens_proj = self.feature_proj(special_token)
        patch_tokens_proj = self.feature_proj(patch_token)
        
        feat_46 = aggregated_tokens_list[min(46, len(aggregated_tokens_list)-2)][:, :, :].squeeze(0).float()
        feat_47 = aggregated_tokens_list[min(47, len(aggregated_tokens_list)-1)][:, :, :].squeeze(0).float()
        cam_input = torch.cat([feat_46, feat_47], dim=-1)

        special_tokens = cam_input[:, :patch_start_idx, :]            # camera + register
        patch_tokens   = cam_input[:, patch_start_idx:, :]            # image patches

        special_feat = self.special_proj(special_tokens)              # [B*S, n_special, 512]
        patch_feat   = self.patch_proj_for_cam(patch_tokens)          # [B*S, n_patch, 512]

        attn_out, attn_weights = self.special_to_patch_attn(
            query=patch_feat,
            key=special_feat,
            value=special_feat
        )

        cam_patch = self.special_to_patch_norm(patch_feat + attn_out) # [B*S, n_patch, 512]
        
        if flows is not None:
            flow_patch = self.flow_encoder(flows.reshape(B*S, 1, H, W), patch_h, patch_w).flatten(2).transpose(1, 2)
        else:
            flow_patch = torch.zeros(B*S, patch_h * patch_w, 128, device=images.device)

        # sys.stderr.write("ENTER fusion block\n")
        # sys.stderr.write(f"patch_tokens_proj.shape = {patch_tokens_proj.shape}\n")
        # sys.stderr.write(f"cam_patch.shape         = {cam_patch.shape}\n")
        # sys.stderr.write(f"flow_patch.shape        = {flow_patch.shape}\n")
        # sys.stderr.flush()
        fused_patch = self.fusion_proj(torch.cat([patch_tokens_proj, cam_patch, flow_patch], dim=-1))
        
        # 3. Motion Decoder 交互
        hidden = torch.cat([special_tokens_proj, fused_patch], dim=1)
        for blk in self.motion_decoder:
            hidden = blk(hidden)
            
        # 4. 生成 Motion Mask (关闭 autocast 以保证 Sigmoid 精度)
        with torch.cuda.amp.autocast(enabled=False):
            hidden_for_head = hidden[:, patch_start_idx:, :].float()
            mask_logits = self.motion_head(hidden_for_head) # [B*S, H*W, 1]
            
            # Reshape & Interpolate
            mask_logits_2d = mask_logits.transpose(1, 2).reshape(B*S, 1, patch_h, patch_w)
            mask_logits_up = F.interpolate(
                mask_logits_2d, 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            )
            # motion_mask = torch.sigmoid(mask_logits_up).reshape(B, S, H, W)
            motion_logits = mask_logits_up.reshape(B, S, H, W)

        return {
            "motion_logits": motion_logits,
            "world_points": world_points.float(),      # 确保吐出的是 float32 供 Loss 计算
            "world_points_conf": world_points_conf.float(),
            "camera_poses": pose_enc.float()
        }



class CombinedLoss(nn.Module):
    """
    综合损失函数 (Combined Loss for Motion Segmentation)。
    结合了用于像素级分类的 BCE Loss、用于缓解类别不平衡的 Dice Loss / Focal Loss，
    以及基于 3D 几何先验的无监督运动一致性约束 (Motion Consistency Loss)。
    """
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, 
                 focal_weight: float = 0.0, motion_consistency_weight: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.motion_consistency_weight = motion_consistency_weight
        # self.bce_loss = nn.BCELoss(reduction='mean')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='mean')  # 如果 motion_logits 没有经过 Sigmoid 激活，使用 BCEWithLogitsLoss 更稳定
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """Dice loss for handling class imbalance"""
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice
    
    def focal_loss(self, pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
        """Focal loss for handling hard examples"""
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * bce_loss
        return focal_loss.mean()
    
    def motion_consistency_loss(self, pred_mask: torch.Tensor, world_points: torch.Tensor, 
                              world_points_conf: torch.Tensor, camera_poses: torch.Tensor) -> torch.Tensor:
        """
        基于 3D 物理法则的运动一致性约束。
        约束目标：预测的 Mask 应该等于 [物体的 3D 绝对运动] 剔除 [相机自身运动 (Ego-Motion)] 后的结果。
        """
        B, N, H, W = pred_mask.shape
        
        if N <= 1:
            return torch.tensor(0.0, device=pred_mask.device)
            
        # 统一置信度的形状到 [B, N, H, W]
        if world_points_conf.dim() == 5:
            world_points_conf = world_points_conf.squeeze(2)
        
        # 1. 计算 3D 物理世界中点的绝对运动 (Point Motion)
        point_motion = torch.zeros_like(world_points[..., 0])
        point_motion[:, 1:] = torch.norm(
            world_points[:, 1:] - world_points[:, :-1], dim=-1
        )
        
        # 2. 计算相机的 Ego-Motion
        # 兼容 VGGT 的 9D pose_enc [B, N, 9] (前6维旋转, 后3维平移 xyz)
        if camera_poses.shape[-1] == 9:
            translations = camera_poses[..., 6:9]  # [B, N, 3]
        # 兼容传统 4x4 变换矩阵 [B, N, 4, 4]
        elif camera_poses.shape[-1] == 4 and camera_poses.shape[-2] == 4:
            translations = camera_poses[:, :, :3, 3]
        else:
            raise ValueError(f"Unknown camera_poses shape: {camera_poses.shape}")

        camera_motion = torch.zeros(B, N, device=pred_mask.device)
        camera_motion[:, 1:] = torch.norm(
            translations[:, 1:] - translations[:, :-1], dim=-1
        )
        # 扩展到图像分辨率以便后续相除 [B, N, H, W]
        camera_motion = camera_motion.view(B, N, 1, 1).expand(-1, -1, H, W)
        
        # 3. 计算期望的物体运动
        expected_object_motion = point_motion / (camera_motion + 1e-6)
        expected_object_motion = torch.clamp(expected_object_motion, 0, 1)
        
        # 使用高置信度区域作为有效掩码
        valid_mask = world_points_conf > 0.5
        expected_object_motion = expected_object_motion * valid_mask
        
        # 4. L2 Loss 约束网络预测
        consistency_loss = F.mse_loss(
            pred_mask * valid_mask,
            expected_object_motion,
            reduction='mean'
        )
        
        return consistency_loss
    
    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        world_points: Optional[torch.Tensor] = None,
        world_points_conf: Optional[torch.Tensor] = None,
        camera_poses: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算运动分割的前向传播损失。

        Args:
            pred (torch.Tensor): 网络预测的运动掩码 (经过 Sigmoid 激活的概率图)。
                - 维度: [B, N, H, W]，其中 B 是 Batch Size，N 是视频帧数 (Sequence Length)。
                - 类型: torch.float32，数值范围 [0, 1]。
            
            target (torch.Tensor): 运动分割的 Ground Truth (真实标签)。
                - 维度: [B, N, H, W]。
                - 类型: torch.float32，数值通常为 0.0 (静止) 或 1.0 (运动)。
            
            world_points (torch.Tensor, optional): 当前像素对应的 3D 世界坐标系坐标点。
                用于计算 motion_consistency_loss 物理约束。
                - 维度: [B, N, H, W, 3]，最后一维对应 (X, Y, Z)。
                - 类型: torch.float32。
            
            world_points_conf (torch.Tensor, optional): 3D 坐标点的预测置信度。
                用于过滤掉不准确的 3D 点区域。
                - 维度: [B, N, H, W] 或 [B, N, 1, H, W]。
                - 类型: torch.float32，数值范围 [0, 1]。
            
            camera_poses (torch.Tensor, optional): 相机自身运动的位姿 (Camera Ego-Motion)。
                支持两种格式的输入：
                  1. VGGT 的 9D 编码: [B, N, 9] (前6维旋转，后3维平移)。
                  2. 标准 4x4 变换矩阵: [B, N, 4, 4]。
                - 类型: torch.float32。
                
            mask (torch.Tensor, optional): 有效像素掩码 (Valid Pixel Mask)。
                如果提供，将忽略 padding 或无效区域的 Loss 计算。
                - 维度: [B, N, H, W]。
                - 类型: torch.float32。

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]:
                - total_loss (torch.Tensor): 标量 Tensor，加权求和后的总损失，用于反向传播。
                - loss_dict (Dict): 记录各项独立损失值的字典，方便写入 TensorBoard 或终端打印。
        """
        if mask is not None:
            pred = pred * mask
            target = target * mask
        
        total_loss = 0
        loss_dict = {}
        
        # 1. 分类损失 (BCE)
        if self.bce_weight > 0:
            bce = self.bce_loss(pred, target)
            total_loss += self.bce_weight * bce
            loss_dict['bce_loss'] = bce.item()
        
        # 2. 形状/边缘损失 (Dice)
        if self.dice_weight > 0:
            dice = self.dice_loss(pred, target)
            total_loss += self.dice_weight * dice
            loss_dict['dice_loss'] = dice.item()
        
        # 3. 难样本挖掘损失 (Focal)
        if self.focal_weight > 0:
            focal = self.focal_loss(pred, target)
            total_loss += self.focal_weight * focal
            loss_dict['focal_loss'] = focal.item()
        
        # 4. 运动一致性物理约束损失 (Unsupervised Motion Consistency)
        if (self.motion_consistency_weight > 0 and 
            world_points is not None and world_points_conf is not None and camera_poses is not None):
            motion_consistency = self.motion_consistency_loss(
                pred, world_points, world_points_conf, camera_poses
            )
            total_loss += self.motion_consistency_weight * motion_consistency
            loss_dict['motion_consistency_loss'] = motion_consistency.item()
        
        return total_loss, loss_dict


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    """Compute IoU for motion segmentation from logits"""
    pred_prob = torch.sigmoid(pred)
    pred_binary = (pred_prob > threshold).float()

    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection

    iou = intersection / (union + 1e-6)
    return iou.item()

def compute_metrics(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    """Compute comprehensive metrics for motion segmentation from logits"""
    pred_prob = torch.sigmoid(pred)
    pred_binary = (pred_prob > threshold).float()

    # IoU
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    iou = intersection / (union + 1e-6)

    # Precision, Recall, F1
    tp = (pred_binary * target).sum()
    fp = (pred_binary * (1 - target)).sum()
    fn = ((1 - pred_binary) * target).sum()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    return {
        'iou': iou.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }