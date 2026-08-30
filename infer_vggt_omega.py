import os
import re
import cv2
import json
import time
import shutil
import argparse
import tempfile
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as transforms

from vggt_omega.models.vggt_omega_seg import MotionSegmentationVGGTOmega
import motion_seg_inference
from motion_seg_inference import (
    initialize_raft_model,
    compute_optical_flow_magnitude_batch,
)


DEFAULT_RAFT_MODEL_PATH = "./raft_large_C_T_SKHT_V2-ff5fadd5.pth"
DEFAULT_SAM2_CONFIG_PATH = "configs/sam2.1/sam2.1_hiera_l.yaml"
DEFAULT_SAM2_CHECKPOINT_PATH = "./sam2-main/checkpoints/sam2.1_hiera_large.pt"

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP")


def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("true", "1", "yes", "y")


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_image_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(tuple(e.lower() for e in IMAGE_EXTS))
    ]
    return sorted(files, key=natural_key)


def discover_sequences(image_root: str) -> List[Tuple[str, str]]:
    """
    Returns:
        List of (sequence_name, sequence_frame_dir).

    Supports:
        1. image_root directly contains frames.
        2. image_root contains sequence subfolders.
    """
    direct_files = list_image_files(image_root)
    if len(direct_files) > 0:
        seq_name = os.path.basename(os.path.normpath(image_root))
        return [(seq_name, image_root)]

    sequences = []
    for name in sorted(os.listdir(image_root), key=natural_key):
        path = os.path.join(image_root, name)
        if not os.path.isdir(path):
            continue
        if len(list_image_files(path)) > 0:
            sequences.append((name, path))

    return sequences


def read_sequence_frames(seq_dir: str, max_frames: Optional[int] = None):
    frame_files = list_image_files(seq_dir)

    if max_frames is not None:
        frame_files = frame_files[:max_frames]

    frames = []
    frame_stems = []
    original_sizes = []

    for fname in frame_files:
        path = os.path.join(seq_dir, fname)
        img = Image.open(path).convert("RGB")
        frames.append(img)
        frame_stems.append(os.path.splitext(fname)[0])
        original_sizes.append(img.size)  # (W, H)

    return frames, frame_files, frame_stems, original_sizes


def save_mask_png(
    mask: np.ndarray,
    save_path: str,
    threshold: Optional[float] = None,
    resize_to: Optional[Tuple[int, int]] = None,
):
    """
    Args:
        mask: [H, W], soft mask in [0, 1].
        threshold:
            None: save soft grayscale mask.
            float: save binary mask.
        resize_to:
            Optional (W, H).
    """
    mask = np.asarray(mask)

    if mask.ndim == 3:
        mask = mask.squeeze()

    mask = np.nan_to_num(mask)
    mask = np.clip(mask, 0.0, 1.0)

    if resize_to is not None:
        mask = cv2.resize(mask, resize_to, interpolation=cv2.INTER_LINEAR)

    if threshold is None:
        out = (mask * 255.0).clip(0, 255).astype(np.uint8)
    else:
        out = (mask > threshold).astype(np.uint8) * 255

    ensure_dir(os.path.dirname(save_path))
    cv2.imwrite(save_path, out)


def save_sequence_predictions(
    masks: np.ndarray,
    frame_stems: List[str],
    output_dir: str,
    threshold: Optional[float] = None,
    resize_to_original: bool = False,
    original_sizes: Optional[List[Tuple[int, int]]] = None,
):
    ensure_dir(output_dir)

    for i, stem in enumerate(frame_stems):
        resize_to = None
        if resize_to_original and original_sizes is not None:
            resize_to = original_sizes[i]

        save_path = os.path.join(output_dir, f"{stem}.png")

        save_mask_png(
            mask=masks[i],
            save_path=save_path,
            threshold=threshold,
            resize_to=resize_to,
        )


def overlay_mask_on_image(
    image: Image.Image,
    mask: np.ndarray,
    threshold: float = 0.5,
    alpha: float = 0.5,
    color=(255, 0, 0),
) -> Image.Image:
    image = image.convert("RGB")
    img_np = np.array(image).astype(np.float32)
    W, H = image.size

    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask.squeeze()

    mask = np.nan_to_num(mask)
    mask = np.clip(mask, 0.0, 1.0)

    if mask.shape != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)

    binary = mask > threshold

    color_np = np.zeros_like(img_np)
    color_np[:, :, 0] = color[0]
    color_np[:, :, 1] = color[1]
    color_np[:, :, 2] = color[2]

    out = img_np.copy()
    out[binary] = img_np[binary] * (1.0 - alpha) + color_np[binary] * alpha
    out = np.clip(out, 0, 255).astype(np.uint8)

    return Image.fromarray(out)


def save_compare_visualization(
    frames: List[Image.Image],
    frame_stems: List[str],
    raw_masks: np.ndarray,
    final_masks: np.ndarray,
    vis_dir: str,
    threshold: float = 0.5,
    max_frames: int = 8,
    save_per_frame: bool = True,
):
    """
    Save:
        vis_dir/summary_compare.png
        vis_dir/frames/<frame>_compare.png

    Layout:
        Original | Raw mask | Raw overlay | Final/SAM mask | Final/SAM overlay
    """
    ensure_dir(vis_dir)

    if len(frames) == 0:
        return

    import matplotlib.pyplot as plt

    T = len(frames)
    indices = np.linspace(0, T - 1, min(max_frames, T), dtype=int)

    fig, axes = plt.subplots(5, len(indices), figsize=(3.2 * len(indices), 12))
    if len(indices) == 1:
        axes = axes.reshape(5, 1)

    for col, idx in enumerate(indices):
        img = frames[idx]
        W, H = img.size

        raw = raw_masks[idx]
        final = final_masks[idx]

        raw_resized = cv2.resize(raw, (W, H), interpolation=cv2.INTER_LINEAR)
        final_resized = cv2.resize(final, (W, H), interpolation=cv2.INTER_LINEAR)

        raw_overlay = overlay_mask_on_image(
            img,
            raw_resized,
            threshold=threshold,
        )
        final_overlay = overlay_mask_on_image(
            img,
            final_resized,
            threshold=threshold,
        )

        axes[0, col].imshow(img)
        axes[0, col].set_title(frame_stems[idx])
        axes[0, col].axis("off")

        axes[1, col].imshow(raw_resized, cmap="gray", vmin=0, vmax=1)
        axes[1, col].set_title("Before SAM")
        axes[1, col].axis("off")

        axes[2, col].imshow(raw_overlay)
        axes[2, col].set_title("Before overlay")
        axes[2, col].axis("off")

        axes[3, col].imshow(final_resized, cmap="gray", vmin=0, vmax=1)
        axes[3, col].set_title("After SAM / Final")
        axes[3, col].axis("off")

        axes[4, col].imshow(final_overlay)
        axes[4, col].set_title("After overlay")
        axes[4, col].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, "summary_compare.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    if not save_per_frame:
        return

    frame_vis_dir = os.path.join(vis_dir, "frames")
    ensure_dir(frame_vis_dir)

    for idx in indices:
        img = frames[idx]
        W, H = img.size

        raw = cv2.resize(raw_masks[idx], (W, H), interpolation=cv2.INTER_LINEAR)
        final = cv2.resize(final_masks[idx], (W, H), interpolation=cv2.INTER_LINEAR)

        raw_overlay = overlay_mask_on_image(img, raw, threshold=threshold)
        final_overlay = overlay_mask_on_image(img, final, threshold=threshold)

        fig, axes = plt.subplots(1, 5, figsize=(18, 4))

        axes[0].imshow(img)
        axes[0].set_title("Original")
        axes[0].axis("off")

        axes[1].imshow(raw, cmap="gray", vmin=0, vmax=1)
        axes[1].set_title("Before SAM")
        axes[1].axis("off")

        axes[2].imshow(raw_overlay)
        axes[2].set_title("Before overlay")
        axes[2].axis("off")

        axes[3].imshow(final, cmap="gray", vmin=0, vmax=1)
        axes[3].set_title("After SAM / Final")
        axes[3].axis("off")

        axes[4].imshow(final_overlay)
        axes[4].set_title("After overlay")
        axes[4].axis("off")

        plt.tight_layout()
        save_path = os.path.join(frame_vis_dir, f"{frame_stems[idx]}_compare.png")
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        plt.close(fig)


def import_sam2_video_predictor_builder():
    try:
        from sam2.build_sam import build_sam2_video_predictor
        return build_sam2_video_predictor
    except Exception:
        pass

    try:
        from sam2 import build_sam
        return build_sam.build_sam2_video_predictor
    except Exception as e:
        raise ImportError(
            "Cannot import SAM2 video predictor. Please check SAM2 installation."
        ) from e


def build_sam2_predictor(
    config_path: str,
    checkpoint_path: str,
    device: str = "cuda",
):
    builder = import_sam2_video_predictor_builder()
    return builder(config_path, checkpoint_path, device=device)


def preprocess_mask_for_sam(mask: np.ndarray, threshold: float = 0.8) -> torch.Tensor:
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask.squeeze()

    mask = np.nan_to_num(mask)
    mask = np.clip(mask, 0.0, 1.0)
    binary = mask > threshold

    return torch.from_numpy(binary.astype(np.bool_))


def refine_masks_with_sam2(
    frames: List[Image.Image],
    raw_masks: np.ndarray,
    predictor,
    seed_threshold: float = 0.8,
    device: str = "cuda",
) -> np.ndarray:
    """
    SAM2 refinement with odd/even propagation.

    Pass 1:
        seed odd frames, fill even frames.
    Pass 2:
        seed even frames, fill odd frames.

    If no valid SAM seed exists, fallback to raw mask.
    """
    if predictor is None:
        return raw_masks

    raw_masks = np.asarray(raw_masks).astype(np.float32)

    if raw_masks.ndim != 3:
        raise ValueError(f"raw_masks should be [T,H,W], got {raw_masks.shape}")

    T, mask_h, mask_w = raw_masks.shape

    tmp_dir = tempfile.mkdtemp()

    try:
        for i, img in enumerate(frames):
            img_resized = img.resize((mask_w, mask_h), Image.BILINEAR)
            arr = np.array(img_resized).astype(np.uint8)
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(tmp_dir, f"{i:05d}.jpg"), bgr)

        refined = np.copy(raw_masks)

        autocast_enabled = device.startswith("cuda")

        with torch.inference_mode():
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=autocast_enabled,
            ):
                inference_state = predictor.init_state(tmp_dir)

                for target_parity, seed_parity in [(0, 1), (1, 0)]:
                    predictor.reset_state(inference_state)

                    obj_id = 1
                    num_seeds = 0

                    for idx in range(T):
                        if idx % 2 != seed_parity:
                            continue

                        clean_mask = preprocess_mask_for_sam(
                            raw_masks[idx],
                            threshold=seed_threshold,
                        )

                        if clean_mask.sum().item() == 0:
                            continue

                        predictor.add_new_mask(
                            inference_state=inference_state,
                            frame_idx=idx,
                            obj_id=obj_id,
                            mask=clean_mask,
                        )
                        num_seeds += 1

                    if num_seeds == 0:
                        continue

                    video_segments = {}

                    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                        inference_state,
                        start_frame_idx=0,
                    ):
                        out_frame_idx = int(out_frame_idx)
                        video_segments[out_frame_idx] = {}

                        for i, out_obj_id in enumerate(out_obj_ids):
                            video_segments[out_frame_idx][int(out_obj_id)] = (
                                out_mask_logits[i] > 0.0
                            ).float()

                    for out_frame_idx in range(T):
                        if out_frame_idx % 2 != target_parity:
                            continue
                        if out_frame_idx not in video_segments:
                            continue
                        if obj_id not in video_segments[out_frame_idx]:
                            continue

                        m = video_segments[out_frame_idx][obj_id]
                        if m.ndim == 3:
                            m = m[0]

                        refined[out_frame_idx] = m.detach().cpu().numpy().astype(np.float32)

        return refined

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def parse_selected_layers(layer_string: Optional[str]) -> Optional[List[str]]:
    if layer_string is None:
        return None

    layer_string = layer_string.strip()
    if len(layer_string) == 0:
        return None

    return [x.strip() for x in layer_string.split(",") if x.strip()]

def sequence_already_done(seq_dir, output_root, seq_name, max_frames=None):
    _, _, frame_stems, _ = read_sequence_frames(seq_dir, max_frames=max_frames)

    final_dir = os.path.join(output_root, "final", seq_name)

    if not os.path.isdir(final_dir):
        return False

    for stem in frame_stems:
        pred_path = os.path.join(final_dir, f"{stem}.png")
        if not os.path.exists(pred_path):
            return False

    return True

class VGGTOmegaFullCheckpointInference:
    """
    Inference with full MotionSegmentationVGGTOmega checkpoint.

    The checkpoint is expected to contain:
        vggt_backbone.*
        feature_proj.*
        special_proj.*
        patch_proj_for_cam.*
        flow_encoder.*
        depth_mlp.*
        fusion_proj.*
        motion_decoder.*
        motion_head.*

    This matches training code:
        torch.save(original_model.state_dict(), best_path)
    """

    def __init__(
        self,
        model_path: str,
        raft_model_path: Optional[str] = None,
        img_size: int = 512,
        device: str = "cuda",
        mixed_precision: bool = True,
        freeze_backbone: bool = True,
        motion_decoder_depth: int = 5,
        decoder_stride: int = 16,
        selected_layer_names: Optional[List[str]] = None,
    ):
        self.model_path = model_path
        self.device = device
        self.img_size = img_size
        self.mixed_precision = mixed_precision

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        print("[Inference] Building MotionSegmentationVGGTOmega")
        print(f"[Inference] Full checkpoint: {model_path}")

        self.model = MotionSegmentationVGGTOmega(
            checkpoint_path=None,
            freeze_backbone=freeze_backbone,
            motion_decoder_depth=motion_decoder_depth,
            decoder_stride=decoder_stride,
            selected_layer_names=selected_layer_names,
        ).to(device)

        self._load_full_checkpoint(model_path)
        self.model.eval()

        if raft_model_path is None:
            raft_model_path = os.environ.get("RAFT_MODEL_PATH", DEFAULT_RAFT_MODEL_PATH)

        print(f"[Inference] Initializing RAFT: {raft_model_path}")
        initialize_raft_model(device=device, raft_model_path=raft_model_path)

    def _load_full_checkpoint(self, model_path: str):
        ckpt = torch.load(model_path, map_location=self.device)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"]
        else:
            state_dict = ckpt

        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            cleaned[k] = v

        has_backbone = any(k.startswith("vggt_backbone.") for k in cleaned.keys())
        if not has_backbone:
            raise ValueError(
                "This infer_vggt_omega.py expects a FULL checkpoint containing "
                "'vggt_backbone.*'. Your checkpoint looks like a head-only checkpoint. "
                "Please use the full best_model.pth saved by training."
            )

        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)

        print("[Inference] Loaded full checkpoint.")
        print(f"[Inference] Missing keys: {len(missing)}")
        print(f"[Inference] Unexpected keys: {len(unexpected)}")

        if len(missing) > 0:
            print("[Inference] Missing examples:", missing[:20])
        if len(unexpected) > 0:
            print("[Inference] Unexpected examples:", unexpected[:20])

    def preprocess_frames(self, frames: List[Image.Image]) -> torch.Tensor:
        tensors = []

        for img in frames:
            img = img.convert("RGB")
            img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
            x = transforms.ToTensor()(img)
            x = self.normalize(x)
            tensors.append(x)

        video = torch.stack(tensors, dim=0).unsqueeze(0)
        return video

    @torch.no_grad()
    def predict_chunk(self, frames: List[Image.Image]) -> np.ndarray:
        """
        Returns:
            prob: [T, img_size, img_size], soft mask in [0,1].
        """
        if len(frames) == 0:
            return np.zeros((0, self.img_size, self.img_size), dtype=np.float32)

        video_tensor = self.preprocess_frames(frames).to(self.device)

        flow_mags = compute_optical_flow_magnitude_batch(
            video_tensor,
            self.device,
            bidirectional=True,
        )

        autocast_enabled = self.mixed_precision and self.device.startswith("cuda")

        with torch.amp.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=autocast_enabled,
        ):
            output = self.model(video_tensor, flow_mags)
            logits = output["motion_logits"]

        prob = torch.sigmoid(logits).squeeze(0)
        prob = prob.detach().float().cpu().numpy()

        return prob

    @torch.no_grad()
    def predict_sequence(
        self,
        frames: List[Image.Image],
        sequence_length: int = 16,
    ) -> np.ndarray:
        """
        Predict a full sequence.

        For normal chunks:
            use non-overlapping chunks.

        For the last incomplete chunk:
            use a full-length tail window as temporal context,
            then only keep the predictions for the remaining frames.

        Example:
            T=65, sequence_length=32

            chunk 1: frames[0:32]   -> keep 32 preds
            chunk 2: frames[32:64]  -> keep 32 preds
            tail:    frames[33:65]  -> keep only last 1 pred
        """
        T = len(frames)

        if T == 0:
            return np.zeros((0, self.img_size, self.img_size), dtype=np.float32)

        if T <= sequence_length:
            return self.predict_chunk(frames)

        all_masks = []

        num_full_chunks = T // sequence_length
        remainder = T % sequence_length

        # 1. Predict full non-overlapping chunks.
        for chunk_idx in tqdm(range(num_full_chunks), desc="Predict full chunks", leave=False):
            start = chunk_idx * sequence_length
            end = start + sequence_length

            chunk = frames[start:end]
            masks = self.predict_chunk(chunk)

            all_masks.append(masks)

        # 2. Handle the remaining tail frames with full temporal context.
        if remainder > 0:
            tail_start = T - sequence_length
            tail_end = T

            tail_chunk = frames[tail_start:tail_end]
            tail_masks = self.predict_chunk(tail_chunk)

            # Keep only the predictions corresponding to the remaining frames.
            tail_keep = tail_masks[-remainder:]

            all_masks.append(tail_keep)

        return np.concatenate(all_masks, axis=0)


def process_single_sequence(
    inference: VGGTOmegaFullCheckpointInference,
    seq_name: str,
    seq_dir: str,
    output_root: str,
    sequence_length: int,
    max_frames: Optional[int],
    use_sam_refine: bool,
    sam_predictor,
    threshold: float,
    sam_seed_threshold: float,
    save_raw: bool,
    save_sam: bool,
    save_final: bool,
    save_vis: bool,
    save_per_frame_vis: bool,
    save_masks_at_original_size: bool,
):
    print(f"\n=== Processing sequence: {seq_name} ===")
    print(f"Input: {seq_dir}")

    frames, frame_files, frame_stems, original_sizes = read_sequence_frames(
        seq_dir,
        max_frames=max_frames,
    )

    if len(frames) == 0:
        print(f"[Warning] No frames in {seq_dir}, skip.")
        return

    t0 = time.time()

    raw_masks = inference.predict_sequence(
        frames=frames,
        sequence_length=sequence_length,
    )

    if use_sam_refine:
        print("[SAM2] Refining raw masks...")
        sam_masks = refine_masks_with_sam2(
            frames=frames,
            raw_masks=raw_masks,
            predictor=sam_predictor,
            seed_threshold=sam_seed_threshold,
            device=inference.device,
        )
        final_masks = sam_masks
    else:
        sam_masks = None
        final_masks = raw_masks

    t1 = time.time()

    if save_raw:
        raw_dir = os.path.join(output_root, "raw", seq_name)
        save_sequence_predictions(
            masks=raw_masks,
            frame_stems=frame_stems,
            output_dir=raw_dir,
            threshold=None,
            resize_to_original=save_masks_at_original_size,
            original_sizes=original_sizes,
        )

    if save_sam and sam_masks is not None:
        sam_dir = os.path.join(output_root, "sam", seq_name)
        save_sequence_predictions(
            masks=sam_masks,
            frame_stems=frame_stems,
            output_dir=sam_dir,
            threshold=None,
            resize_to_original=save_masks_at_original_size,
            original_sizes=original_sizes,
        )

    if save_final:
        final_dir = os.path.join(output_root, "final", seq_name)
        save_sequence_predictions(
            masks=final_masks,
            frame_stems=frame_stems,
            output_dir=final_dir,
            threshold=threshold,
            resize_to_original=save_masks_at_original_size,
            original_sizes=original_sizes,
        )

    if save_vis:
        vis_dir = os.path.join(output_root, "vis", seq_name)
        save_compare_visualization(
            frames=frames,
            frame_stems=frame_stems,
            raw_masks=raw_masks,
            final_masks=final_masks,
            vis_dir=vis_dir,
            threshold=threshold,
            max_frames=8,
            save_per_frame=save_per_frame_vis,
        )

    summary = {
        "sequence_name": seq_name,
        "num_frames": len(frames),
        "use_sam_refine": bool(use_sam_refine),
        "raw_motion_ratio": float((raw_masks > threshold).mean()),
        "final_motion_ratio": float((final_masks > threshold).mean()),
        "time_seconds": float(t1 - t0),
    }

    summary_dir = os.path.join(output_root, "vis", seq_name)
    ensure_dir(summary_dir)
    with open(os.path.join(summary_dir, "inference_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"[{seq_name}] frames={len(frames)}, "
        f"raw_motion_ratio={summary['raw_motion_ratio']:.2%}, "
        f"final_motion_ratio={summary['final_motion_ratio']:.2%}, "
        f"time={summary['time_seconds']:.2f}s"
    )

def print_parameter_statistics(model):
    """
    Print parameter statistics for OmegaMotion + RAFT.
    """

    def count_params(module):
        if module is None:
            return 0, 0

        total = sum(p.numel() for p in module.parameters())
        trainable = sum(
            p.numel()
            for p in module.parameters()
            if p.requires_grad
        )
        return total, trainable

    print("\n")
    print("=" * 95)
    print("MODEL PARAMETER STATISTICS")
    print("=" * 95)

    # ---------------------------------------------------------
    # OmegaMotion model
    # ---------------------------------------------------------
    omega_total, omega_trainable = count_params(model)
    omega_frozen = omega_total - omega_trainable

    print("\n[OmegaMotion]")
    print(f"Total parameters     : {omega_total:,}")
    print(f"Total parameters     : {omega_total / 1e6:.2f} M")
    print(f"Trainable parameters : {omega_trainable:,}")
    print(f"Trainable parameters : {omega_trainable / 1e6:.2f} M")
    print(f"Frozen parameters    : {omega_frozen:,}")
    print(f"Frozen parameters    : {omega_frozen / 1e6:.2f} M")

    # ---------------------------------------------------------
    # Submodules
    # ---------------------------------------------------------
    print("\n" + "-" * 95)
    print(
        f"{'Module':32s}"
        f"{'Total Params':>20s}"
        f"{'Trainable Params':>22s}"
    )
    print("-" * 95)

    module_names = [
        "vggt_backbone",
        "feature_proj",
        "special_proj",
        "patch_proj_for_cam",
        "flow_encoder",
        "depth_mlp",
        "fusion_proj",
        "motion_decoder",
        "motion_head",
    ]

    for name in module_names:
        if not hasattr(model, name):
            print(
                f"{name:32s}"
                f"{'N/A':>20s}"
                f"{'N/A':>22s}"
            )
            continue

        module = getattr(model, name)

        total, trainable = count_params(module)

        print(
            f"{name:32s}"
            f"{total / 1e6:17.2f} M"
            f"{trainable / 1e6:19.2f} M"
        )

    # ---------------------------------------------------------
    # RAFT
    # ---------------------------------------------------------
    raft_model = getattr(
        motion_seg_inference,
        "RAFT_MODEL",
        None,
    )

    raft_total, raft_trainable = count_params(raft_model)

    print("-" * 95)

    if raft_model is not None:
        print(
            f"{'RAFT':32s}"
            f"{raft_total / 1e6:17.2f} M"
            f"{raft_trainable / 1e6:19.2f} M"
        )
    else:
        print(
            f"{'RAFT':32s}"
            f"{'NOT LOADED':>20s}"
            f"{'-':>22s}"
        )

    # ---------------------------------------------------------
    # Full inference pipeline
    # ---------------------------------------------------------
    full_total = omega_total + raft_total

    # RAFT is used as frozen external optical-flow model.
    # For paper reporting, OmegaMotion trainable params are
    # determined by the segmentation model itself.
    full_trainable = omega_trainable

    full_frozen = full_total - full_trainable

    print("=" * 95)

    print("\n[Full Inference Pipeline: OmegaMotion + RAFT]")

    print(
        f"Total parameters     : "
        f"{full_total:,}"
    )
    print(
        f"Total parameters     : "
        f"{full_total / 1e6:.2f} M"
    )

    print(
        f"Trainable parameters : "
        f"{full_trainable:,}"
    )
    print(
        f"Trainable parameters : "
        f"{full_trainable / 1e6:.2f} M"
    )

    print(
        f"Frozen parameters    : "
        f"{full_frozen:,}"
    )
    print(
        f"Frozen parameters    : "
        f"{full_frozen / 1e6:.2f} M"
    )

    print("\n[Paper-friendly summary]")

    print(
        f"OmegaMotion model         : "
        f"{omega_total / 1e6:.2f} M"
    )

    if hasattr(model, "vggt_backbone"):
        backbone_total, _ = count_params(
            model.vggt_backbone
        )

        added_params = omega_total - backbone_total

        print(
            f"VGGT-Omega backbone       : "
            f"{backbone_total / 1e6:.2f} M"
        )

        print(
            f"Motion-specific modules   : "
            f"{added_params / 1e6:.2f} M"
        )

    print(
        f"RAFT                      : "
        f"{raft_total / 1e6:.2f} M"
    )

    print(
        f"Full pipeline             : "
        f"{full_total / 1e6:.2f} M"
    )

    print(
        f"Trainable parameters      : "
        f"{omega_trainable / 1e6:.2f} M"
    )

    print("=" * 95)

def main():
    parser = argparse.ArgumentParser("VGGT-Omega motion segmentation inference with full checkpoint")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Full MotionSegmentationVGGTOmega checkpoint, e.g. best_model.pth.")
    parser.add_argument("--raft_model_path", type=str, default=None)

    parser.add_argument("--input_dir", type=str, default=None,
                        help="Single sequence directory containing frames.")
    parser.add_argument("--image_root", type=str, default=None,
                        help="Dataset root containing sequence folders, or a direct frame folder.")

    parser.add_argument("--output_root", type=str, default="./outputs",
                        help="Output root directory for predictions and visualizations.")

    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--sequence_length", type=int, default=16)
    parser.add_argument("--max_frames", type=int, default=None)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mixed_precision", type=str2bool, default=True)
    parser.add_argument("--freeze_backbone", type=str2bool, default=True)
    parser.add_argument("--motion_decoder_depth", type=int, default=5)
    parser.add_argument("--decoder_stride", type=int, default=16)
    parser.add_argument("--selected_layer_names", type=str, default=None,
                        help="Optional comma-separated hook layer names. Must match training if changed.")

    parser.add_argument("--use_sam_refine", type=str2bool, default=False)
    parser.add_argument("--sam2_config_path", type=str, default=DEFAULT_SAM2_CONFIG_PATH)
    parser.add_argument("--sam2_checkpoint_path", type=str, default=DEFAULT_SAM2_CHECKPOINT_PATH)
    parser.add_argument("--sam_seed_threshold", type=float, default=0.8)

    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Only used for visualization and motion-ratio logging.")

    parser.add_argument("--save_raw", type=str2bool, default=True)
    parser.add_argument("--save_sam", type=str2bool, default=True)
    parser.add_argument("--save_final", type=str2bool, default=True)
    parser.add_argument("--save_vis", type=str2bool, default=True)
    parser.add_argument("--save_per_frame_vis", type=str2bool, default=True)
    parser.add_argument("--skip_existing", type=str2bool, default=True)
    parser.add_argument("--save_masks_at_original_size", type=str2bool, default=False,
                        help="If False, masks are saved at model output size. "
                             "evaluate_predictions.py will resize to GT size automatically.")
    parser.add_argument(
        "--params_only",
        type=str2bool,
        default=False,
        help="Load models, print parameter statistics, and exit without inference.",
    )


    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("[Warning] CUDA not available, switching to CPU.")
        args.device = "cpu"

    if not args.params_only:
        if args.input_dir is None and args.image_root is None:
            raise ValueError("Please provide either --input_dir or --image_root.")

    ensure_dir(args.output_root)

    selected_layer_names = parse_selected_layers(args.selected_layer_names)

    inference = VGGTOmegaFullCheckpointInference(
        model_path=args.model_path,
        raft_model_path=args.raft_model_path,
        img_size=args.img_size,
        device=args.device,
        mixed_precision=args.mixed_precision,
        freeze_backbone=args.freeze_backbone,
        motion_decoder_depth=args.motion_decoder_depth,
        decoder_stride=args.decoder_stride,
        selected_layer_names=selected_layer_names,
    )

    if args.params_only:
        print_parameter_statistics(inference.model)
        return

    sam_predictor = None
    if args.use_sam_refine:
        print("[SAM2] Building SAM2 video predictor...")
        sam_predictor = build_sam2_predictor(
            config_path=args.sam2_config_path,
            checkpoint_path=args.sam2_checkpoint_path,
            device=args.device,
        )

    if args.input_dir is not None:
        seq_name = os.path.basename(os.path.normpath(args.input_dir))
        sequences = [(seq_name, args.input_dir)]
    else:
        sequences = discover_sequences(args.image_root)

    print(f"[Inference] Found {len(sequences)} sequence(s).")

    for seq_name, seq_dir in tqdm(sequences, desc="Sequences"):
        if args.skip_existing and sequence_already_done(
            seq_dir=seq_dir,
            output_root=args.output_root,
            seq_name=seq_name,
            max_frames=args.max_frames,
        ):
            print(f"[Skip] {seq_name}: final predictions already complete.")
            continue

        process_single_sequence(
            inference=inference,
            seq_name=seq_name,
            seq_dir=seq_dir,
            output_root=args.output_root,
            sequence_length=args.sequence_length,
            max_frames=args.max_frames,
            use_sam_refine=args.use_sam_refine,
            sam_predictor=sam_predictor,
            threshold=args.threshold,
            sam_seed_threshold=args.sam_seed_threshold,
            save_raw=args.save_raw,
            save_sam=args.save_sam,
            save_final=args.save_final,
            save_vis=args.save_vis,
            save_per_frame_vis=args.save_per_frame_vis,
            save_masks_at_original_size=args.save_masks_at_original_size,
        )

    print("\nInference completed.")
    print(f"Output root: {args.output_root}")
    print(f"Raw predictions:   {os.path.join(args.output_root, 'raw')}")
    print(f"SAM predictions:   {os.path.join(args.output_root, 'sam')}")
    print(f"Final predictions: {os.path.join(args.output_root, 'final')}")


if __name__ == "__main__":
    main()