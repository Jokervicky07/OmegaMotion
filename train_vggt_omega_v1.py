import os
import argparse
import random
import socket
import csv
import logging

import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from vggt_omega.models.vggt_omega_seg_v1 import MotionSegmentationVGGTOmega, CombinedLoss
from core.utils.utils import load_config_file
# from motion_seg_inference import compute_optical_flow_magnitude_batch, initialize_raft_model
from core.utils.flow import compute_optical_flow_magnitude_batch, initialize_raft_model

from core.dataset.hoi4d import MotionSegmentationDataset
from core.dataset.dynamic_stereo import StereoMotionSegmentationDataset
from core.dataset.dynamicverse_aug import MotionSegmentationDatasetDynamic


torch.autograd.set_detect_anomaly(True)


DEFAULT_RAFT_MODEL_PATH = "./raft_large_C_T_SKHT_V2-ff5fadd5.pth"
DEFAULT_VGGT_OMEGA_MODEL_PATH = "checkpoints/vggt_omega_1b_512.pt"


class ProbabilisticDataset(Dataset):
    def __init__(self, datasets, probabilities):
        if len(datasets) != len(probabilities):
            raise ValueError("Number of datasets must match number of probabilities")

        self.datasets = datasets
        self.probabilities = probabilities
        self.cumulative_probs = [
            sum(probabilities[: i + 1])
            for i in range(len(probabilities))
        ]

    def __len__(self):
        return max(len(dataset) for dataset in self.datasets)

    def __getitem__(self, idx):
        rand_value = random.random()

        for i, cum_prob in enumerate(self.cumulative_probs):
            if rand_value <= cum_prob:
                selected_dataset = self.datasets[i]
                return selected_dataset[idx % len(selected_dataset)]

        return self.datasets[-1][idx % len(self.datasets[-1])]


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


def setup_distributed():
    if not dist.is_available() or not torch.cuda.is_available():
        return

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size == 1:
        return

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=rank,
    )

    print(
        f"Distributed training enabled: "
        f"rank {rank}, world_size {world_size}, local_rank {local_rank}"
    )


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def setup_logger(log_dir, rank=0):
    if rank != 0:
        return None

    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("motion_segmentation")
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    log_file = os.path.join(log_dir, "training.log")

    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def denormalize_image(
    img_tensor,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
):
    mean = torch.tensor(mean, device=img_tensor.device).view(3, 1, 1)
    std = torch.tensor(std, device=img_tensor.device).view(3, 1, 1)

    return img_tensor * std + mean


def save_visualization(
    images,
    predictions,
    ground_truth,
    save_path,
    max_samples=4,
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    batch_size = min(images.shape[0], max_samples)
    seq_len = images.shape[1]

    pred_prob = torch.sigmoid(predictions)

    for b in range(batch_size):
        fig, axes = plt.subplots(3, seq_len, figsize=(seq_len * 4, 12))

        if seq_len == 1:
            axes = axes.reshape(3, 1)

        for t in range(seq_len):
            img = denormalize_image(images[b, t]).detach().cpu().numpy()
            img = img.transpose(1, 2, 0)
            img = np.clip(img, 0.0, 1.0)

            gt_mask = ground_truth[b, t].detach().cpu().numpy()
            pred_mask = pred_prob[b, t].detach().cpu().numpy()

            axes[0, t].imshow(img)
            axes[0, t].axis("off")
            axes[0, t].set_title("Image")

            axes[1, t].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
            axes[1, t].axis("off")
            axes[1, t].set_title("GT")

            axes[2, t].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
            axes[2, t].axis("off")
            axes[2, t].set_title("Pred")

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()


def _build_dataset(dataset_name, data_dir, split, cfg):
    img_size = cfg.img_size
    sequence_length = cfg.sequence_length

    if dataset_name == "hoi4d":
        return MotionSegmentationDataset(
            data_dir=data_dir,
            split=split,
            img_size=img_size,
            sequence_length=sequence_length,
            sample_stride=1,
        )

    if dataset_name == "dynamic_stereo":
        return StereoMotionSegmentationDataset(
            data_dir=data_dir,
            split=split,
            img_size=img_size,
            sequence_length=sequence_length,
            sample_stride=4,
        )

    if dataset_name == "dynamicverse":
        return MotionSegmentationDatasetDynamic(
            data_dir=data_dir,
            split=split,
            img_size=img_size,
            sequence_length=sequence_length,
            sample_stride=1,
        )

    if dataset_name == "mose":
        return MotionSegmentationDatasetDynamic(
            data_dir=data_dir,
            split=split,
            img_size=img_size,
            sequence_length=sequence_length,
            sample_stride=1,
        )

    raise ValueError(f"Unknown dataset name: {dataset_name}")


def setup_motion_seg_dataset(cfg):
    train_datasets = []

    for i, dataset_name in enumerate(cfg.train_dataset):
        data_dir = cfg.train_root[i]

        train_datasets.append(
            _build_dataset(
                dataset_name=dataset_name,
                data_dir=data_dir,
                split="train",
                cfg=cfg,
            )
        )

    if len(train_datasets) == 1:
        combined_dataset = train_datasets[0]
    else:
        if (
            hasattr(cfg, "dataset_sampling_strategy")
            and cfg.dataset_sampling_strategy == "probabilistic"
        ):
            probabilities = getattr(
                cfg,
                "dataset_probabilities",
                [1.0 / len(train_datasets)] * len(train_datasets),
            )

            combined_dataset = ProbabilisticDataset(
                train_datasets,
                probabilities,
            )
        else:
            combined_dataset = ConcatDataset(train_datasets)

    test_loaders = {}

    for i, dataset_name in enumerate(cfg.test_dataset):
        data_dir = cfg.test_root[i]

        test_ds = StereoMotionSegmentationDataset(
            data_dir=cfg.test_root[i], 
            split='test', 
            img_size=cfg.img_size, 
            sequence_length=cfg.sequence_length, 
            sample_stride=4
        )

        test_loaders[dataset_name] = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=min(cfg.num_workers, 4),
            pin_memory=True,
        )

    return combined_dataset, test_loaders


def setup_motion_seg_model(cfg, device):
    omega_path = getattr(cfg, "vggt_omega_model_path", None) or os.environ.get(
        "VGGT_OMEGA_MODEL_PATH",
        DEFAULT_VGGT_OMEGA_MODEL_PATH,
    )

    raft_path = getattr(cfg, "raft_model_path", None) or os.environ.get(
        "RAFT_MODEL_PATH",
        DEFAULT_RAFT_MODEL_PATH,
    )

    selected_layer_names = getattr(cfg, "omega_selected_layer_names", None)

    model = MotionSegmentationVGGTOmega(
        checkpoint_path=omega_path,
        freeze_backbone=cfg.freeze_backbone,
        motion_decoder_depth=getattr(cfg, "motion_decoder_depth", 5),
        decoder_stride=getattr(cfg, "decoder_stride", 16),
        selected_layer_names=selected_layer_names,

        use_geo_feature=getattr(cfg, "use_geo_feature", True),
        use_cam_feature=getattr(cfg, "use_cam_feature", True),
        use_flow_feature=getattr(cfg, "use_flow_feature", True),
        use_depth_feature=getattr(cfg, "use_depth_feature", True),
    )

    if getattr(cfg, "use_flow_feature", True):
        initialize_raft_model(
            device=device,
            raft_model_path=raft_path,
        )

    return model


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    threshold: float = 0.5,
):
    pred_prob = torch.sigmoid(pred)
    pred_binary = (pred_prob > threshold).float()

    tp = (pred_binary * target).sum()
    fp = (pred_binary * (1.0 - target)).sum()
    fn = ((1.0 - pred_binary) * target).sum()

    iou = tp / (tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)

    return {
        "iou": iou.item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item(),
    }


def train_motion_seg_epoch(
    cfg,
    model,
    optimizer,
    train_loader,
    epoch,
    device,
    logger,
    scaler,
):
    model.train()

    criterion = CombinedLoss(
        bce_weight=cfg.bce_weight,
        dice_weight=cfg.dice_weight,
        motion_consistency_weight=getattr(cfg, "consistency_weight", 0.0),
    )

    running_loss = 0.0
    running_iou = 0.0
    running_f1 = 0.0
    valid_steps = 0

    is_main = (not dist.is_initialized()) or (dist.get_rank() == 0)

    for idx, sample in enumerate(train_loader):
        if sample is None:
            continue

        images = sample["images"].float().to(device, non_blocking=True)
        motion_gt = sample["motion_masks"].float().to(device, non_blocking=True)

        if getattr(cfg, "use_flow_feature", True):
            with torch.no_grad():
                flow_mags = compute_optical_flow_magnitude_batch(
                    images,
                    device,
                    bidirectional=True,
                )
        else:
            flow_mags = None

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=cfg.mixed_precision,
        ):
            output = model(images, flow_mags)
            motion_pred = output["motion_logits"]

            loss, loss_components = criterion(
                motion_pred,
                motion_gt,
            )

        scaler.scale(loss).backward()

        if cfg.gradient_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        valid_steps += 1

        with torch.no_grad():
            m = compute_metrics(motion_pred, motion_gt)
            running_iou += m["iou"]
            running_f1 += m["f1"]

        if is_main and idx % cfg.print_freq == 0:
            bce_val = loss_components.get("bce_loss", 0.0)
            dice_val = loss_components.get("dice_loss", 0.0)
            cons_val = loss_components.get("motion_consistency_loss", 0.0)

            logger.info(
                f"Epoch {epoch} [{idx}/{len(train_loader)}] "
                f"Loss: {loss.item():.4f} | "
                f"BCE: {bce_val:.4f} | "
                f"Dice: {dice_val:.4f} | "
                f"Consist: {cons_val:.4f} | "
                f"IoU: {m['iou']:.4f} | "
                f"F1: {m['f1']:.4f}"
            )

    denom = max(valid_steps, 1)

    return {
        "loss": running_loss / denom,
        "iou": running_iou / denom,
        "f1": running_f1 / denom,
    }


@torch.no_grad()
def test_motion_seg_epoch(
    cfg,
    model,
    test_loader,
    device,
    logger=None,
    vis_dir=None,
    epoch=0,
    dataset_name="test",
):
    model.eval()

    total_metrics = {
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
    }

    total_samples = 0

    for idx, sample in enumerate(test_loader):
        if sample is None:
            continue

        images = sample["images"].float().to(device, non_blocking=True)
        motion_gt = sample["motion_masks"].float().to(device, non_blocking=True)

        if getattr(cfg, "use_flow_feature", True):
            flow_mags = compute_optical_flow_magnitude_batch(
                images,
                device,
                bidirectional=True,
            )
        else:
            flow_mags = None

        with torch.amp.autocast(
            "cuda",
            dtype=torch.bfloat16,
            enabled=cfg.mixed_precision,
        ):
            output = model(images, flow_mags)
            motion_pred = output["motion_logits"]

        m = compute_metrics(motion_pred, motion_gt)

        bs = images.shape[0]

        for k in total_metrics:
            total_metrics[k] += m[k] * bs

        total_samples += bs

        if vis_dir is not None and idx < 3:
            save_path = os.path.join(
                vis_dir,
                f"{dataset_name}_epoch{epoch}_sample{idx}.png",
            )

            save_visualization(
                images=images,
                predictions=motion_pred,
                ground_truth=motion_gt,
                save_path=save_path,
            )

    if total_samples == 0:
        avg_metrics = {k: 0.0 for k in total_metrics}
    else:
        avg_metrics = {
            k: v / total_samples
            for k, v in total_metrics.items()
        }

    if logger is not None and ((not dist.is_initialized()) or dist.get_rank() == 0):
        logger.info(
            f"[{dataset_name}] Test IoU: {avg_metrics['iou']:.4f} | "
            f"Precision: {avg_metrics['precision']:.4f} | "
            f"Recall: {avg_metrics['recall']:.4f} | "
            f"F1: {avg_metrics['f1']:.4f}"
        )

    return avg_metrics


def save_training_log_csv(
    log_dir,
    epoch,
    train_stats,
    test_results,
    lr,
):
    csv_path = os.path.join(log_dir, "training_log.csv")

    row = {
        "epoch": epoch,
        "train_loss": train_stats["loss"],
        "train_iou": train_stats["iou"],
        "train_f1": train_stats["f1"],
        "lr": lr,
    }

    for name, res in test_results.items():
        for k, v in res.items():
            row[f"{name}_{k}"] = v

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def plot_training_curves(log_dir):
    csv_path = os.path.join(log_dir, "training_log.csv")

    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)

    if "train_loss" in df.columns:
        plt.plot(df["epoch"], df["train_loss"], label="train_loss")

    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)

    iou_cols = [c for c in df.columns if "iou" in c.lower()]

    for c in iou_cols:
        plt.plot(df["epoch"], df[c], label=c)

    plt.title("IoU")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "curves.png"))
    plt.close()


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]

    if len(batch) == 0:
        return None

    return torch.utils.data.default_collate(batch)


def main(cfg):
    setup_distributed()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    if rank == 0:
        os.makedirs(cfg.log_dir, exist_ok=True)

    logger = setup_logger(cfg.log_dir, rank)

    if rank == 0:
        logger.info("Starting VGGT-Omega motion segmentation training")
        logger.info(f"Device: {device}")
        logger.info(f"World size: {world_size}")
        logger.info(f"Image size: {cfg.img_size}")
        logger.info(f"Sequence length: {cfg.sequence_length}")
        logger.info(f"Mixed precision: {cfg.mixed_precision}")
        logger.info(f"Freeze backbone: {cfg.freeze_backbone}")
        logger.info(
            "Ablation features: "
            f"geo={getattr(cfg, 'use_geo_feature', True)}, "
            f"cam={getattr(cfg, 'use_cam_feature', True)}, "
            f"flow={getattr(cfg, 'use_flow_feature', True)}, "
            f"depth={getattr(cfg, 'use_depth_feature', True)}"
        )

    train_ds, test_loaders = setup_motion_seg_dataset(cfg)

    sampler = DistributedSampler(train_ds) if world_size > 1 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_skip_none,
    )

    model = setup_motion_seg_model(cfg, device).to(device)

    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=False,
        )

    original_model = model.module if world_size > 1 else model

    if cfg.freeze_backbone:
        trainable_params = [
            p
            for n, p in original_model.named_parameters()
            if "vggt_backbone" not in n and p.requires_grad
        ]
    else:
        trainable_params = [
            p for p in original_model.parameters() if p.requires_grad
        ]

    if rank == 0:
        num_trainable = sum(p.numel() for p in trainable_params)
        logger.info(f"Trainable parameters: {num_trainable / 1e6:.2f}M")

    lr = getattr(cfg, "lr", getattr(cfg, "learning_rate", 1e-4))

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=cfg.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.max_epochs,
        eta_min=cfg.min_lr,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.mixed_precision)

    start_epoch = 0
    best_iou = 0.0

    resume_model_path = getattr(cfg, "resume_model_path", None)

    if resume_model_path is not None and os.path.exists(resume_model_path):
        if rank == 0:
            logger.info(f"Loading model weights from {resume_model_path}")

        state_dict = torch.load(resume_model_path, map_location=device)
        original_model.load_state_dict(state_dict, strict=False)

    for epoch in range(start_epoch, cfg.max_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        train_stats = train_motion_seg_epoch(
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            epoch=epoch,
            device=device,
            logger=logger,
            scaler=scaler,
        )

        scheduler.step()

        if rank == 0:
            logger.info(
                f"Epoch {epoch} Train Loss: {train_stats['loss']:.4f} | "
                f"Train IoU: {train_stats['iou']:.4f} | "
                f"Train F1: {train_stats['f1']:.4f}"
            )

            test_results = {}

            vis_dir = os.path.join(cfg.log_dir, "vis")
            os.makedirs(vis_dir, exist_ok=True)

            for name, loader in test_loaders.items():
                test_results[name] = test_motion_seg_epoch(
                    cfg=cfg,
                    model=original_model,
                    test_loader=loader,
                    device=device,
                    logger=logger,
                    vis_dir=vis_dir,
                    epoch=epoch,
                    dataset_name=name,
                )

            save_training_log_csv(
                log_dir=cfg.log_dir,
                epoch=epoch,
                train_stats=train_stats,
                test_results=test_results,
                lr=optimizer.param_groups[0]["lr"],
            )

            plot_training_curves(cfg.log_dir)

            curr_iou = (
                max([r["iou"] for r in test_results.values()])
                if len(test_results) > 0
                else train_stats["iou"]
            )

            if curr_iou > best_iou:
                best_iou = curr_iou

                best_path = os.path.join(cfg.log_dir, "best_model.pth")
                torch.save(original_model.state_dict(), best_path)

                logger.info(f"Saved best model to {best_path}, IoU={best_iou:.4f}")

            latest_path = os.path.join(cfg.log_dir, "latest_model.pth")
            torch.save(original_model.state_dict(), latest_path)

    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")

    args = parser.parse_args()

    cfg = load_config_file(args.config_file)

    main(cfg)