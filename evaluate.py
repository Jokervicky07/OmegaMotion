import os
import re
import json
import argparse
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


MASK_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG", ".BMP")


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def list_mask_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = [
        f for f in os.listdir(folder)
        if f.lower().endswith(tuple(e.lower() for e in MASK_EXTS))
    ]
    return sorted(files, key=natural_key)


def make_file_map(folder: str) -> Dict[str, str]:
    files = list_mask_files(folder)
    mp = {}
    for f in files:
        stem = os.path.splitext(f)[0]
        mp[stem] = os.path.join(folder, f)
    return mp


def root_has_direct_masks(root: str) -> bool:
    return len(list_mask_files(root)) > 0


def discover_sequences(gt_root: str) -> List[str]:
    """
    Supports:
        1. gt_root/seq_name/*.png
        2. gt_root directly contains masks.
    """
    if root_has_direct_masks(gt_root):
        return [os.path.basename(os.path.normpath(gt_root))]

    seqs = []
    for name in sorted(os.listdir(gt_root), key=natural_key):
        path = os.path.join(gt_root, name)
        if not os.path.isdir(path):
            continue
        if len(list_mask_files(path)) > 0:
            seqs.append(name)

    return seqs


def get_sequence_dir(root: str, seq_name: str) -> str:
    if root_has_direct_masks(root):
        return root
    return os.path.join(root, seq_name)


def read_mask(path: str, non_black_as_foreground: bool = False) -> np.ndarray:
    img = Image.open(path)

    if non_black_as_foreground:
        arr = np.array(img.convert("RGB"))

        # Any non-black pixel is foreground.
        mask = np.any(arr > 0, axis=-1).astype(np.float32)

        return mask

    mask = np.array(img.convert("L")).astype(np.float32)

    if mask.max() > 1.0:
        mask = mask / 255.0

    mask = np.nan_to_num(mask)
    mask = np.clip(mask, 0.0, 1.0)

    return mask


def resize_pred_to_gt(pred_mask: np.ndarray, gt_shape: Tuple[int, int]) -> np.ndarray:
    """
    Resize soft prediction mask to GT size.

    Args:
        pred_mask: [H,W], soft mask in [0,1].
        gt_shape: (gt_h, gt_w).
    """
    gt_h, gt_w = gt_shape

    if pred_mask.shape == (gt_h, gt_w):
        return pred_mask

    return cv2.resize(pred_mask, (gt_w, gt_h), interpolation=cv2.INTER_LINEAR)


def load_project_metric_functions():
    """
    Prefer the project's DAVIS metric implementation if available.
    Fallback to local implementation otherwise.
    """
    try:
        from core.eval.eval_mask import db_eval_iou, db_eval_boundary

        def j_func(gt_bin, pred_bin):
            return float(db_eval_iou(gt_bin.astype(np.uint8), pred_bin.astype(np.uint8)))

        def f_func(gt_bin, pred_bin, bound_th=0.008):
            return float(
                db_eval_boundary(
                    gt_bin.astype(np.uint8),
                    pred_bin.astype(np.uint8),
                    bound_th=bound_th,
                )
            )

        print("[Metric] Using core.eval.eval_mask.db_eval_iou/db_eval_boundary")
        return j_func, f_func

    except Exception as e:
        print("[Metric] core.eval.eval_mask not available. Using fallback implementation.")
        print(f"[Metric] Import error: {e}")

        def j_func(gt_bin, pred_bin):
            return compute_iou_fallback(gt_bin, pred_bin)

        def f_func(gt_bin, pred_bin, bound_th=0.008):
            return compute_boundary_fallback(gt_bin, pred_bin, bound_th=bound_th)

        return j_func, f_func


def compute_iou_fallback(gt_bin: np.ndarray, pred_bin: np.ndarray) -> float:
    gt = gt_bin.astype(bool)
    pred = pred_bin.astype(bool)

    inter = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()

    if union == 0:
        return 1.0

    return float(inter / union)


def seg2bmap(seg: np.ndarray) -> np.ndarray:
    seg = seg.astype(bool)
    h, w = seg.shape

    bmap = np.zeros((h, w), dtype=np.bool_)

    bmap[:-1, :] |= seg[:-1, :] != seg[1:, :]
    bmap[:, :-1] |= seg[:, :-1] != seg[:, 1:]

    return bmap


def disk_kernel(radius: int) -> np.ndarray:
    y, x = np.ogrid[-radius: radius + 1, -radius: radius + 1]
    disk = (x * x + y * y) <= radius * radius
    return disk.astype(np.uint8)


def compute_boundary_fallback(gt_bin: np.ndarray, pred_bin: np.ndarray, bound_th: float = 0.008) -> float:
    """
    Approximate DAVIS boundary F-score.
    Prefer official/project implementation if available.
    """
    gt = gt_bin.astype(bool)
    pred = pred_bin.astype(bool)

    h, w = gt.shape
    bound_pix = int(np.ceil(bound_th * np.linalg.norm([h, w])))
    bound_pix = max(bound_pix, 1)

    gt_b = seg2bmap(gt)
    pred_b = seg2bmap(pred)

    if gt_b.sum() == 0 and pred_b.sum() == 0:
        return 1.0
    if gt_b.sum() == 0 or pred_b.sum() == 0:
        return 0.0

    kernel = disk_kernel(bound_pix)

    gt_dil = cv2.dilate(gt_b.astype(np.uint8), kernel).astype(bool)
    pred_dil = cv2.dilate(pred_b.astype(np.uint8), kernel).astype(bool)

    precision = np.logical_and(pred_b, gt_dil).sum() / (pred_b.sum() + 1e-8)
    recall = np.logical_and(gt_b, pred_dil).sum() / (gt_b.sum() + 1e-8)

    if precision + recall == 0:
        return 0.0

    return float(2.0 * precision * recall / (precision + recall))


def compute_recall(values: np.ndarray, threshold: float = 0.5) -> float:
    if len(values) == 0:
        return 0.0
    return float((values > threshold).mean())


def compute_decay(values: np.ndarray, num_bins: int = 4) -> float:
    """
    DAVIS-style temporal decay.

    Split sequence into temporal bins.
    Decay = first_bin_mean - last_bin_mean.

    Positive:
        performance decreases over time.
    Negative:
        performance improves over time.
    """
    values = np.asarray(values, dtype=np.float32)

    if len(values) == 0:
        return 0.0

    if len(values) == 1:
        return 0.0

    if len(values) < num_bins:
        mid = len(values) // 2
        if mid == 0:
            return 0.0
        first = values[:mid]
        last = values[mid:]
        return float(first.mean() - last.mean())

    bins = np.array_split(values, num_bins)
    return float(bins[0].mean() - bins[-1].mean())


def compute_sequence_statistics(
    seq_name: str,
    frame_metrics: List[Dict],
    recall_threshold: float = 0.5,
    decay_bins: int = 4,
) -> Dict:
    j_values = np.array([m["J"] for m in frame_metrics], dtype=np.float32)
    f_values = np.array([m["F"] for m in frame_metrics], dtype=np.float32)

    j_mean = float(j_values.mean()) if len(j_values) > 0 else 0.0
    f_mean = float(f_values.mean()) if len(f_values) > 0 else 0.0

    j_recall = compute_recall(j_values, threshold=recall_threshold)
    f_recall = compute_recall(f_values, threshold=recall_threshold)

    j_decay = compute_decay(j_values, num_bins=decay_bins)
    f_decay = compute_decay(f_values, num_bins=decay_bins)

    jf_mean = 0.5 * (j_mean + f_mean)

    return {
        "Sequence": seq_name,
        "J&F-Mean": jf_mean,
        "J-Mean": j_mean,
        "J-Recall": j_recall,
        "J-Decay": j_decay,
        "F-Mean": f_mean,
        "F-Recall": f_recall,
        "F-Decay": f_decay,
        "NumFrames": int(len(frame_metrics)),
    }


def evaluate_sequence(
    seq_name: str,
    gt_dir: str,
    pred_dir: str,
    threshold: float,
    bound_th: float,
    missing_policy: str,
    j_func,
    f_func,
) -> Tuple[Dict, List[Dict]]:
    """
    Evaluation uses GT frame stems as reference.

    For each GT mask:
        find prediction with same stem.
        if missing:
            missing_policy == "zero": empty prediction.
            missing_policy == "skip": skip this frame.
    """
    gt_map = make_file_map(gt_dir)
    pred_map = make_file_map(pred_dir) if os.path.isdir(pred_dir) else {}

    gt_stems = sorted(gt_map.keys(), key=natural_key)

    frame_metrics = []
    missing = []

    for frame_idx, stem in enumerate(gt_stems):
        gt_path = gt_map[stem]
        pred_path = pred_map.get(stem, None)

        if pred_path is None:
            missing.append(stem)
            if missing_policy == "skip":
                continue

        gt_soft = read_mask(gt_path)
        gt_bin = (gt_soft > 0.0).astype(np.uint8)

        if pred_path is None:
            pred_soft_resized = np.zeros_like(gt_soft, dtype=np.float32)
        else:
            pred_soft = read_mask(pred_path, non_black_as_foreground=True)
            pred_soft_resized = resize_pred_to_gt(pred_soft, gt_bin.shape)

        pred_bin = (pred_soft_resized > threshold).astype(np.uint8)

        j_score = j_func(gt_bin, pred_bin)
        f_score = f_func(gt_bin, pred_bin, bound_th=bound_th)

        frame_metrics.append({
            "Sequence": seq_name,
            "Frame": stem,
            "FrameIndex": int(frame_idx),
            "J": float(j_score),
            "F": float(f_score),
            "J&F": float(0.5 * (j_score + f_score)),
            "GT-Positive-Ratio": float(gt_bin.mean()),
            "Pred-Positive-Ratio": float(pred_bin.mean()),
            "PredictionMissing": bool(pred_path is None),
        })

    if len(missing) > 0:
        print(f"  Warning: missing {len(missing)} predictions. Examples: {missing[:5]}")

    if len(frame_metrics) == 0:
        raise RuntimeError(f"No valid frames evaluated for sequence {seq_name}")

    seq_stats = compute_sequence_statistics(seq_name, frame_metrics)

    return seq_stats, frame_metrics


def compute_dataset_statistics(sequence_rows: List[Dict]) -> Dict:
    """
    Dataset-level results use sequence-average.

    This means each sequence has equal weight.
    """
    keys = [
        "J&F-Mean",
        "J-Mean",
        "J-Recall",
        "J-Decay",
        "F-Mean",
        "F-Recall",
        "F-Decay",
    ]

    stats = {}

    for k in keys:
        values = np.array([row[k] for row in sequence_rows], dtype=np.float32)
        stats[k] = float(values.mean()) if len(values) > 0 else 0.0

    stats["NumSequences"] = int(len(sequence_rows))
    stats["TotalFrames"] = int(sum(row["NumFrames"] for row in sequence_rows))

    return stats


def save_csv(rows: List[Dict], path: str):
    ensure_dir(os.path.dirname(path))
    pd.DataFrame(rows).to_csv(path, index=False)


def save_json(obj: Dict, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def parse_sequence_list(seq_list_arg: Optional[str]) -> Optional[List[str]]:
    if seq_list_arg is None:
        return None

    if os.path.exists(seq_list_arg):
        seqs = []
        with open(seq_list_arg, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    seqs.append(line)
        return seqs

    return [x.strip() for x in seq_list_arg.split(",") if x.strip()]


def evaluate_predictions(
    pred_root: str,
    gt_root: str,
    output_dir: str,
    threshold: float = 0.5,
    bound_th: float = 0.008,
    missing_policy: str = "zero",
    sequence_list: Optional[List[str]] = None,
):
    ensure_dir(output_dir)

    j_func, f_func = load_project_metric_functions()

    if sequence_list is None:
        sequences = discover_sequences(gt_root)
    else:
        sequences = sequence_list

    if len(sequences) == 0:
        raise RuntimeError(f"No sequences found under GT root: {gt_root}")

    print("\n========== Evaluation Configuration ==========")
    print(f"Pred root      : {pred_root}")
    print(f"GT root        : {gt_root}")
    print(f"Output dir     : {output_dir}")
    print(f"Threshold      : {threshold}")
    print(f"Boundary th    : {bound_th}")
    print(f"Missing policy : {missing_policy}")
    print(f"Sequences      : {len(sequences)}")
    print("==============================================\n")

    sequence_rows = []
    frame_rows = []

    for seq_name in tqdm(sequences, desc="Evaluating sequences"):
        gt_dir = get_sequence_dir(gt_root, seq_name)
        pred_dir = get_sequence_dir(pred_root, seq_name)

        if not os.path.isdir(gt_dir):
            print(f"[Warning] GT sequence not found: {seq_name}, skip.")
            continue

        if not os.path.isdir(pred_dir):
            print(f"[Warning] Prediction sequence not found: {seq_name}")
            if missing_policy == "skip":
                print("  skip because missing_policy=skip.")
                continue

        print(f"\n=== {seq_name} ===")
        print(f"  GT   : {gt_dir}")
        print(f"  Pred : {pred_dir}")

        seq_stats, seq_frame_rows = evaluate_sequence(
            seq_name=seq_name,
            gt_dir=gt_dir,
            pred_dir=pred_dir,
            threshold=threshold,
            bound_th=bound_th,
            missing_policy=missing_policy,
            j_func=j_func,
            f_func=f_func,
        )

        sequence_rows.append(seq_stats)
        frame_rows.extend(seq_frame_rows)

        print(
            f"  J&F-Mean: {seq_stats['J&F-Mean']:.4f} | "
            f"J-Mean: {seq_stats['J-Mean']:.4f} | "
            f"F-Mean: {seq_stats['F-Mean']:.4f} | "
            f"Frames: {seq_stats['NumFrames']}"
        )

    if len(sequence_rows) == 0:
        raise RuntimeError("No sequence was evaluated.")

    dataset_stats = compute_dataset_statistics(sequence_rows)

    global_row = {
        "J&F-Mean": dataset_stats["J&F-Mean"],
        "J-Mean": dataset_stats["J-Mean"],
        "J-Recall": dataset_stats["J-Recall"],
        "J-Decay": dataset_stats["J-Decay"],
        "F-Mean": dataset_stats["F-Mean"],
        "F-Recall": dataset_stats["F-Recall"],
        "F-Decay": dataset_stats["F-Decay"],
        "NumSequences": dataset_stats["NumSequences"],
        "TotalFrames": dataset_stats["TotalFrames"],
        "Threshold": float(threshold),
        "PredRoot": pred_root,
        "GTRoot": gt_root,
    }

    save_csv([global_row], os.path.join(output_dir, "global_results.csv"))
    # save_csv(sequence_rows, os.path.join(output_dir, "sequence_results.csv"))
    sequence_rows_sorted = sorted(
        sequence_rows,
        key=lambda x: x["J&F-Mean"],
        reverse=True,
    )

    save_csv(sequence_rows_sorted, os.path.join(output_dir, "sequence_results.csv"))
    save_csv(frame_rows, os.path.join(output_dir, "frame_results.csv"))

    json_obj = {
        "global": global_row,
        "sequence_results": sequence_rows_sorted,
        "config": {
            "pred_root": pred_root,
            "gt_root": gt_root,
            "output_dir": output_dir,
            "threshold": threshold,
            "bound_th": bound_th,
            "missing_policy": missing_policy,
        },
    }
    save_json(json_obj, os.path.join(output_dir, "dataset_metrics.json"))

    print("\n" + "=" * 70)
    print("DAVIS-style Evaluation Summary")
    print("=" * 70)
    print(f"J&F-Mean : {dataset_stats['J&F-Mean']:.4f}")
    print(f"J-Mean   : {dataset_stats['J-Mean']:.4f}")
    print(f"J-Recall : {dataset_stats['J-Recall']:.4f}")
    print(f"J-Decay  : {dataset_stats['J-Decay']:.4f}")
    print(f"F-Mean   : {dataset_stats['F-Mean']:.4f}")
    print(f"F-Recall : {dataset_stats['F-Recall']:.4f}")
    print(f"F-Decay  : {dataset_stats['F-Decay']:.4f}")
    print(f"Sequences: {dataset_stats['NumSequences']}")
    print(f"Frames   : {dataset_stats['TotalFrames']}")
    print("=" * 70 + "\n")

    print(f"Saved global results   : {os.path.join(output_dir, 'global_results.csv')}")
    print(f"Saved sequence results : {os.path.join(output_dir, 'sequence_results.csv')}")
    print(f"Saved frame results    : {os.path.join(output_dir, 'frame_results.csv')}")
    print(f"Saved JSON             : {os.path.join(output_dir, 'dataset_metrics.json')}")

    return dataset_stats


def main():
    parser = argparse.ArgumentParser("Evaluate saved motion segmentation predictions")

    parser.add_argument("--pred_root", type=str, required=True,
                        help="Prediction root. Expected format: pred_root/seq_name/*.png")
    parser.add_argument("--gt_root", type=str, required=True,
                        help="GT annotation root. Expected format: gt_root/seq_name/*.png")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Threshold for binarizing prediction soft masks.")
    parser.add_argument("--bound_th", type=float, default=0.008,
                        help="Boundary threshold for F-measure.")
    parser.add_argument("--missing_policy", type=str, default="zero",
                        choices=["zero", "skip"],
                        help="If a prediction for a GT frame is missing: "
                             "'zero' means use empty mask; 'skip' means ignore frame.")
    parser.add_argument("--sequence_list", type=str, default=None,
                        help="Optional txt file or comma-separated sequence names.")

    args = parser.parse_args()

    sequence_list = parse_sequence_list(args.sequence_list)

    evaluate_predictions(
        pred_root=args.pred_root,
        gt_root=args.gt_root,
        output_dir=args.output_dir,
        threshold=args.threshold,
        bound_th=args.bound_th,
        missing_policy=args.missing_policy,
        sequence_list=sequence_list,
    )


if __name__ == "__main__":
    main()