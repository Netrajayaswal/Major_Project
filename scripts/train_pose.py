from dataclasses import asdict
from pathlib import Path
from contextlib import nullcontext
import argparse
import csv
import json
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signlang.models.pose_transformer import PoseTransformerConfig, PoseTransformerModel, Vocabulary
from signlang.topology import FRAME_DIM


class PoseDataset(Dataset):
    def __init__(self, json_dir, vocab=None, max_frames=None, label_filter=None):
        self.items = load_items(json_dir, max_frames=max_frames, label_filter=label_filter)
        if not self.items:
            raise ValueError(f"No keypoint JSON files found in {json_dir}")
        self.vocab = vocab or Vocabulary.build(item["gloss"] for item in self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        return {
            "tokens": self.vocab.encode(item["gloss"]),
            "frames": item["frames"],
            "gloss": item["gloss"],
        }


def main():
    parser = argparse.ArgumentParser(description="Train gloss-to-pose transformer.")
    parser.add_argument("--train-dir", default="data/keypoints/train")
    parser.add_argument("--valid-dir", default="data/keypoints/valid")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/pose_transformer.pt")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=256)
    parser.add_argument("--eos-weight", type=float, default=0.1)
    parser.add_argument("--pck-threshold", type=float, default=0.05)
    parser.add_argument("--labels-csv", default="data/labels/train_labels.csv")
    parser.add_argument("--no-label-filter", action="store_true", help="Train on every JSON clip even if labels CSV exists.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_clip_frames = max(1, args.max_frames - 1)
    label_filter = None if args.no_label_filter else read_label_filter(args.labels_csv)
    if label_filter:
        print(f"label filter: {len(label_filter['video_paths'])} videos, {len(label_filter['glosses'])} glosses")
    train_dataset = PoseDataset(args.train_dir, max_frames=max_clip_frames, label_filter=label_filter)
    valid_dataset = (
        PoseDataset(args.valid_dir, vocab=train_dataset.vocab, max_frames=max_clip_frames, label_filter=label_filter)
        if list(Path(args.valid_dir).rglob("*.json"))
        else train_dataset
    )

    config = PoseTransformerConfig(
        vocab_size=len(train_dataset.vocab),
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_frames=args.max_frames,
    )
    model = PoseTransformerModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_pose_batch(batch, train_dataset.vocab),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_pose_batch(batch, train_dataset.vocab),
    )
    total_steps = max(1, args.epochs * len(train_loader) // max(args.grad_accum, 1))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    log_path = Path("outputs/logs") / f"pose_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best_mpjpe = float("inf")

    with open(log_path, "w", encoding="utf-8", newline="") as log_file:
        writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "epoch",
                "train_loss",
                "valid_loss",
                "valid_mpjpe",
                "keypoint_accuracy",
                "learning_rate",
            ],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                scaler,
                device,
                eos_weight=args.eos_weight,
                grad_accum=args.grad_accum,
            )
            metrics = evaluate(
                model,
                valid_loader,
                device,
                eos_weight=args.eos_weight,
                pck_threshold=args.pck_threshold,
            )
            learning_rate = optimizer.param_groups[0]["lr"]
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": metrics["loss"],
                "valid_mpjpe": metrics["mpjpe"],
                "keypoint_accuracy": metrics["keypoint_accuracy"],
                "learning_rate": learning_rate,
            }
            writer.writerow(row)
            log_file.flush()
            print(
                f"epoch {epoch:03d} "
                f"train_loss={train_loss:.5f} "
                f"valid_loss={metrics['loss']:.5f} "
                f"mpjpe={metrics['mpjpe']:.5f} "
                f"keypoint_acc={metrics['keypoint_accuracy']:.4f}"
            )
            if metrics["mpjpe"] < best_mpjpe:
                best_mpjpe = metrics["mpjpe"]
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": asdict(config),
                        "vocab": train_dataset.vocab.to_dict(),
                        "best_mpjpe": best_mpjpe,
                        "metadata": {
                            "train_dir": str(args.train_dir),
                            "valid_dir": str(args.valid_dir),
                            "train_clips": len(train_dataset),
                            "valid_clips": len(valid_dataset),
                            "train_gloss_counts": gloss_counts(train_dataset.items),
                            "valid_gloss_counts": gloss_counts(valid_dataset.items),
                            "max_clip_frames": max_clip_frames,
                            "labels_csv": str(args.labels_csv) if label_filter else "",
                        },
                    },
                    args.checkpoint,
                )

    print(f"best checkpoint: {args.checkpoint}")
    print(f"training log: {log_path}")


def load_items(json_dir, max_frames=None, label_filter=None):
    items = []
    for path in sorted(Path(json_dir).rglob("*.json")):
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not item_matches_label_filter(data, label_filter):
            continue
        frames = np.asarray(data.get("frames", []), dtype=np.float32)
        if frames.ndim != 2 or frames.shape[1] != FRAME_DIM or len(frames) == 0:
            print(f"[skip] bad frame shape in {path}")
            continue
        gloss = str(data.get("gloss", "")).strip().upper()
        if not gloss:
            print(f"[skip] missing gloss in {path}")
            continue
        if max_frames and len(frames) > max_frames:
            frames = resample_frames(frames, max_frames)
        items.append(
            {
                "gloss": gloss,
                "frames": frames,
                "path": path,
            }
        )
    return items


def read_label_filter(labels_csv):
    labels_path = Path(labels_csv)
    if not labels_path.exists():
        return None
    video_paths = set()
    glosses = set()
    with open(labels_path, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            video_name = (row.get("video_filename") or row.get("filename") or row.get("video") or "").strip()
            gloss = (row.get("gloss") or row.get("label") or "").strip().upper()
            if video_name and gloss:
                video_paths.add(normalize_path_text(video_name))
                glosses.add(gloss)
    if not video_paths:
        return None
    return {"video_paths": video_paths, "glosses": glosses}


def item_matches_label_filter(data, label_filter):
    if not label_filter:
        return True
    gloss = str(data.get("gloss", "")).strip().upper()
    if gloss not in label_filter["glosses"]:
        return False
    source_video = normalize_path_text(data.get("source_video", ""))
    if not source_video:
        return True
    return any(source_video.endswith(video_path) for video_path in label_filter["video_paths"])


def normalize_path_text(path_text):
    return str(path_text).replace("\\", "/").strip().lower()


def resample_frames(frames, max_frames):
    indices = np.linspace(0, len(frames) - 1, max_frames).round().astype(np.int64)
    return frames[indices].astype(np.float32)


def gloss_counts(items):
    counts = {}
    for item in items:
        counts[item["gloss"]] = counts.get(item["gloss"], 0) + 1
    return dict(sorted(counts.items()))


def collate_pose_batch(batch, vocab):
    max_tokens = max(len(item["tokens"]) for item in batch)
    max_frames = max(len(item["frames"]) + 1 for item in batch)
    batch_size = len(batch)

    gloss_tokens = torch.zeros((batch_size, max_tokens), dtype=torch.long)
    decoder_inputs = torch.zeros((batch_size, max_frames, FRAME_DIM), dtype=torch.float32)
    target_coords = torch.zeros((batch_size, max_frames, FRAME_DIM), dtype=torch.float32)
    target_eos = torch.zeros((batch_size, max_frames), dtype=torch.float32)
    frame_mask = torch.zeros((batch_size, max_frames), dtype=torch.bool)

    for batch_index, item in enumerate(batch):
        tokens = torch.tensor(item["tokens"], dtype=torch.long)
        frames = torch.tensor(item["frames"], dtype=torch.float32)
        frame_count = len(frames)
        gloss_tokens[batch_index, : len(tokens)] = tokens
        target_coords[batch_index, :frame_count] = frames
        target_eos[batch_index, frame_count] = 1.0
        decoder_inputs[batch_index, 1 : frame_count + 1] = frames
        frame_mask[batch_index, : frame_count + 1] = True

    gloss_padding_mask = gloss_tokens.eq(vocab.token_to_id[Vocabulary.pad_token])
    return {
        "gloss_tokens": gloss_tokens,
        "gloss_padding_mask": gloss_padding_mask,
        "decoder_inputs": decoder_inputs,
        "target_coords": target_coords,
        "target_eos": target_eos,
        "frame_mask": frame_mask,
    }


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, eos_weight, grad_accum):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    total_batches = 0
    progress = tqdm(loader, desc="train", leave=False)
    for step, batch in enumerate(progress, start=1):
        batch = move_batch(batch, device)
        amp_context = torch.cuda.amp.autocast() if device.type == "cuda" else nullcontext()
        with amp_context:
            coords, eos_logits = model(
                batch["gloss_tokens"],
                batch["decoder_inputs"],
                batch["gloss_padding_mask"],
            )
            loss = pose_loss(coords, eos_logits, batch, eos_weight=eos_weight) / max(grad_accum, 1)

        scaler.scale(loss).backward()
        if step % grad_accum == 0 or step == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += float(loss.detach().cpu()) * max(grad_accum, 1)
        total_batches += 1
        progress.set_postfix(loss=total_loss / total_batches)

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, eos_weight, pck_threshold):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    mpjpe_values = []
    pck_values = []
    for batch in loader:
        batch = move_batch(batch, device)
        coords, eos_logits = model(
            batch["gloss_tokens"],
            batch["decoder_inputs"],
            batch["gloss_padding_mask"],
        )
        loss = pose_loss(coords, eos_logits, batch, eos_weight=eos_weight)
        non_eos_mask = batch["frame_mask"] & (batch["target_eos"] < 0.5)
        predicted = coords[non_eos_mask].reshape(-1, 75, 2)
        target = batch["target_coords"][non_eos_mask].reshape(-1, 75, 2)
        if predicted.numel() > 0:
            distances = torch.linalg.norm(predicted - target, dim=-1)
            mpjpe_values.append(float(distances.mean().detach().cpu()))
            pck_values.append(float((distances < pck_threshold).float().mean().detach().cpu()))
        total_loss += float(loss.detach().cpu())
        total_batches += 1

    return {
        "loss": total_loss / max(total_batches, 1),
        "mpjpe": float(np.mean(mpjpe_values)) if mpjpe_values else 0.0,
        "keypoint_accuracy": float(np.mean(pck_values)) if pck_values else 0.0,
    }


def pose_loss(coords, eos_logits, batch, eos_weight):
    frame_mask = batch["frame_mask"]
    coords_loss = ((coords - batch["target_coords"]) ** 2)[frame_mask].mean()
    eos_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        eos_logits[frame_mask],
        batch["target_eos"][frame_mask],
    )
    return coords_loss + eos_weight * eos_loss


def move_batch(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


if __name__ == "__main__":
    main()
