from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
import sys

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import (
    EfficientNet_B0_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    resnet18,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.palmistry.gate_policy import GATE_DECISIONS


GATE_LABEL_TO_ID = {label: idx for idx, label in enumerate(GATE_DECISIONS)}
GATE_ID_TO_LABEL = {idx: label for label, idx in GATE_LABEL_TO_ID.items()}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight three-class palmistry gate classifier.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", choices=("resnet18", "efficientnet_b0"), default="resnet18")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


class GatePolicyDataset(Dataset):
    def __init__(self, rows: list[dict], image_folder: Path, transform: transforms.Compose) -> None:
        self.rows = rows
        self.image_folder = image_folder
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        image_path = Path(row["image"])
        if not image_path.is_absolute():
            image_path = self.image_folder / image_path
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), GATE_LABEL_TO_ID[row["gate_decision"]]


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_tf, eval_tf


def build_model(model_name: str, pretrained: bool) -> nn.Module:
    if model_name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, len(GATE_DECISIONS))
        return model
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(GATE_DECISIONS))
    return model


def compute_class_weights(rows: list[dict], device: torch.device) -> torch.Tensor:
    counts = Counter(row["gate_decision"] for row in rows)
    total = sum(counts.values())
    weights = []
    for label in GATE_DECISIONS:
        count = counts[label]
        weights.append(total / (len(GATE_DECISIONS) * max(count, 1)))
    return torch.tensor(weights, dtype=torch.float32, device=device)


def confusion_and_metrics(logits: torch.Tensor, labels: torch.Tensor) -> tuple[list[list[int]], dict[str, float]]:
    preds = logits.argmax(dim=1)
    num_classes = len(GATE_DECISIONS)
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for gold, pred in zip(labels.cpu(), preds.cpu()):
        cm[gold, pred] += 1

    total = cm.sum().item()
    correct = torch.diag(cm).sum().item()
    acc = correct / total if total else 0.0

    precisions = []
    recalls = []
    f1s = []
    per_class = {}
    for idx, label in enumerate(GATE_DECISIONS):
        tp = cm[idx, idx].item()
        fp = cm[:, idx].sum().item() - tp
        fn = cm[idx, :].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(cm[idx, :].sum().item()),
        }

    metrics = {
        "accuracy": acc,
        "macro_precision": sum(precisions) / num_classes,
        "macro_recall": sum(recalls) / num_classes,
        "macro_f1": sum(f1s) / num_classes,
        "per_class": per_class,
    }
    return cm.tolist(), metrics


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype,
) -> tuple[float, dict[str, float], list[list[int]]]:
    model.eval()
    losses = []
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            losses.append(loss.item())
            logits_list.append(logits.detach().cpu())
            labels_list.append(labels.detach().cpu())

    all_logits = torch.cat(logits_list, dim=0)
    all_labels = torch.cat(labels_list, dim=0)
    cm, metrics = confusion_and_metrics(all_logits, all_labels)
    return sum(losses) / max(len(losses), 1), metrics, cm


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    train_rows = load_jsonl(Path(args.train_jsonl))
    val_rows = load_jsonl(Path(args.val_jsonl))
    image_folder = Path(args.image_folder)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_tf, eval_tf = build_transforms(args.image_size)
    train_ds = GatePolicyDataset(train_rows, image_folder=image_folder, transform=train_tf)
    val_ds = GatePolicyDataset(val_rows, image_folder=image_folder, transform=eval_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    model = build_model(args.model_name, pretrained=args.pretrained).to(device)
    class_weights = compute_class_weights(train_rows, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)

    history_path = output_dir / "history.jsonl"
    summary_path = output_dir / "summary.json"
    best_path = output_dir / "best.pt"
    latest_path = output_dir / "latest.pt"

    best_macro_f1 = -1.0
    best_epoch = -1
    epochs_without_improve = 0
    started_at = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
            num_batches += 1

        train_loss = running_loss / max(num_batches, 1)
        val_loss, val_metrics, confusion_matrix = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            amp_dtype=amp_dtype,
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_per_class": val_metrics["per_class"],
            "val_confusion_matrix": confusion_matrix,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(epoch_record, ensure_ascii=False) + "\n")

        checkpoint = {
            "epoch": epoch,
            "model_name": args.model_name,
            "state_dict": model.state_dict(),
            "label_to_id": GATE_LABEL_TO_ID,
            "id_to_label": GATE_ID_TO_LABEL,
            "image_size": args.image_size,
        }
        torch.save(checkpoint, latest_path)

        improved = val_metrics["macro_f1"] > best_macro_f1
        if improved:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save(checkpoint, best_path)
        else:
            epochs_without_improve += 1

        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_macro_f1={val_metrics['macro_f1']:.4f}",
            flush=True,
        )

        if epochs_without_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}.", flush=True)
            break

    summary = {
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "image_folder": str(image_folder),
        "output_dir": str(output_dir),
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "epochs_requested": args.epochs,
        "best_epoch": best_epoch,
        "best_macro_f1": best_macro_f1,
        "runtime_seconds": time.time() - started_at,
        "label_to_id": GATE_LABEL_TO_ID,
        "train_label_counts": dict(Counter(row["gate_decision"] for row in train_rows)),
        "val_label_counts": dict(Counter(row["gate_decision"] for row in val_rows)),
        "best_checkpoint": str(best_path),
        "latest_checkpoint": str(latest_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
