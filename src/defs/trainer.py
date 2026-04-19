"""
trainer.py
----------
Скрипты обучения и предсказания для BertForSentiment.

Модули:
  - train_epoch()   — одно прохождение по всем тренировочным данным
  - eval_epoch()    — одно прохождение по всем валидационным/тестовым данным
  - Trainer         — полный обучающий цикл с историей, контрольными точками, и графическим отображением
"""

from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from defs.model import BertForSentiment


# ---------------------------------------------------------------------------
# Функции для одной эпохи
# ---------------------------------------------------------------------------

def train_epoch(
    model: BertForSentiment,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler=None,
    max_grad_norm: float = 1.0,
) -> tuple[float, float]:
    """
    Один полный проход обучения.

    Возвращает:
    -------
    (accuracy, mean_loss) как Python floats
    """
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    losses, correct, total = [], 0, 0

    with tqdm(data_loader, desc="  train", unit="batch", leave=False) as pbar:
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            preds = outputs.logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}")

    return correct / total, float(np.mean(losses))


@torch.no_grad()
def eval_epoch(
    model: BertForSentiment,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """
    Один полный проход предсказания (без градиентов).

    Возвращает:
    -------
    (accuracy, mean_loss) как Python floats
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    losses, correct, total = [], 0, 0

    with tqdm(data_loader, desc="    val", unit="batch", leave=False) as pbar:
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, targets)

            preds = outputs.logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            losses.append(loss.item())
            pbar.set_postfix(loss=f"{np.mean(losses):.4f}")

    return correct / total, float(np.mean(losses))


# ---------------------------------------------------------------------------
# Полный тренер
# ---------------------------------------------------------------------------

class Trainer:
    """
    Класс полного тренировочного процесса.

    Параметры:
    ----------
    model        : BertForSentiment
    train_loader : DataLoader
    val_loader   : DataLoader
    device       : torch.device
    lr           : learning rate для AdamW
    epochs       : количество тренировочных эпох
    checkpoint_dir : директория для сохранения весов лучшей модели
    """

    def __init__(
        self,
        model: BertForSentiment,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        lr: float = 2e-5,
        epochs: int = 5,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(model.parameters(), lr=lr)
        num_steps = epochs * len(train_loader)
        self.scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=num_steps,
        )
        self.history: dict[str, list] = defaultdict(list)

    def fit(self) -> dict[str, list]:
        """
        Запускает полный цикл обучения.

        Возвращает:
        -------
        словарь истории обучения с ключами: train_acc, train_loss, val_acc, val_loss
        """
        best_val_acc = 0.0

        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")
            print("-" * 40)

            train_acc, train_loss = train_epoch(
                self.model, self.train_loader, self.optimizer,
                self.device, self.scheduler,
            )
            val_acc, val_loss = eval_epoch(
                self.model, self.val_loader, self.device,
            )

            print(f"  Train  loss: {train_loss:.4f}  acc: {train_acc:.4f}")
            print(f"  Val    loss: {val_loss:.4f}  acc: {val_acc:.4f}")

            self.history["train_acc"].append(train_acc)
            self.history["train_loss"].append(train_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_loss"].append(val_loss)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = self.checkpoint_dir / "best_model.pth"
                torch.save(self.model.state_dict(), save_path)
                print(f"  ✓ New best ({val_acc:.4f}) — saved to {save_path}")

        print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
        return dict(self.history)

    def plot_history(self, save_path: Optional[str] = None) -> None:
        """График train/val loss и кривые accuracy вместе."""
        epochs = range(1, len(self.history["train_acc"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Loss
        axes[0].plot(epochs, self.history["train_loss"], label="Train", marker="o")
        axes[0].plot(epochs, self.history["val_loss"], label="Val", marker="o")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("CrossEntropy Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Accuracy
        axes[1].plot(epochs, self.history["train_acc"], label="Train", marker="o")
        axes[1].plot(epochs, self.history["val_acc"], label="Val", marker="o")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle("Training History", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")
        plt.show()


# ---------------------------------------------------------------------------
# Получем предсказания модели
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    model: BertForSentiment,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[list, list, list]:
    """
    Запускаем инференс модели по DataLoader и собираем тексты, предсказания, основную истину.

    Возвращает:
    -------
    (texts, predictions, true_labels) — всё это Python lists
    """
    model.eval()
    all_texts, all_preds, all_targets = [], [], []

    for batch in tqdm(data_loader, desc="Evaluating", unit="batch"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = outputs.logits.argmax(dim=1)

        all_texts.extend(batch.get("text", [""] * len(targets)))
        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(targets.cpu().tolist())

    return all_texts, all_preds, all_targets


def print_report(
    y_true: list,
    y_pred: list,
    class_names: list[str],
) -> None:
    """Выводит sklearn classification report."""
    print(classification_report(y_true, y_pred, target_names=class_names))


def plot_confusion_matrix(
    y_true: list,
    y_pred: list,
    class_names: list[str],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
) -> None:
    """График нормализованной матрицы ошибок в виде тепловой карты."""
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, normalize="true")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_ylabel("True label", fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    plt.show()
