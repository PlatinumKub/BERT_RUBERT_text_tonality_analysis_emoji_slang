"""
inference.py
------------
Оценивает обученную модель и визуализирует результаты.

Использование:
    python src/inference.py --lang en
    python src/inference.py --lang ru

Что генерируется:
    1. Кривые обучения (loss + accuracy)
    2. Нормализованная матрица ошибок (confusion matrix)
    3. Classification report (precision / recall / f1 по классам)
    4. Bar chart точности по классам
    5. Таблица примеров с правильными и неправильными предсказаниями
    6. Все графики сохраняются в reports/<lang>/

Пример вывода:
    reports/
    └── en/
        ├── training_curves.png
        ├── confusion_matrix.png
        ├── per_class_accuracy.png
        └── examples.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoTokenizer

from config import cfg, ROOT
from defs.dataset import create_dataloader
from defs.model import EN_ID2LABEL, RU_ID2LABEL, load_model
from defs.preprocessing import TextPreprocessor

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({"figure.autolayout": True, "axes.titleweight": "bold"})


# ---------------------------------------------------------------------------
# Сбор предсказаний
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_predictions(
    model,
    loader,
    device: torch.device,
) -> tuple[list[str], list[int], list[int], list[list[float]]]:
    """
    Прогоняет весь DataLoader, возвращает:
        texts      - исходные тексты
        preds      - предсказанные метки (int)
        targets    - истинные метки (int)
        probs      - вероятности по всем классам (list of lists)
    """
    model.eval()
    texts, preds, targets, probs = [], [], [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        batch_probs = F.softmax(outputs.logits, dim=1).cpu().numpy()
        batch_preds = batch_probs.argmax(axis=1)

        texts.extend(batch["text"])
        preds.extend(batch_preds.tolist())
        targets.extend(batch["targets"].numpy().tolist())
        probs.extend(batch_probs.tolist())

    return texts, preds, targets, probs


# ---------------------------------------------------------------------------
# Графики
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, save_path: str, lang: str) -> None:
    """Loss и Accuracy кривые для train и val."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, metric, title, ylabel in zip(
        axes,
        [("train_loss", "val_loss"), ("train_acc", "val_acc")],
        ["Loss", "Accuracy"],
        ["CrossEntropy Loss", "Accuracy"],
    ):
        ax.plot(epochs, history[metric[0]], "o-", label="Train", linewidth=2)
        ax.plot(epochs, history[metric[1]], "s--", label="Val", linewidth=2)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Training History — {'English BERT' if lang == 'en' else 'Russian RuBERT'}",
        fontsize=14, fontweight="bold",
    )
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {save_path}")


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    save_path: str,
    lang: str,
) -> None:
    """Нормализованная матрица ошибок."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)

    # Подписи значений в ячейках
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, f"{cm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
            )

    ax.set_ylabel("True label", fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_title(
        f"Confusion Matrix — {'EN' if lang == 'en' else 'RU'} (normalized)",
        fontsize=13,
    )

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {save_path}")


def plot_per_class_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    save_path: str,
) -> None:
    """Bar chart: precision, recall, f1 для каждого класса."""
    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True
    )

    metrics = ["precision", "recall", "f1-score"]
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, metric in enumerate(metrics):
        values = [report[cls][metric] for cls in class_names]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize())
        # Подписи на барах
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.2f}",
                ha="center", va="bottom", fontsize=9,
            )

    ax.set_xlabel("Class")
    ax.set_ylabel("Score")
    ax.set_title("Per-Class Metrics (Precision / Recall / F1)")
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0, 1.12)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Общая accuracy как горизонтальная линия
    acc = report["accuracy"]
    ax.axhline(acc, color="red", linestyle="--", linewidth=1.5, label=f"Accuracy = {acc:.3f}")
    ax.legend()

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {save_path}")


def plot_examples(
    texts: list[str],
    preds: list[int],
    targets: list[int],
    probs: list[list[float]],
    class_names: list[str],
    save_path: str,
    n_per_type: int = 4,
) -> None:
    """
    Таблица примеров: n правильных и n неправильных предсказаний.
    Показывает текст, истинную метку, предсказание и уверенность модели.
    """
    rows = []
    correct_idx = [i for i, (p, t) in enumerate(zip(preds, targets)) if p == t]
    wrong_idx = [i for i, (p, t) in enumerate(zip(preds, targets)) if p != t]

    # Случайная выборка примеров
    rng = np.random.default_rng(42)
    sample_correct = rng.choice(correct_idx, size=min(n_per_type, len(correct_idx)), replace=False)
    sample_wrong = rng.choice(wrong_idx, size=min(n_per_type, len(wrong_idx)), replace=False)

    for idx in sample_correct:
        rows.append({
            "Status": "Correct",
            "True": class_names[targets[idx]],
            "Pred": class_names[preds[idx]],
            "Conf": f"{max(probs[idx]):.2f}",
            "Text": texts[idx][:80] + ("..." if len(texts[idx]) > 80 else ""),
        })
    for idx in sample_wrong:
        rows.append({
            "Status": "Wrong",
            "True": class_names[targets[idx]],
            "Pred": class_names[preds[idx]],
            "Conf": f"{max(probs[idx]):.2f}",
            "Text": texts[idx][:80] + ("..." if len(texts[idx]) > 80 else ""),
        })

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(14, len(df) * 0.55 + 1.5))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.auto_set_column_width(col=list(range(len(df.columns))))

    # Цвет строк: зелёный для правильных, красный для неправильных
    for row_idx, status in enumerate(df["Status"], start=1):
        color = "#d4f1d4" if status == "Correct" else "#f1d4d4"
        for col_idx in range(len(df.columns)):
            table[row_idx, col_idx].set_facecolor(color)

    ax.set_title("Example Predictions", fontweight="bold", pad=12, fontsize=12)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Сохранено: {save_path}")


# ---------------------------------------------------------------------------
# Главная функция инференса
# ---------------------------------------------------------------------------

def run_inference(lang: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mcfg = cfg.en if lang == "en" else cfg.ru
    id2label = EN_ID2LABEL if lang == "en" else RU_ID2LABEL
    class_names = [id2label[i] for i in range(mcfg.num_labels)]

    report_dir = ROOT / "reports" / lang
    report_dir.mkdir(parents=True, exist_ok=True)

    # --- Загрузка модели ---
    print(f"\n[1/5] Загружаем модель из {mcfg.checkpoint_path}...")
    model = load_model(
        model_name=mcfg.model_name,
        weights_path=mcfg.checkpoint_path,
        lang=lang,
        device=device,
    )

    # --- Тестовые данные ---
    print("\n[2/5] Загружаем тестовую выборку...")
    test_csv = Path(mcfg.checkpoint_dir) / f"test_{lang}.csv"
    if not test_csv.exists():
        raise FileNotFoundError(
            f"Файл {test_csv} не найден. Сначала запусти train.py --lang {lang}"
        )
    df_test = pd.read_csv(test_csv)
    print(f"  Примеров: {len(df_test)}")

    # --- Препроцессинг и DataLoader ---
    print("\n[3/5] Подготавливаем DataLoader...")
    slang_csv = cfg.data.en_slang_csv if lang == "en" else cfg.data.ru_slang_csv
    preprocessor = TextPreprocessor.from_csv(
        emoji_csv=cfg.data.emoji_csv,
        slang_csv=slang_csv,
        lang=lang,
    )
    tokenizer = AutoTokenizer.from_pretrained(mcfg.model_name)

    test_loader = create_dataloader(
        df=df_test,
        text_col=mcfg.text_col,
        label_col=mcfg.label_col,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        max_len=mcfg.max_len,
        batch_size=mcfg.batch_size,
        shuffle=False,
        num_workers=mcfg.num_workers,
    )

    # --- Предсказания ---
    print("\n[4/5] Собираем предсказания...")
    texts, preds, targets, probs = collect_predictions(model, test_loader, device)

    # --- Метрики в консоль ---
    print("\n" + "="*55)
    print("CLASSIFICATION REPORT")
    print("="*55)
    print(classification_report(targets, preds, target_names=class_names, digits=4))

    # --- Графики ---
    print("\n[5/5] Строим графики...")

    # 1. Кривые обучения (если есть история)
    history_path = Path(mcfg.history_path)
    if history_path.exists():
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)
        plot_training_curves(
            history,
            save_path=str(report_dir / "training_curves.png"),
            lang=lang,
        )
    else:
        print(f"  История не найдена ({history_path}), пропускаем кривые обучения")

    # 2. Матрица ошибок
    plot_confusion_matrix(
        targets, preds, class_names,
        save_path=str(report_dir / "confusion_matrix.png"),
        lang=lang,
    )

    # 3. Метрики по классам
    plot_per_class_metrics(
        targets, preds, class_names,
        save_path=str(report_dir / "per_class_metrics.png"),
    )

    # 4. Примеры предсказаний
    plot_examples(
        texts, preds, targets, probs, class_names,
        save_path=str(report_dir / "examples.png"),
    )

    print(f"\nВсе графики сохранены в {report_dir}/")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Инференс и визуализация результатов"
    )
    parser.add_argument(
        "--lang",
        choices=["en", "ru"],
        required=True,
        help="Язык модели: 'en' или 'ru'",
    )
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  Инференс: {'English BERT' if args.lang == 'en' else 'Russian RuBERT'}")
    print(f"{'='*50}\n")

    run_inference(args.lang)