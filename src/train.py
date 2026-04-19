"""
train.py
--------
Загружает данные, обучает модель, сохраняет веса и историю обучения.

Использование:
    python src/train.py --lang en
    python src/train.py --lang ru

Что происходит под капотом:
    1. Загрузка и балансировка датасета
    2. Train / Val / Test split (стратифицированный, фиксированный seed)
    3. Препроцессинг + токенизация через DataLoader
    4. Обучение через Trainer из defs/trainer.py
    5. Сохранение весов → checkpoints/
    6. Сохранение истории обучения → history/ (нужен inference.py для графиков)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# defs/ находится рядом с этим файлом в src/
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

from config import cfg
from defs.dataset import create_dataloader, inspect_batch
from defs.model import EN_ID2LABEL, RU_ID2LABEL, build_model
from defs.preprocessing import TextPreprocessor
from defs.trainer import Trainer

#Устанавливаем токен с HuggingFace для загрузки датасета
from huggingface_hub import login
login(token="hf_token")

# ---------------------------------------------------------------------------
# Воспроизводимость
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Загрузка и подготовка данных
# ---------------------------------------------------------------------------

def load_english_data() -> pd.DataFrame:
    """
    Загружает три EN датасета, объединяет, балансирует классы
    и возвращает DataFrame с колонками ['OriginalTweet', 'Sentiment'].

    Метки: 0 = Negative, 1 = Neutral, 2 = Positive
    """
    # --- Twitter Sentiment ---
    df_twitter = pd.read_csv(cfg.data.en_tweets_csv, encoding="ISO-8859-1")
    df_twitter = df_twitter.rename(
        columns={"selected_text": "OriginalTweet", "sentiment": "Sentiment"}
    )[["OriginalTweet", "Sentiment"]]
    df_twitter["Sentiment"] = df_twitter["Sentiment"].map(
        {"negative": 0, "neutral": 1, "positive": 2}
    )

    # --- Corona NLP train + test ---
    df_corona_train = pd.read_csv(
        cfg.data.en_corona_train_csv, encoding="ISO-8859-1"
    )[["OriginalTweet", "Sentiment"]]
    df_corona_test = pd.read_csv(
        cfg.data.en_corona_test_csv
    )[["OriginalTweet", "Sentiment"]]

    label_map = {
        "Extremely Negative": 0, "Negative": 0,
        "Neutral": 1,
        "Positive": 2, "Extremely Positive": 2,
    }
    df_corona_train["Sentiment"] = df_corona_train["Sentiment"].map(label_map)
    df_corona_test["Sentiment"] = df_corona_test["Sentiment"].map(label_map)

    # Берём только нейтральные твиты из Twitter, чтобы добалансировать
    df_neutral_twitter = df_twitter[df_twitter["Sentiment"] == 1]
    full_df = pd.concat(
        [df_neutral_twitter, df_corona_train, df_corona_test], ignore_index=True
    ).dropna(subset=["Sentiment"])
    full_df["Sentiment"] = full_df["Sentiment"].astype(int)

    # Балансировка: обрезаем до размера наименьшего класса
    min_class_size = full_df["Sentiment"].value_counts().min()
    balanced_df = (
        full_df.groupby("Sentiment", group_keys=False)
        .apply(lambda x: x.sample(min_class_size, random_state=cfg.train.seed))
        .sample(frac=1, random_state=cfg.train.seed)
        .reset_index(drop=True)
    )

    print(f"[EN] Загружено: {len(balanced_df)} примеров")
    print(balanced_df["Sentiment"].value_counts().sort_index().to_string())
    return balanced_df


def load_russian_data() -> pd.DataFrame:
    """
    Загружает RU датасет с HuggingFace, объединяет train+validation,
    возвращает DataFrame с колонками ['text', 'label'].

    Метки: 0 = Neutral, 1 = Positive, 2 = Negative
    """
    from datasets import load_dataset as hf_load

    ds = hf_load(cfg.data.ru_hf_dataset)
    df_train = ds["train"].to_pandas().rename(
        columns={"sentiment": "label"}
    )[["text", "label"]]
    df_val = ds["validation"].to_pandas().rename(
        columns={"sentiment": "label"}
    )[["text", "label"]]

    full_df = pd.concat([df_train, df_val], ignore_index=True)
    full_df = full_df.sample(frac=1, random_state=cfg.train.seed).reset_index(drop=True)

    print(f"[RU] Загружено: {len(full_df)} примеров")
    print(full_df["label"].value_counts().sort_index().to_string())
    return full_df


# ---------------------------------------------------------------------------
# Главная функция обучения
# ---------------------------------------------------------------------------

def train(lang: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nУстройство: {device}")

    set_seed(cfg.train.seed)

    # Выбираем конфигурацию для языка
    mcfg = cfg.en if lang == "en" else cfg.ru

    # Создаём директории для сохранения
    Path(mcfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(mcfg.history_dir).mkdir(parents=True, exist_ok=True)

    # --- Данные ---
    print(f"\n[1/5] Загружаем данные ({lang.upper()})...")
    df = load_english_data() if lang == "en" else load_russian_data()

    # Стратифицированный split: сначала отрезаем test, потом val от остатка
    df_trainval, df_test = train_test_split(
        df,
        test_size=cfg.train.test_split,
        stratify=df[mcfg.label_col],
        random_state=cfg.train.seed,
    )
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=cfg.train.val_split,
        stratify=df_trainval[mcfg.label_col],
        random_state=cfg.train.seed,
    )

    print(
        f"  Train: {len(df_train)} | Val: {len(df_val)} | Test: {len(df_test)}"
    )

    # Сохраняем тестовую выборку рядом с чекпоинтом
    # — inference.py загрузит именно её, чтобы не было data leakage
    test_save_path = Path(mcfg.checkpoint_dir) / f"test_{lang}.csv"
    df_test.to_csv(test_save_path, index=False)
    print(f"  Тест сохранён: {test_save_path}")

    # --- Препроцессинг и токенизация ---
    print("\n[2/5] Инициализируем препроцессор и токенайзер...")
    slang_csv = cfg.data.en_slang_csv if lang == "en" else cfg.data.ru_slang_csv
    preprocessor = TextPreprocessor.from_csv(
        emoji_csv=cfg.data.emoji_csv,
        slang_csv=slang_csv,
        lang=lang,
    )
    tokenizer = AutoTokenizer.from_pretrained(mcfg.model_name)

    # --- DataLoaders ---
    print("\n[3/5] Создаём DataLoader...")
    train_loader = create_dataloader(
        df=df_train,
        text_col=mcfg.text_col,
        label_col=mcfg.label_col,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        max_len=mcfg.max_len,
        batch_size=mcfg.batch_size,
        shuffle=True,
        num_workers=mcfg.num_workers,
    )
    val_loader = create_dataloader(
        df=df_val,
        text_col=mcfg.text_col,
        label_col=mcfg.label_col,
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        max_len=mcfg.max_len,
        batch_size=mcfg.batch_size,
        shuffle=False,
        num_workers=mcfg.num_workers,
    )

    # Быстрая проверка батча перед обучением
    inspect_batch(train_loader, n=2)

    # --- Модель ---
    print("\n[4/5] Строим модель...")
    id2label = EN_ID2LABEL if lang == "en" else RU_ID2LABEL
    model = build_model(mcfg.model_name, lang=lang, device=device)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"  Всего параметров   : {total_params:.1f}M")
    print(f"  Обучаемых          : {trainable_params:.1f}M")

    # --- Обучение ---
    print("\n[5/5] Начинаем обучение...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=mcfg.lr,
        epochs=mcfg.epochs,
        checkpoint_dir=mcfg.checkpoint_dir,
    )

    history = trainer.fit()

    # Trainer сохраняет best_model.pth, переименовываем в нужное имя
    best_path = Path(mcfg.checkpoint_dir) / "best_model.pth"
    target_path = Path(mcfg.checkpoint_path)
    if best_path.exists():
        best_path.rename(target_path)
        print(f"\nВеса сохранены: {target_path}")

    # Сохраняем историю обучения для inference.py
    history_serializable = {k: [float(v) for v in vals] for k, vals in history.items()}
    with open(mcfg.history_path, "w", encoding="utf-8") as f:
        json.dump(history_serializable, f, indent=2)
    print(f"История сохранена: {mcfg.history_path}")

    # Финальный график прямо из тренера
    plot_path = str(Path(mcfg.history_dir) / f"training_curves_{lang}.png")
    trainer.plot_history(save_path=plot_path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Обучение BERT-модели для анализа тональности"
    )
    parser.add_argument(
        "--lang",
        choices=["en", "ru"],
        required=True,
        help="Язык модели: 'en' (BERT) или 'ru' (RuBERT)",
    )
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  Обучение: {'English BERT' if args.lang == 'en' else 'Russian RuBERT'}")
    print(f"{'='*50}\n")

    train(args.lang)
