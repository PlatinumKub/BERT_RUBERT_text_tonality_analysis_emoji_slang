"""
dataset.py
----------
PyTorch Dataset и фабрика DataLoader для обоих языков.

Главная идея: Dataset не знает ни о EN, ни о RU.
Он просто принимает массив текстов, массив меток,
токенайзер и препроцессор — и работает.

Использование:
    from dataset import create_dataloader
    from preprocessing import TextPreprocessor
    from transformers import AutoTokenizer
    from config import cfg

    preprocessor = TextPreprocessor.from_csv(
        cfg.data.emoji_csv, cfg.data.en_slang_csv, lang="en"
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.en.model_name)

    loader = create_dataloader(
        df=df_train,
        text_col="OriginalTweet",
        label_col="Sentiment",
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        max_len=cfg.en.max_len,
        batch_size=cfg.en.batch_size,
        shuffle=True,
    )
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from defs.preprocessing import TextPreprocessor


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SentimentDataset(Dataset):
    """
    Универсальный Dataset для задачи классификации тональности.

    Препроцессинг (эмодзи, сленг, пунктуация) выполняется на лету
    в __getitem__, так что на диск чистые тексты не пишутся.

    Parameters
    ----------
    texts        : массив сырых текстов (numpy array или list)
    labels       : массив целочисленных меток (0, 1, 2)
    tokenizer    : любой HuggingFace-совместимый токенайзер
    preprocessor : экземпляр TextPreprocessor (из preprocessing.py)
    max_len      : максимальная длина токенов (256 для EN, 512 для RU)
    """

    def __init__(
        self,
        texts: np.ndarray,
        labels: np.ndarray,
        tokenizer: PreTrainedTokenizerBase,
        preprocessor: TextPreprocessor,
        max_len: int,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        raw_text = str(self.texts[idx])
        clean_text = self.preprocessor(raw_text)

        encoding = self.tokenizer(
            clean_text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )

        return {
            # Сохраняем чистый текст — пригодится в inference.py для примеров
            "text": clean_text,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloader(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
    tokenizer: PreTrainedTokenizerBase,
    preprocessor: TextPreprocessor,
    max_len: int,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    DataFrame → DataLoader одной строкой.

    Parameters
    ----------
    df           : pandas DataFrame с колонками text_col и label_col
    text_col     : название колонки с текстами ('OriginalTweet' для EN, 'text' для RU)
    label_col    : название колонки с метками (0/1/2)
    tokenizer    : HuggingFace токенайзер
    preprocessor : TextPreprocessor
    max_len      : максимальная длина последовательности
    batch_size   : размер батча
    shuffle      : перемешивать ли (True для train, False для val/test)
    num_workers  : количество воркеров DataLoader
    """
    dataset = SentimentDataset(
        texts=df[text_col].to_numpy(),
        labels=df[label_col].to_numpy(),
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        max_len=max_len,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        # pin_memory ускоряет CPU→GPU трансфер, если GPU есть
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Утилита для быстрой проверки батча
# ---------------------------------------------------------------------------

def inspect_batch(loader: DataLoader, n: int = 2) -> None:
    """
    Печатает форму тензоров и пример текстов из первого батча.
    Удобно запускать после создания DataLoader, чтобы убедиться,
    что всё собралось правильно.

    Parameters
    ----------
    loader : DataLoader
    n      : сколько примеров показать (по умолчанию 2)
    """
    batch = next(iter(loader))
    print("=== Batch inspection ===")
    print(f"  input_ids shape  : {batch['input_ids'].shape}")
    print(f"  attention_mask   : {batch['attention_mask'].shape}")
    print(f"  targets          : {batch['targets'].shape}")
    print(f"  unique labels    : {batch['targets'].unique().tolist()}")
    print()
    for i in range(min(n, len(batch["text"]))):
        label = batch["targets"][i].item()
        print(f"  [{i}] label={label} | text: {batch['text'][i][:80]}...")
    print()


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pandas as pd
    from transformers import AutoTokenizer
    from config import cfg

    print("Создаём игрушечный DataFrame для проверки...")

    dummy_df = pd.DataFrame({
        "OriginalTweet": [
            "I love this! 😍 So great!!",
            "This is terrible, worst day ever 😭",
            "Just another normal day tbh lmk",
        ],
        "Sentiment": [2, 0, 1],
    })

    preprocessor = TextPreprocessor.from_csv(
        emoji_csv=cfg.data.emoji_csv,
        slang_csv=cfg.data.en_slang_csv,
        lang="en",
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.en.model_name)

    loader = create_dataloader(
        df=dummy_df,
        text_col="OriginalTweet",
        label_col="Sentiment",
        tokenizer=tokenizer,
        preprocessor=preprocessor,
        max_len=cfg.en.max_len,
        batch_size=3,
        shuffle=False,
    )

    inspect_batch(loader, n=3)
    print("dataset.py smoke-test прошёл успешно.")