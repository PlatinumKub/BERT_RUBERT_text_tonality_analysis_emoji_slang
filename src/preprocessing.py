"""
preprocessing.py
----------------
Предобработка текста для дальнейшего обучения
Как для Русского (ru) так и для Английского (en).

Здесь описаны функции обработки эмоджи/сленга, lowercasing, удаления пунктуации
и пустых мест.
"""

import re
import string
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Функции загрузки и создания словарей эмоджи и сленга
# ---------------------------------------------------------------------------

def load_emoji_dict(csv_path: str, lang: str = "en") -> dict[str, str]:
    """
    Загружает словарь эмоджи из CSV-файла в зависимости от языка.

    Колонки:
      - 'emoji'
      - 'description_en' (описание на Английском)  или
      - 'description_ru' (описание на Русском)
    """
    df = pd.read_csv(csv_path)
    df.rename(columns={'name': 'description_en'}, inplace=True)
    desc_col = "description_ru" if lang == "ru" else "description_en"
    if desc_col not in df.columns:
        raise ValueError(
            f"CSV at '{csv_path}' must contain columns 'emoji' and '{desc_col}'."
        )
    return dict(zip(df["emoji"], df[desc_col]))


def load_slang_dict(csv_path: str) -> dict[str, str]:
    """
    Загружает словарь сленга из CSV файла

    Колонки: 'acronym', 'expansion'
    """
    df = pd.read_csv(csv_path)
    if not {"acronym", "expansion"}.issubset(df.columns):
        raise ValueError(
            f"CSV at '{csv_path}' must contain columns 'acronym' and 'expansion'."
        )
    # для упрощения дальнейшей работы приводим к нижнему регистру
    return {row["acronym"].lower(): row["expansion"] for _, row in df.iterrows()}


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Функции обработки текста
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def replace_emojis(text: str, emoji_dict: dict[str, str]) -> str:
    """Заменяет эмоджи в тексте на его описание"""
    for emj, desc in emoji_dict.items():
        text = text.replace(emj, f" {desc} ")
    return text


def expand_slang(text: str, slang_dict: dict[str, str]) -> str:
    """Заменяет сленг в тексте на его описание."""
    words = text.split()
    return " ".join(slang_dict.get(w.lower(), w) for w in words)


def remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def remove_urls(text: str) -> str:
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_hashtags(text: str) -> str:
    return re.sub(r"#\w+", "", text)


def remove_mentions(text: str) -> str:
    return re.sub(r"@\w+", "", text)


def collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Pipeline
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class TextPreprocessor:
    """
    Полная предобработка текста с эмоджи и сленгом
    в единном `__call__` интерфейсе.

    Параметры:
    ----------
    emoji_dict : dict
        Словарь с описанием значений эмоджи
    slang_dict : dict
        Словарь с описанием значений сленга
    remove_social : bool
        Если True, убирает URL-ы, хэштеги, and @упоминания (полезно для Twitter-датасетов).
    """

    def __init__(
        self,
        emoji_dict: dict[str, str],
        slang_dict: dict[str, str],
        remove_social: bool = True,
    ):
        self.emoji_dict = emoji_dict
        self.slang_dict = slang_dict
        self.remove_social = remove_social

    def __call__(self, text: str) -> str:
        """Запускает полный пайплайн предобработки для одной строки"""
        text = text.lower()
        if self.remove_social:
            text = remove_urls(text)
            text = remove_hashtags(text)
            text = remove_mentions(text)
        text = replace_emojis(text, self.emoji_dict)
        text = expand_slang(text, self.slang_dict)
        text = remove_punctuation(text)
        text = collapse_whitespace(text)
        return text

    @classmethod
    def from_csv(
        cls,
        emoji_csv: str,
        slang_csv: str,
        lang: str = "en",
        remove_social: bool = True,
    ) -> "TextPreprocessor":
        """
        Конструктор — создает словари эмоджи и сленга из CSV-файлов по прописанным путям.

        Parameters
        ----------
        emoji_csv  : путь к emoji CSV
        slang_csv  : путь к slang CSV
        lang       : 'en' или 'ru' (определяет колонку описания в emoji CSV)
        """
        emoji_dict = load_emoji_dict(emoji_csv, lang=lang)
        slang_dict = load_slang_dict(slang_csv)
        return cls(emoji_dict, slang_dict, remove_social=remove_social)
