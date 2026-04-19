"""
predict.py
----------
Интерфейс для инференса

Автоматически определяет язык, загружает нужную модель,
и возвращает метку тональности с показателями уверенности.

Использование:
-----
    from predict import SentimentAnalyzer

    analyzer = SentimentAnalyzer(
        en_weights="checkpoints/bert_model_eng.pth",
        ru_weights="checkpoints/bert_model_rus.pth",
        en_emoji_csv="data/emoji/emoji_df.csv",
        en_slang_csv="data/slang/slang.csv",
        ru_emoji_csv="data/emoji/emoji_dataset_russian.csv",
        ru_slang_csv="data/slang/russian_slang.csv",
    )

    result = analyzer.predict("This movie was absolutely fantastic!")
    # {"text": "...", "language": "en", "sentiment": "Positive", "confidence": 0.94,
    #  "scores": {"Negative": 0.02, "Neutral": 0.04, "Positive": 0.94}}
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from langdetect import detect
from transformers import AutoTokenizer

from model import (
    EN_ID2LABEL,
    RU_ID2LABEL,
    load_model,
)
from preprocessing import TextPreprocessor


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

EN_MODEL_NAME = "bert-base-uncased"
RU_MODEL_NAME = "seara/rubert-tiny2-russian-sentiment"

EN_MAX_LEN = 256
RU_MAX_LEN = 512


# ---------------------------------------------------------------------------
# Определение языка
# ---------------------------------------------------------------------------

def detect_language(text: str) -> str:
    """
    Определяет, является ли язык текста русским ('ru') или английским ('en').

    Возвращает 'ru', 'en', или вызывает ValueError для языков, которые не поддерживаются.
    """
    try:
        lang = detect(text)
    except Exception as e:
        raise ValueError(f"Language detection failed: {e}") from e

    if lang == "ru":
        return "ru"
    if lang == "en":
        return "en"
    raise ValueError(
        f"Unsupported language detected: '{lang}'. "
        "This analyzer supports Russian ('ru') and English ('en') only."
    )


# ---------------------------------------------------------------------------
# Главный класс инференса
# ---------------------------------------------------------------------------

class SentimentAnalyzer:
    """
    Анализирует тональность с учетом языка.

    Автоматически выбирает английскую или русскую модель по входящему тексту,
    возвращает структурированное предсказание с показателями уверенности.

    Параметры:
    ----------
    en_weights    : Путь к state_dict английской модели (.pth)
    ru_weights    : Путь к state_dict русской модели (.pth)
    en_emoji_csv  : Путь к CSV-файлу эмоджи английской модели
    en_slang_csv  : Путь к CSV-файлу сленга английской модели
    ru_emoji_csv  : Путь к CSV-файлу эмоджи русской модели
    ru_slang_csv  : Путь к CSV-файлу сленга русской модели
    device        : torch device (определяется автоматически, если None)

    Примечание:
    -----
    Модели загружаются лениво при первом использовании для сохранения памяти,
    если нужен только один язык.
    """

    def __init__(
        self,
        en_weights: str,
        ru_weights: str,
        en_emoji_csv: str,
        en_slang_csv: str,
        ru_emoji_csv: str,
        ru_slang_csv: str,
        device: torch.device | None = None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._en_weights = en_weights
        self._ru_weights = ru_weights

        # Предобработка
        self._preprocessors = {
            "en": TextPreprocessor.from_csv(en_emoji_csv, en_slang_csv, lang="en"),
            "ru": TextPreprocessor.from_csv(ru_emoji_csv, ru_slang_csv, lang="ru"),
        }

        # Токенизация
        self._tokenizers = {
            "en": AutoTokenizer.from_pretrained(EN_MODEL_NAME),
            "ru": AutoTokenizer.from_pretrained(RU_MODEL_NAME),
        }

        self._max_lens = {"en": EN_MAX_LEN, "ru": RU_MAX_LEN}
        self._id2label = {"en": EN_ID2LABEL, "ru": RU_ID2LABEL}

        # Кэш моделей с ленивой загрузкой
        self._models: dict[str, object] = {}

    def _get_model(self, lang: str):
        """Загружает модель при первом обращении и помещает в кэш"""
        if lang not in self._models:
            weights = self._en_weights if lang == "en" else self._ru_weights
            model_name = EN_MODEL_NAME if lang == "en" else RU_MODEL_NAME
            self._models[lang] = load_model(
                model_name, weights, lang=lang, device=self.device
            )
        return self._models[lang]

    def predict(self, text: str, lang: str | None = None) -> dict:
        """
        Предсказывает тональность для одиночного текста.

        Параметры:
        ----------
        text : сырой входной текст (эмоджи, сленг и любой регистр принимаются как есть)
        lang : 'en' или 'ru'; если None, язык определяется автоматически

        Возвращает:
        -------
        dict с полями:
            text        — входной текст
            language    — обнаруженный или переданный код языка
            sentiment   — предсказание класса (например, 'Positive')
            confidence  — вероятность класса (0–1)
            scores      — dict с вероятностью для каждого класса
        """
        if lang is None:
            lang = detect_language(text)

        preprocessor = self._preprocessors[lang]
        tokenizer = self._tokenizers[lang]
        max_len = self._max_lens[lang]
        id2label = self._id2label[lang]
        model = self._get_model(lang)

        clean_text = preprocessor(text)

        encoding = tokenizer(
            clean_text,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_token_type_ids=False,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = F.softmax(outputs.logits, dim=1).squeeze()

        pred_idx = probs.argmax().item()
        scores = {id2label[i]: round(probs[i].item(), 4) for i in range(len(id2label))}

        return {
            "text": text,
            "language": lang,
            "sentiment": id2label[pred_idx],
            "confidence": round(probs[pred_idx].item(), 4),
            "scores": scores,
        }

    def predict_batch(
        self, texts: list[str], lang: str | None = None
    ) -> list[dict]:
        """
        Предсказывает тональность для множества текстов.

        Каждый текст обрабатывается независимо (язык определяется для каждого текста отдельно,
        если lang равен None).
        """
        return [self.predict(t, lang=lang) for t in texts]