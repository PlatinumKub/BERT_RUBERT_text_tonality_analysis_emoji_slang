import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "defs"))

import pandas as pd
from sklearn.metrics import accuracy_score
from defs.predict import SentimentAnalyzer
from defs.preprocessing import TextPreprocessor, load_emoji_dict, load_slang_dict

# ---- ПУТИ ----
EN_WEIGHTS   = os.path.join(ROOT, "checkpoints", "bert_model_eng.pth")
RU_WEIGHTS   = os.path.join(ROOT, "checkpoints", "bert_model_rus.pth")
EN_EMOJI_CSV = os.path.join(ROOT, "data", "emoji-with-descriptions-en-ru.csv")
EN_SLANG_CSV = os.path.join(ROOT, "data", "slang_en.csv")
RU_EMOJI_CSV = os.path.join(ROOT, "data", "emoji-with-descriptions-en-ru.csv")
RU_SLANG_CSV = os.path.join(ROOT, "data", "slang_ru.csv")

# ---- ДАННЫЕ ----
en_df     = pd.read_csv(os.path.join(ROOT, "checkpoints", "test_en.csv"))
en_texts  = en_df["OriginalTweet"].tolist()
en_labels = en_df["Sentiment"].astype(int).tolist()

ru_df     = pd.read_csv(os.path.join(ROOT, "checkpoints", "test_ru.csv"))
ru_texts  = ru_df["text"].tolist()
ru_labels = ru_df["label"].astype(int).tolist()

# ---- КОНФИГИ ----
CONFIGS = {
    "baseline":   {"use_emoji": False, "use_slang": False},
    "emoji_only": {"use_emoji": True,  "use_slang": False},
    "slang_only": {"use_emoji": False, "use_slang": True},
    "full":       {"use_emoji": True,  "use_slang": True},
}

# ---- ЗАПУСК ----
def run_ablation(lang, texts, labels, emoji_csv, slang_csv, weights_path, model_name, label_map):

    if lang == "en":
        analyzer = SentimentAnalyzer(
            en_weights=weights_path,
            ru_weights=RU_WEIGHTS,
            en_emoji_csv=emoji_csv,
            en_slang_csv=slang_csv,
            ru_emoji_csv=RU_EMOJI_CSV,
            ru_slang_csv=RU_SLANG_CSV,
        )
    else:
        analyzer = SentimentAnalyzer(
            en_weights=EN_WEIGHTS,
            ru_weights=weights_path,
            en_emoji_csv=EN_EMOJI_CSV,
            en_slang_csv=EN_SLANG_CSV,
            ru_emoji_csv=emoji_csv,
            ru_slang_csv=slang_csv,
        )

    results = {}

    for config_name, flags in CONFIGS.items():
        print(f"\n[{lang.upper()}] Config: {config_name}")

        analyzer._preprocessors[lang] = TextPreprocessor(
            emoji_dict=load_emoji_dict(emoji_csv, lang=lang),
            slang_dict=load_slang_dict(slang_csv),
            use_emoji=flags["use_emoji"],
            use_slang=flags["use_slang"],
        )

        preds = []
        for i, text in enumerate(texts):
            result = analyzer.predict(text, lang=lang)
            preds.append(result["sentiment"])
            if i % 100 == 0:
                print(f"  {i}/{len(texts)}", end="\r")

        preds_num = [label_map[p] for p in preds]

        acc = accuracy_score(labels, preds_num)
        results[config_name] = round(acc, 4)
        print(f"  Accuracy: {acc:.4f}")

    return results


# ---- ВЫЗОВЫ ----
en_results = run_ablation(
    lang="en",
    texts=en_texts,
    labels=en_labels,
    emoji_csv=EN_EMOJI_CSV,
    slang_csv=EN_SLANG_CSV,
    weights_path=EN_WEIGHTS,
    model_name="bert-base-uncased",
    label_map={"Negative": 0, "Neutral": 1, "Positive": 2},
)

ru_results = run_ablation(
    lang="ru",
    texts=ru_texts,
    labels=ru_labels,
    emoji_csv=RU_EMOJI_CSV,
    slang_csv=RU_SLANG_CSV,
    weights_path=RU_WEIGHTS,
    model_name="seara/rubert-tiny2-russian-sentiment",
    label_map={"Neutral": 0, "Positive": 1, "Negative": 2},
)

# ---- ТАБЛИЦА РЕЗУЛЬТАТОВ ----
print("\n\n===== ABLATION RESULTS =====")
print(f"{'Config':<15} {'EN':>8} {'RU':>8}")
print("-" * 33)
for config in CONFIGS:
    print(f"{config:<15} {en_results[config]:>8.4f} {ru_results[config]:>8.4f}")