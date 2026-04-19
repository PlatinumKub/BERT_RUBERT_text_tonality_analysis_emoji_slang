"""
config.py
---------
Единственное место для всех гиперпараметров и путей.

Хочешь поменять learning rate? Идёшь сюда.
Хочешь сменить путь к данным? Сюда.
Нигде больше магических чисел нет.

Использование:
    from config import cfg
    print(cfg.en.model_name)
    print(cfg.data.emoji_csv)
"""

from dataclasses import dataclass, field
from pathlib import Path

# Корень проекта — директория, где лежит этот файл
ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Пути к данным
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    # Один CSV для эмодзи на оба языка:
    #   - для EN используется колонка 'name'
    #   - для RU используется колонка 'description_ru'
    emoji_csv: str = str(ROOT / "data" / "emoji-with-descriptions-en-ru.csv")

    en_slang_csv: str = str(ROOT / "data" / "slang_en.csv")
    ru_slang_csv: str = str(ROOT / "data" / "slang_ru.csv")

    # Локальные CSV для обучения на английском
    # Ожидаемые колонки: text_col (см. EnglishTrainConfig) + label_col
    en_tweets_csv: str = str(ROOT / "data" / "en_tweets.csv")
    en_corona_train_csv = str(ROOT / "data" / "en_corona_train.csv")
    en_corona_test_csv = str(ROOT / "data" / "en_corona_test.csv")

    # HuggingFace датасет для русского
    ru_hf_dataset: str = "MonoHime/ru_sentiment_dataset"


# ---------------------------------------------------------------------------
# Конфигурация обучения
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Параметры, общие для обоих языков."""
    seed: int = 42
    val_split: float = 0.10   # доля от train → validation
    test_split: float = 0.10  # доля от полного датасета → test
    max_grad_norm: float = 1.0


# ---------------------------------------------------------------------------
# Конфигурации моделей
# ---------------------------------------------------------------------------

@dataclass
class EnglishModelConfig:
    # Архитектура
    model_name: str = "bert-base-uncased"
    num_labels: int = 3

    # Токенизация
    max_len: int = 256

    # DataLoader
    batch_size: int = 16
    num_workers: int = 0

    # Оптимизация
    lr: float = 2e-6
    epochs: int = 3

    # Колонки датасета
    text_col: str = "OriginalTweet"
    label_col: str = "Sentiment"

    # Где сохранять лучшую модель
    checkpoint_dir: str = str(ROOT / "checkpoints")
    checkpoint_name: str = "bert_model_eng.pth"

    @property
    def checkpoint_path(self) -> str:
        return str(Path(self.checkpoint_dir) / self.checkpoint_name)

    # Где сохранять историю
    history_dir: str = str(ROOT / "history")
    history_name: str = "history_en.json"

    @property
    def history_path(self) -> str:
        return str(Path(self.history_dir) / self.history_name)


@dataclass
class RussianModelConfig:
    # Архитектура
    model_name: str = "seara/rubert-tiny2-russian-sentiment"
    num_labels: int = 3

    # Токенизация
    max_len: int = 512

    # DataLoader
    batch_size: int = 32
    num_workers: int = 0

    # Оптимизация
    lr: float = 2e-6
    epochs: int = 3

    # Колонки датасета
    text_col: str = "text"
    label_col: str = "label"

    # Где сохранять лучшую модель
    checkpoint_dir: str = str(ROOT / "checkpoints")
    checkpoint_name: str = "bert_model_rus.pth"

    @property
    def checkpoint_path(self) -> str:
        return str(Path(self.checkpoint_dir) / self.checkpoint_name)

    # Где сохранять историю
    history_dir: str = str(ROOT / "history")
    history_name: str = "history_ru.json"

    @property
    def history_path(self) -> str:
        return str(Path(self.history_dir) / self.history_name)


# ---------------------------------------------------------------------------
# Корневой конфиг — единственный объект, который импортируют другие модули
# ---------------------------------------------------------------------------

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    en: EnglishModelConfig = field(default_factory=EnglishModelConfig)
    ru: RussianModelConfig = field(default_factory=RussianModelConfig)


# Singleton — весь проект делает `from config import cfg`
cfg = Config()


# ---------------------------------------------------------------------------
# Небольшой smoke-test при запуске напрямую
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Config smoke-test ===")
    print(f"EN model  : {cfg.en.model_name}")
    print(f"EN max_len: {cfg.en.max_len}")
    print(f"EN lr     : {cfg.en.lr}")
    print(f"EN ckpt   : {cfg.en.checkpoint_path}")
    print()
    print(f"RU model  : {cfg.ru.model_name}")
    print(f"RU max_len: {cfg.ru.max_len}")
    print(f"RU lr     : {cfg.ru.lr}")
    print(f"RU ckpt   : {cfg.ru.checkpoint_path}")
    print()
    print(f"Emoji CSV : {cfg.data.emoji_csv}")
    print(f"Slang EN CSV : {cfg.data.en_slang_csv}")
    print(f"Slang RU CSV : {cfg.data.ru_slang_csv}")
    print(f"En_Tweets CSV : {cfg.data.en_tweets_csv}")
    print(f"En_Corona_test CSV : {cfg.data.en_corona_test_csv}")
    print(f"En_Corona_train CSV : {cfg.data.en_corona_train_csv}")
    print(f"Seed      : {cfg.train.seed}")
