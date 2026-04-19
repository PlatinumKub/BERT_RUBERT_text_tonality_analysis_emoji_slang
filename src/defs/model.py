"""
model.py
--------
Кастомный BERT-based классификатор для 3-классового анализа тональности текста.

Архитектура:
    BERT encoder (pooled [CLS] output)
    → Dropout
    → Линейный слой (hidden_size, num_labels)

Работает с любыми BERT-family чекпоинтами из HuggingFace Hub
(bert-base-uncased, seara/rubert-tiny2-russian-sentiment, etc.)
"""

import torch
import torch.nn as nn
from transformers import AutoConfig, BertModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


# ---------------------------------------------------------------------------
# Маппинг меток
# ---------------------------------------------------------------------------

EN_ID2LABEL = {0: "Negative", 1: "Neutral", 2: "Positive"}
EN_LABEL2ID = {v: k for k, v in EN_ID2LABEL.items()}

RU_ID2LABEL = {0: "Neutral", 1: "Positive", 2: "Negative"}
RU_LABEL2ID = {v: k for k, v in RU_ID2LABEL.items()}


# ---------------------------------------------------------------------------
# Модель
# ---------------------------------------------------------------------------

class BertForSentiment(BertPreTrainedModel):
    """
    BERT fine-tuned для 3-классового анализа тональности текста.

    Получает pooled [CLS] представление токенов, добавляет dropout,
    и проецирует в пространство num_labels классов через линейный слой.

    Параметры унаследованы из BertPreTrainedModel; num_labels
    задаётся через config.num_labels (по умолчанию 3).
    """

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        """
        Forward pass.

        Параметры:
        ----------
        input_ids       : (batch, seq_len)
        attention_mask  : (batch, seq_len)  — 1 для реальных токенов, 0 для padding
        token_type_ids  : (batch, seq_len)  — опционально, для NSP задач
        labels          : (batch)          — опционально; если передан (не None), считаем loss

        Возвращает:
        -------
        SequenceClassifierOutput с полями:
            .loss    — CrossEntropy loss (None если labels не передан)
            .logits  — (batch, num_labels) сырые логиты
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        # outputs[1] - это pooled [CLS] представление
        pooled = self.dropout(outputs[1])
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# ---------------------------------------------------------------------------
# Инициализация и загрузка модели
# ---------------------------------------------------------------------------

def build_model(
    model_name: str,
    lang: str = "en",
    device: torch.device | None = None,
) -> BertForSentiment:
    """
    Инициализирует BertForSentiment на основе HuggingFace чекпойнта.

    Параметры:
    ----------
    model_name : HuggingFace model ID или локальный путь
    lang       : 'en' or 'ru' — выбираем язык
    device     : torch device; по умолчанию CUDA если доступна, иначе CPU

    Возвращает:
    -------
    BertForSentiment, готовая для обучения или дообучения
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    id2label = EN_ID2LABEL if lang == "en" else RU_ID2LABEL
    label2id = EN_LABEL2ID if lang == "en" else RU_LABEL2ID

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
    )
    model = BertForSentiment.from_pretrained(model_name, config=config)
    return model.to(device)


def load_model(
    model_name: str,
    weights_path: str,
    lang: str = "en",
    device: torch.device | None = None,
) -> BertForSentiment:
    """
    Загружает обученную BertForSentiment модель из state_dict.

    Параметры:
    ----------
    model_name   : HuggingFace model ID (должна совпадать с архитектурой, используемой во время обучения)
    weights_path : путь к .pth файлу, сохранённому через torch.save(model.state_dict(), ...)
    lang         : 'en' or 'ru'
    device       : torch device

    Возвращает:
    -------
    BertForSentiment в eval mode
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(model_name, lang=lang, device=device)
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model
