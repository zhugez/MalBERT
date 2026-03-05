"""
BERT-based models for malware detection.
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig


class MalBERTClassifier(nn.Module):
    """BERT-based malware classifier."""

    def __init__(self, model_name: str = 'distilbert-base-uncased',
                 num_classes: int = 2, dropout: float = 0.3,
                 freeze_bert: bool = False):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


class MalBERTWithFeatures(nn.Module):
    """BERT with additional PE features."""

    def __init__(self, model_name: str = 'distilbert-base-uncased',
                 num_classes: int = 2, num_features: int = 50,
                 dropout: float = 0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.feature_extractor = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

        # Fuse BERT (768) + features (64) = 832
        self.classifier = nn.Linear(self.bert.config.hidden_size + 64, num_classes)

    def forward(self, input_ids, attention_mask, pe_features):
        # BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_output = outputs.last_hidden_state[:, 0, :]

        # Feature encoding
        feature_output = self.feature_extractor(pe_features)

        # Fuse
        combined = torch.cat([bert_output, feature_output], dim=1)
        combined = self.dropout(combined)
        logits = self.classifier(combined)
        return logits


class TinyBERTClassifier(nn.Module):
    """Lightweight BERT for resource-constrained environments."""

    def __init__(self, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        config = DistilBertConfig(
            n_layers=4,
            n_heads=4,
            hidden_dim=256,
            dim=256,
            dropout=dropout
        )
        self.bert = DistilBertModel(config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


def get_model(model_name: str, **kwargs):
    """Get model by name."""
    models = {
        'malbert': MalBERTClassifier,
        'malbert_features': MalBERTWithFeatures,
        'tinybert': TinyBERTClassifier,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")
    return models[model_name](**kwargs)
