"""
Training script for MalBERT - BERT-based malware detection.
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path


class MalwareDataset(Dataset):
    """Dataset for PE malware detection."""

    def __init__(self, data_path, tokenizer, max_length=512):
        self.data = self._load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _load_data(self, data_path):
        # Load data from file
        # Replace with actual data loading
        return []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class MalBERTClassifier(nn.Module):
    """BERT-based malware classifier."""

    def __init__(self, model_name='distilbert-base-uncased', num_classes=2, dropout=0.3):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(dataloader), correct / total


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # For demonstration, create synthetic data
    # Replace with actual data loading
    print("Loading data...")

    # This is a placeholder - replace with actual tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    # Create model
    model = MalBERTClassifier(
        model_name=args.model_name,
        num_classes=2,
        dropout=args.dropout
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Training loop
    best_val_acc = 0

    for epoch in range(args.epochs):
        # In practice, load actual data here
        print(f"Epoch {epoch + 1}/{args.epochs}")

        # Save model checkpoint
        if (epoch + 1) % 10 == 0:
            save_path = Path(args.save_dir) / f"malbert_epoch_{epoch+1}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  Saved: {save_path}")

    print("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MalBERT for malware detection')
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased')
    parser.add_argument('--data_path', type=str, default='data/malware_dataset.pt')
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    args = parser.parse_args()
    main(args)
