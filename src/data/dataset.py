"""
Dataset class for PE string classification with BERT.
"""

import torch
from torch.utils.data import Dataset
from transformers import DistilBertTokenizer
import pandas as pd


class PEStringDataset(Dataset):
    """Dataset for PE string classification."""

    def __init__(self, data_path: str, tokenizer_name: str = 'distilbert-base-uncased',
                 max_length: int = 512, split: str = 'train'):
        self.data = pd.read_csv(data_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = str(self.data.iloc[idx]['texts'])
        label = int(self.data.iloc[idx]['label'])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


class PEByteDataset(Dataset):
    """Dataset for PE byte sequence classification."""

    def __init__(self, data_path: str, max_length: int = 512, split: str = 'train'):
        self.data = pd.read_csv(data_path)
        self.max_length = max_length
        self.split = split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Convert hex string to integers
        byte_str = str(self.data.iloc[idx]['bytes'])
        bytes_list = [int(byte_str[i:i+2], 16) if i+2 <= len(byte_str) else 0
                      for i in range(0, min(len(byte_str), self.max_length*2), 2)]

        # Pad or truncate
        if len(bytes_list) < self.max_length:
            bytes_list += [0] * (self.max_length - len(bytes_list))
        else:
            bytes_list = bytes_list[:self.max_length]

        label = int(self.data.iloc[idx]['label'])

        return {
            'bytes': torch.tensor(bytes_list, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }
