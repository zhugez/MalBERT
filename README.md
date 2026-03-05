# MalBERT

BERT-based transformer models for malware detection from PE strings.

## Overview

- Use pre-trained BERT/DistilBERT for malware classification
- Fine-tune on PE string features
- Compare with custom tokenizers

## Quick Start

```bash
pip install transformers torch
python scripts/train_bert.py --model distilbert
```

## Architecture

- Input: PE strings → tokenized
- Backbone: BERT/DistilBERT
- Classification head: Linear + Softmax

## Paper Outline

See `docs/paper_outline.md`
