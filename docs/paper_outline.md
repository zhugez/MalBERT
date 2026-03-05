# MalBERT Paper Outline

## Title
> MalBERT: Transformer-Based Malware Detection from PE Strings

## Sections

### 1. Introduction
- PE strings contain rich semantic information
- Transformers can capture long-range dependencies in strings
- Pre-trained models provide better representations

### 2. Related Work
- CNN/RNN for malware
- BERT in security domain
- PE string analysis

### 3. Methodology
- PE string extraction → tokenizer → BERT → classifier
- Pre-training on large corpus
- Fine-tuning strategies

### 4. Experiments
- Dataset: SOREL-20M subset
- Baselines: CNN, LSTM, Random Forest
- Comparison with LVForge DML approach

### 5. Results
- Accuracy, F1, AUC comparison
- Attention visualization
- Transfer learning analysis
