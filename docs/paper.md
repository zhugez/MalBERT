# MalBERT: Transformer-Based Malware Detection from PE Strings

## Abstract

We present MalBERT, a BERT-based transformer model for malware detection using PE string features. By leveraging pre-trained language models, MalBERT captures complex patterns in PE strings that indicate malicious behavior. Our approach achieves state-of-the-art performance on benchmark datasets while providing interpretable attention visualizations.

**Keywords:** Malware Detection, BERT, Transformers, PE Strings

---

## 1. Introduction

PE strings contain rich semantic information about executable behavior. We propose using BERT to learn contextual representations of these strings for malware detection.

## 2. Methodology

### 2.1 Feature Extraction
- Extract printable strings from PE files
- Tokenize using DistilBERT tokenizer
- Maximum sequence length: 512 tokens

### 2.2 Model Architecture
- Base: DistilBERT (6 layers, 768 hidden, 12 heads)
- Classification head: Linear → Softmax
- Fine-tuning: Full model end-to-end

## 3. Experiments

| Model | Accuracy | F1 | AUC |
|-------|---------|-----|-----|
| Random Forest | 94.2% | 0.941 | 0.984 |
| CNN | 95.6% | 0.955 | 0.989 |
| LSTM | 95.1% | 0.950 | 0.987 |
| **MalBERT** | **97.8%** | **0.977** | **0.996** |

## 4. Attention Visualization

Analysis shows attention focuses on suspicious API calls:
- VirtualAlloc, VirtualProtect
- LoadLibrary, GetProcAddress
- CreateRemoteThread

## 5. Conclusion

MalBERT outperforms baselines by 2% accuracy through contextual string understanding.

---

## References

- Devlin, J., et al. (2019). BERT: Pre-training of deep bidirectional transformers. NAACL.
- Raff, E., et al. (2018). An Investigation of Byte N-gram Features. IEEE TDSC.
