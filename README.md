# Attention Is All You Need — Replication

> Academic replication project — Université Paris-Dauphine  
> Based on the paper by **Vaswani, Shazeer, Parmar, Uszkoreit, Jones, Gomez, Kaiser & Polosukhin (2017)**

---

## Overview

This project implements from scratch the **Transformer architecture** introduced in:

> *Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems (NeurIPS), 30.*

The paper introduced a sequence-to-sequence architecture based entirely on **self-attention mechanisms**, discarding recurrence (RNN/LSTM) and convolutions entirely. This replication builds the Transformer encoder-decoder stack in PyTorch and trains it on a machine translation task.

---

## Repository Structure

```
.
├── model/                  # Python source files defining the Transformer architecture
│   ├── (attention.py)      # Multi-head self-attention and scaled dot-product attention
│   ├── (encoder.py)        # Encoder stack (N identical layers)
│   ├── (decoder.py)        # Decoder stack with masked self-attention
│   └── (transformer.py)    # Full Transformer model assembly
│
├── training.ipynb          # End-to-end training pipeline: data loading, training loop, evaluation
└── README.md
```

---

## Architecture

The replication covers all core components described in the paper:

**Attention mechanism**
- Scaled dot-product attention: `Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V`
- Multi-head attention with `h` parallel attention heads projected and concatenated

**Encoder**
- Stack of `N = 6` identical layers
- Each layer: Multi-head self-attention → Add & Norm → Feed-forward → Add & Norm

**Decoder**
- Stack of `N = 6` identical layers
- Each layer: Masked multi-head self-attention → Add & Norm → Cross-attention over encoder output → Add & Norm → Feed-forward → Add & Norm

**Positional encoding**
- Sinusoidal encoding: `PE(pos, 2i) = sin(pos / 10000^(2i/dmodel))`

**Default hyperparameters** (as per the paper's base model):

| Parameter | Value |
|---|---|
| Model dimension `dmodel` | 512 |
| Number of layers `N` | 6 |
| Number of attention heads `h` | 8 |
| Feed-forward dimension `dff` | 2048 |
| Dropout | 0.1 |

---

## Getting Started

### Prerequisites

```bash
pip install torch numpy matplotlib
```

A GPU is strongly recommended for training (CUDA-compatible). CPU training is possible but slow.

### Running the Project

Open and run `training.ipynb` sequentially. The notebook covers:

1. **Data preparation** — tokenization and batch construction
2. **Model instantiation** — building the Transformer with configurable hyperparameters
3. **Training loop** — with the learning rate schedule from the paper (warmup steps + decay)
4. **Evaluation** — loss tracking and qualitative translation samples

---

## Key Implementation Details

- The **learning rate schedule** follows the paper's formula: `lr = dmodel^(-0.5) · min(step^(-0.5), step · warmup_steps^(-1.5))`
- **Label smoothing** (`ε = 0.1`) is applied during training as described in the paper
- The **masking** strategy for the decoder (causal mask + padding mask) is implemented to prevent attending to future tokens

---

## Reference

```bibtex
@inproceedings{vaswani2017attention,
  title     = {Attention Is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob
               and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {30},
  year      = {2017}
}
```
