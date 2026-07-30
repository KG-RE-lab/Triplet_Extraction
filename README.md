# PLGF: A Parallel Local–Global Feature Learning Framework with Separated Table Modeling and Enhanced Focal Loss for Imbalanced Relation Extraction

> A table-filling-based relation extraction framework that integrates **Separated Table Modeling (STM)**, **Parallel Local–Global Feature Learning**, and **Enhanced Focal Loss (EFL)** to alleviate label imbalance and improve relation extraction performance.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Environment and Dependencies](#environment-and-dependencies)
- [Dataset Preparation](#dataset-preparation)
- [Pre-trained BERT](#pre-trained-bert)
- [Usage](#usage)
- [Model Overview](#model-overview)
- [Evaluation](#evaluation)
- [Acknowledgements](#acknowledgements)

---

## Project Structure

```text
PLGF/
├── README.md                 # Documentation
├── bert-large-cased/         # Pre-trained BERT model
├── dataset/                  # Datasets
│   ├── NYT/
│   ├── NYT_star/
│   ├── WebNLG/
│   └── WebNLG_star/
└── code/
    ├── run.py                # Entry script
    ├── main.py               # Training/testing pipeline
    ├── train.py              # Training procedure
    ├── model.py              # PLGF model
    ├── modules.py            # Cross-attention, GNN, multi-scale modules
    ├── dataloader.py         # Data loading
    ├── evaluation_utils.py   # Evaluation
    ├── util.py               # Utilities
    └── requirements.txt      # Dependencies
```

---

## Environment and Dependencies

Install all required packages:

```bash
cd code
pip install -r requirements.txt
```

Main dependencies include:

- Python 3.8+
- PyTorch
- Transformers
- bert4keras

See `code/requirements.txt` for the complete dependency list.

---

## Dataset Preparation

The experiments are conducted on the following public datasets.

| Dataset | Source |
| :------ | :----- |
| NYT / NYT* | CasRel, CopyRE |
| WebNLG / WebNLG* | JointER, ETL-span |

Each dataset directory should contain:

```text
train.json
dev.json
test.json
rel2id.json
```

Optional evaluation subsets include:

```text
1.json
2.json
3.json
4.json
5.json
epo.json
normal.json
seo.json
```

---

## Pre-trained BERT

Download **BERT-Large-Cased** (or another compatible BERT checkpoint) from Hugging Face:

https://huggingface.co/bert-large-cased

Place the model under

```text
bert-large-cased/
```

The directory should contain:

```text
config.json
vocab.txt
pytorch_model.bin
```

Alternatively, specify the checkpoint path through the command-line argument.

---

## Usage

All commands should be executed inside the `code/` directory.

### Training

Example:

```bash
# WebNLG
python run.py --dataset=WebNLG --file_id=WebNLG --train=train --cuda_id=0

# WebNLG*
python run.py --dataset=WebNLG_star --file_id=WebNLG_star --train=train

# NYT
python run.py --dataset=NYT --file_id=NYT --train=train

# NYT*
python run.py --dataset=NYT_star --file_id=NYT_star --train=train
```

### Common Arguments

| Argument | Description | Default |
| :------- | :---------- | :------ |
| `--cuda_id` | GPU device ID | `3` |
| `--train_batch_size` | Batch size | `6` |
| `--learning_rate` | Learning rate | `2e-5` |
| `--num_train_epochs` | Number of epochs | `100` |
| `--bert_model_path` | Path to BERT weights | `../bert-large-cased/pytorch_model.bin` |
| `--base_path` | Dataset directory | `../dataset` |
| `--loss` | Loss function | `FocalLoss_plus` (`CE`, `FocalLoss` also supported) |
| `--ablate` | Disable specific modules for ablation | `cross_attention`, `reasoning`, `table_enhance` |

### Ablation Examples

Disable the cross-attention module:

```bash
python run.py --dataset=WebNLG --train=train --ablate cross_attention
```

Disable both cross-attention and table enhancement:

```bash
python run.py --dataset=WebNLG --train=train --ablate cross_attention table_enhance
```

---

## Testing

```bash
python run.py --dataset=WebNLG --file_id=WebNLG --train=test
```

Predictions and evaluation results will be saved under

```text
dataset/.../results/<file_id>/
```

---

## Model Overview

PLGF is a table-filling-based framework for joint relation extraction that addresses the severe class imbalance inherent in conventional table representations.

The framework consists of three key components:

### 1. Separated Table Modeling (STM)

Instead of representing entities and relations within a single dense tensor, STM decomposes the original table into separate entity-boundary and relation-classification tables. This redesign significantly reduces redundant negative labels and alleviates label imbalance at the representation level.

### 2. Parallel Local–Global Feature Learning

PLGF jointly models semantic dependencies from different perspectives through a parallel architecture consisting of:

- Cross-attention interaction
- Multi-scale contextual feature extraction
- Graph message passing

These modules enable the model to capture both local contextual information and long-range relational dependencies.

### 3. Enhanced Focal Loss (EFL)

An enhanced focal loss based on the probability margin is introduced to improve optimization under highly imbalanced label distributions. It emphasizes ambiguous samples while suppressing overconfident predictions, resulting in more stable training.

---

## Evaluation

The model is evaluated using three metrics.

| Metric | Description |
| :----- | :---------- |
| Strict Triple | Correct head entity, tail entity, and relation |
| Entity Pair | Correct entity pair regardless of relation |
| Relation | Correct relation classification |

Precision, Recall, and F1-score are reported for each metric.

---

## Acknowledgements

This implementation is built upon several excellent open-source projects.

- bert4keras
- Hugging Face Transformers
- CasRel
- CopyRE
- JointER
- ETL-span

We sincerely thank the authors for making their code and datasets publicly available.
