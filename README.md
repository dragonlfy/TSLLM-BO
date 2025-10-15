# TSLLM‑BO: Structure‑Aware Behavioral Optimization for Time‑Series Forecasting with LLM‑Enhanced Reinforcement Learning

> **TL;DR** TSLLM‑BO couples large‑language‑model (LLM) priors with structure‑aware reinforcement learning to deliver accurate, stable, and generalizable time‑series forecasts. The repo provides an out‑of‑the‑box pipeline covering (i) supervised pre‑training, (ii) RL behavioral optimization, and (iii) evaluation / inference.

---
## ✨ Dataset
https://github.com/thuml/Time-Series-Library

## ✨ Key Features

| Feature                                          | Description                                                                                                                                      |
| ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **LLM‑based temporal encoder**                   | Leverages pretrained language knowledge to model long‑range dependencies while remaining modality‑agnostic.                                      |
| **Structure‑Aware Behavioral Optimization (BO)** | A novel policy‑gradient variant that integrates domain‑specific structure priors (periodicity, trend, cross‑correlation) into the reward design. |
| **Two‑Stage Training**                           | 1️⃣ *Supervised stage* minimises MSE/MAE; 2️⃣ *RL stage* fine‑tunes via TSLLM‑BO to improve long‑horizon stability & multi‑objective trade‑offs. |
| **Plug‑and‑Play Components**                     | Clean abstractions for `data_provider`, custom `layers`, and `models` make it easy to swap new datasets or backbones.                            |
| **DeepSpeed / Accelerate Ready**                 | `ds_config_zero2.json` + launch scripts enable efficient distributed training on multi‑GPU clusters.                                             |

---

## 🗂️ Directory Layout

```
.
├── data_provider/        # Dataset loaders & data augmentation
├── layers/               # Neural network building blocks (Patch, Attention, etc.)
├── models/               # TSLLM backbone & policy networks
├── scripts/              # Helper scripts (metrics, visualization, export)
├── utils/                # Common utilities (config, logging, checkpoint)
├── ds_config_zero2.json  # DeepSpeed ZeRO‑2 config for memory‑efficient training
├── requirements.txt      # Minimal Python dependencies
├── run_main.py           # Supervised pre‑training entry point
├── run_RL_pretrain.py    # Optional warm‑start RL pre‑training
└── run_RL.py             # Structure‑Aware BO fine‑tuning & evaluation
```

---

## 🚀 Quick Start

### 1. Environment

```bash
# Create & activate a new virtual env (optional)
python -m venv tsllm_env && source tsllm_env/bin/activate

# Install core dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

1. Place raw *.csv* or *.npz* files under `data_provider/raw/` following the naming convention described in `data_provider/README.md` (or write your own loader).
2. Optionally run `scripts/make_dataset.py` to generate cached *.pkl* files for faster I/O.

### 3. Supervised Pre‑Training

```bash
python run_main.py \
    --cfg configs/sft.yaml        # sample YAML listing model & data hyper‑params
```

### 4. RL Behavioral Optimization

```bash
# (Optional) policy/value warm‑start
python run_RL_pretrain.py --cfg configs/rl_warmstart.yaml

# Main BO fine‑tune + evaluation
python run_RL.py --cfg configs/rl_bo.yaml \
                 --deepspeed ds_config_zero2.json
```

Training logs & checkpoints are written to `./output/<run_name>/`.

### 5. Inference

```bash
python scripts/infer.py --ckpt output/best_model.pth --horizon 720
```

---

## ⚙️ Configuration Files

All hyper‑parameters are specified in YAML files under `configs/`. Key fields include:

| Field               | Description                                                     |
| ------------------- | --------------------------------------------------------------- |
| `dataset`           | Path & task (e.g., electricity, traffic)                        |
| `model.*`           | Backbone depth, hidden dims, attention heads                    |
| `rl.reward_weights` | \[trend, seasonal, accuracy] weights for multi‑objective reward |
| `optimizer.*`       | Optimiser, LR schedule, gradient clip                           |
| `trainer.*`         | Epochs, eval interval, early stop tolerance                     |

---

## 📊 Reproducing Paper Results

```bash
bash scripts/run_all.sh   # sequentially executes SFT → RL‑BO across 8 public datasets
```

Our reported scores will be dumped to `results/summary.csv` for easy comparison.

---

## 📄 License

This project is released under the Apache 2.0 License. See `LEGAL.md` for details.

---

## 🙏 Acknowledgements

We thank the authors of **Time‑LLM**, **DeepSpeed**, and the **Time‑Series Library** for their open‑source efforts, which our work builds upon.
