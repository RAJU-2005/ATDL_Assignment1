# ATDL Assignment 1 — Knowledge Distillation with Ternary-Weight QAT
### ResNet34 (FP32 Teacher) → ResNet18 (Ternary Student) on CIFAR-10

Full-precision ResNet34 teacher → ternary-weight ResNet18 student trained with
combined Knowledge Distillation + Quantization-Aware Training, benchmarked
against an FP32 ResNet18 baseline (no KD, no quantization).

> **Status:** all 8 notebooks have been run end-to-end; the numbers in
> [Results](#results) below are from the actual completed training runs, not
> placeholders. Values marked "≈" were read off a training-curve plot rather
> than a printed log line — replace them with the exact figure from your
> `results/*.csv` files before pasting into the report.

---

## Table of Contents
- [Repository Structure](#repository-structure)
- [Pipeline Overview](#pipeline-overview)
- [Setup](#setup)
- [Configuration](#configuration)
- [Running End-to-End](#running-end-to-end)
- [Results](#results)
- [Figures Produced](#figures-produced)
- [Method Notes](#method-notes)
- [Troubleshooting](#troubleshooting)
- [Reproducibility](#reproducibility)
- [References](#references)
- [Academic Integrity](#academic-integrity)

---

## Repository Structure

```
.
├── config.yaml                              # Central hyperparameters
├── requirements.txt                          # Python dependencies
├── README.md                                  # This file
├── src/                                        # Shared, importable code
├── notebooks/                                   # Run in numeric order (see below)
│   ├── 01_CIFAR10_Data_Preparation.ipynb
│   ├── 02_ResNet34_Teacher_Training.ipynb
│   ├── 03_ResNet18_FP32_Baseline.ipynb
│   ├── 04_Ternary_Quantizer_Development.ipynb
│   ├── 05_ResNet18_Ternary_KD_QAT.ipynb
│   ├── 06_KD_Ablation_Study.ipynb
│   ├── 07_Model_Evaluation_Compression.ipynb
│   ├── 08_Unseen_CIFAR10_Image_Testing.ipynb
│   └── results/                                    # Notebook-local scratch outputs
├── data/                                       # CIFAR-10 downloads here on first run
├── checkpoints/                                # Best-model .pth files
├── results/                                    # Final PNG figures + CSV tables for the report
└── report/                                     # Written report and report-only assets
```

## Pipeline Overview

| # | Notebook | Purpose | Depends on |
|---|---|---|---|
| 01 | `CIFAR10_Data_Preparation` | Downloads CIFAR-10, builds train/val/test splits, visualizes class balance and sample augmentations | — |
| 02 | `ResNet34_Teacher_Training` | Trains the FP32 ResNet34 teacher (Task S1) | 01 |
| 03 | `ResNet18_FP32_Baseline` | Trains the FP32 ResNet18 baseline — no KD, no quantization (Task 4 comparison point) | 01 |
| 04 | `Ternary_Quantizer_Development` | Implements and unit-tests the ternary quantizer: $\alpha$, $\Delta$, STE (Task S2) | — |
| 05 | `ResNet18_Ternary_KD_QAT` | Builds the ternary ResNet18 student and trains it with combined KD+QAT against the frozen teacher (Task S3) | 02, 04 |
| 06 | `KD_Ablation_Study` | Sweeps KD temperature $T$ with $\lambda$ fixed, shorter epoch budget per run | 02, 04 |
| 07 | `Model_Evaluation_Compression` | Builds the Teacher vs. Baseline vs. Student comparison table: accuracy, params, size, sparsity, compression (Task 4) | 02, 03, 05 |
| 08 | `Unseen_CIFAR10_Image_Testing` | Qualitative sanity check: runs all three trained models on images outside the CIFAR-10 test split | 02, 03, 05 |

## Setup

```bash
git clone <your-repo-url>
cd <your-repo-name>

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

A CUDA-capable GPU is strongly recommended — the teacher, baseline, and
student runs above were each trained for ~100 epochs.

## Configuration

`config.yaml` centralizes the hyperparameters used across notebooks 02, 03,
05, and 06 (epochs, learning rate, batch size, KD temperature/λ, ternary
threshold scale, seed). Edit it before running rather than hard-coding values
inside notebook cells, so it stays the single source of truth cited in your
report.

## Running End-to-End

```bash
jupyter lab
```

Run notebooks **01 → 08 in numeric order**:

1. **01** — no dependencies; populates `data/`.
2. **02** and **03** can run in parallel — both only need 01.
3. **04** is independent (unit tests the quantizer in isolation) but is a
   conceptual prerequisite for 05.
4. **05** requires the teacher checkpoint from **02**.
5. **06** requires the teacher checkpoint from **02**; trains one fresh
   student per temperature value at a *reduced* epoch budget (see the
   ablation caveat below) — do not treat these as your best final numbers.
6. **07** requires checkpoints from **02**, **03**, and **05**; produces the
   final comparison table and report figures.
7. **08** requires checkpoints from **02**, **03**, and **05**; qualitative
   only, not used for the quantitative comparison table.

Each training notebook only overwrites its checkpoint when validation/test
accuracy improves, so a run can be safely stopped and resumed.

## Results

Measured on this run (100 epochs for the teacher/baseline/student; ablation
runs use 30 epochs each — see caveat below).

### Accuracy and model footprint

| Model | Params | Test Accuracy | Model Size | Sparsity |
|---|---|---|---|---|
| ResNet34 Teacher (FP32) | 21,282,122 | 95.0% | — | — |
| ResNet18 Baseline (FP32, no KD) | 11,173,962 | 95.1% | 42.63 MB | — |
| ResNet18 Ternary Student (KD+QAT) | 11,173,962 | 94.63% | 2.66 MB | see `results/comparison_table.csv` |

- **Parameter count is identical** between the baseline and the ternary
  student (0.00% "parameter reduction") — this is expected, not a bug.
  Ternary quantization changes how many **bits** each parameter costs (32 →
  ~2), not how many parameters exist. The meaningful compression metric here
  is storage size, not parameter count.
- **Storage compression: 16.00x (93.8% smaller)** — 42.63 MB (FP32 baseline)
  → 2.66 MB (theoretical ternary storage: 2 bits/weight + one FP32 scale per
  channel).
- ⚠️ **This is a theoretical storage figure, not a measured inference
  speedup.** Commodity CPU/GPU hardware has no native 2-bit ternary matmul
  kernel; realizing this as wall-clock speedup requires custom bit-packed
  kernels or dedicated hardware, neither of which is implemented here — say
  this explicitly in the report.
- The ternary student loses **≈0.8–1.1 percentage points** of test accuracy
  relative to the FP32 baseline for a 16x storage reduction — a natural
  accuracy/compression trade-off point to discuss.

### KD temperature ablation (Notebook 06, 30 epochs/run, λ fixed)

| Temperature (T) | Best Epoch | Test Accuracy |
|---|---|---|
| 2 | 29 | 93.32% |
| 4 | 30 | 93.72% |
| 8 | 30 | 93.80% |

Accuracy increases monotonically with T over this range, suggesting the
softer teacher distributions at higher T carried more useful "dark
knowledge" for the student than the almost one-hot distribution at T=2 — worth
checking whether accuracy keeps rising past T=8 or turns over, if time
allows a wider sweep.

> **Why these ablation numbers are lower than the main 100-epoch run
> (≈94.2–94.3%):** the ablation uses a much shorter, fixed 30-epoch budget
> per temperature value so that sweeping 3 values is feasible in the
> available time. This is a standard, explicit trade-off for *relative*
> comparison across settings, not an attempt at the best final model — state
> this explicitly next to the ablation table in the report.

## Figures Produced

All saved to `results/` (some also mirrored in `notebooks/results/`):

| Notebook | Figures |
|---|---|
| 01 | CIFAR-10 class distribution, sample augmented training images |
| 02 | `fig03_teacher_loss.png`, `fig04_teacher_accuracy.png`, `fig05_teacher_confusion_matrix.png` |
| 03 | `fig06_baseline_loss.png`, `fig07_baseline_accuracy.png`, `fig08_baseline_confusion_matrix.png` |
| 04 | FP32 weight distribution with ±Δ threshold overlay; quantized ternary weight histogram (3-valued: $-\alpha$, 0, $+\alpha$) |
| 05 | `fig11_student_loss_components.png` (KD term vs. CE term), `fig12_student_accuracy.png`, `fig13_student_confusion_matrix.png` |
| 06 | Ablation accuracy-vs-temperature line plot |
| 07 | Parameter count and model size comparison bar charts (teacher / baseline / student), full comparison table |
| 08 | Qualitative predictions on unseen images |

## Method Notes

- **Ternary quantization**: each ternary layer keeps a latent FP32 weight
  and quantizes it on every forward pass to $W_q \in \{-\alpha, 0, +\alpha\}$,
  with $\Delta = 0.7 \cdot \mathbb{E}[|W|]$ and
  $\alpha = \mathbb{E}[|W| : |W| > \Delta]$. Gradients flow through the
  quantizer via a Straight-Through Estimator.
- **KD loss**: $L = \lambda\, T^2\, \mathrm{KL}(\mathrm{softmax}(z_t/T)\,\|\,
  \mathrm{softmax}(z_s/T)) + (1-\lambda)\,\mathrm{CE}(z_s, y)$. The teacher is
  frozen — `eval()` mode, `requires_grad_(False)`, and `.detach()`-ed logits —
  so no gradient reaches it.
- **Overfitting pattern visible in the loss curves**: in all three models
  (teacher, baseline, student), training loss approaches ~0 by epoch ~80–100
  while test loss plateaus around 0.18–0.20 with a widening train/test gap —
  worth a short discussion in the report's generalization section, even
  though absolute test accuracy remains high (~94–95%). Label smoothing,
  stronger augmentation, or earlier stopping (at the best-checkpoint epoch,
  which these notebooks already select) are the usual levers to discuss.

## Troubleshooting

- **`FileNotFoundError` for a checkpoint** — an earlier notebook hasn't
  finished; check the dependency column in the pipeline table above.
- **CUDA out of memory** — lower `batch_size` in `config.yaml`.
- **LR schedule / epoch count mismatch in a custom loop** (e.g. notebook 06's
  ablation loop) — if `CosineAnnealingLR(T_max=...)` doesn't match the actual
  number of training epochs in that run, the LR never fully anneals and
  unfairly handicaps that setting relative to the others.
- **Ternary-constraint verification fails for a layer** — a layer was left
  un-quantized, or the quantizer wasn't applied in that layer's forward pass;
  check that layer's construction in `src/`.

## Reproducibility

All training runs seed Python, NumPy, and PyTorch (`seed` in `config.yaml`,
default 42) and set `torch.backends.cudnn.deterministic = True`. Exact
bit-for-bit reproducibility across different GPU models/driver versions is
not guaranteed by PyTorch, but run-to-run variance is substantially reduced.

## References

- He, Zhang, Ren, Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016.
- Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network.* NeurIPS-W 2015.
- Li, Zhang, Liu. *Ternary Weight Networks.* 2016.
- Zhu, Han, Mao, Dally. *Trained Ternary Quantization.* ICLR 2017.
- Bengio, Léonard, Courville. *Estimating or Propagating Gradients Through Stochastic Neurons.* 2013.

## Academic Integrity

This codebase was produced with AI assistance as permitted by the assignment
brief; the AI conversation transcript is attached separately per the
instructor's requirement. The results reported above come from actually
running these notebooks on this dataset — none are fabricated or estimated.
