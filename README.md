# Knowledge Distillation + Ternary-Weight QAT on CIFAR-10
### ResNet34 (FP32 Teacher) → ResNet18 (Ternary Student)
ATDL Assignment I

This project trains a full-precision ResNet34 teacher on CIFAR-10, an FP32 ResNet18
baseline (no KD), and a ternary-weight ResNet18 student trained end-to-end with a
combined **Knowledge Distillation (KD) + Quantization-Aware Training (QAT)** objective,
then evaluates and compresses the result. This README documents the exact steps to
reproduce every number and figure reported in `ATDL_Assignment1_Report.docx`.

---

## 1. Expected project structure

The notebooks assume the following layout (they `sys.path.insert(0, "..")` and import
from `src/`, and CIFAR-10 downloads to `../data` relative to the `notebooks/` folder):

```
project_root/
├── data/                          # CIFAR-10 auto-downloads here (created on first run)
├── checkpoints/                   # saved .pth checkpoints (created automatically)
│   ├── teacher_resnet34_best.pth
│   ├── baseline_resnet18_fp32_best.pth
│   ├── student_ternary_resnet18_best.pth
│   └── ablation_temperature_student_temperature_*.pth
├── results/                       # CSVs + saved plots (created automatically)
│   ├── ablation_temperature.csv
│   ├── ablation_temperature_plot.png
│   └── ... (other exported plots)
├── src/                           # project source package (imported by every notebook)
│   ├── config.py                  # TeacherConfig / BaselineConfig / StudentKDQATConfig / AblationConfig
│   ├── data.py                    # get_cifar10_loaders(...)
│   ├── resnet_cifar.py            # resnet34_cifar / resnet18_cifar (CIFAR-adapted, 3x3 stem)
│   ├── ternary.py                 # TernaryConv2d, TernaryLinear, STE quantizer
│   ├── kd_loss.py                 # KDLoss (temperature-scaled KL + hard-label CE)
│   ├── engine.py                  # train / eval loops shared by teacher, baseline, student
│   ├── ablation.py                # run_ablation(...) helper used by Notebook 04
│   ├── evaluate.py                # checkpoint_verification_report, compression metrics
│   └── utils.py                   # set_seed, get_device, count_parameters, plot_training_curves
└── notebooks/
    ├── 01_teacher_training.ipynb
    ├── 02_baseline_resnet18.ipynb
    ├── 03_ternary_student_kd_qat.ipynb
    ├── 04_ablation_study.ipynb
    ├── 05_evaluation_compression.ipynb
    └── 06_Unseen_CIFAR10_Image_Testing.ipynb
```

If your local copy already has `src/`, `checkpoints/`, and `results/` populated (as in
this submission), you can skip straight to **Section 4 (Running the notebooks)** for
whichever stage you want to re-verify — you do not have to retrain from scratch to
inspect an existing checkpoint.

---

## 2. Environment setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # or the CPU/CUDA build matching your machine
pip install jupyter matplotlib pandas numpy

# 3. Launch Jupyter from the project root
jupyter notebook
```

**Hardware:** all reported runs use `device = "cuda"` (see the "Using device:" print at
the top of every notebook). Training on CPU will run but is substantially slower —
each of Notebooks 01/02/03 trains for 160 epochs.

**Reproducibility:** every notebook calls `set_seed(42)` before building the data
loaders or the model. Re-running a notebook from a clean `checkpoints/`/`data/` folder
with an unchanged `src/` package should reproduce results close to those in the report
(exact bit-for-bit reproduction across different GPUs/CUDA/cuDNN versions is not
guaranteed, per standard PyTorch non-determinism caveats).

---

## 3. Required reading (for context on the methods used)

1. Knowledge Distillation — https://www.ibm.com/think/topics/knowledge-distillation
2. Quantization-Aware Training — https://www.ibm.com/think/topics/quantization-aware-training
3. Ternary Weight Networks (Li, Zhang & Liu, 2016) — https://arxiv.org/abs/1605.04711
4. ResNet — https://www.geeksforgeeks.org/deep-learning/residual-networks-resnet-deep-learning/
5. Teacher–Student models — https://amit-s.medium.com/everything-you-need-to-know-about-knowledge-distillation-aka-teacher-student-model-d6ee10fe7276

Full academic citations are in Section 14 of the report.

---

## 4. Running the notebooks end-to-end

Run the notebooks **in this exact order** — each stage depends on a checkpoint saved
by the previous one. Approximate wall-clock times below are for a single modern GPU
(e.g. a T4/V100-class card); scale accordingly for your hardware.

### Stage 1 — `01_teacher_training.ipynb`  (~160 epochs)
Trains the CIFAR-adapted ResNet34 teacher from scratch.
- **Input:** none (CIFAR-10 auto-downloads to `../data` on first run).
- **Output:** `checkpoints/teacher_resnet34_best.pth`, `results/teacher_training_curves.png`.
- **What to check when it finishes:** the printed final block should report best
  validation accuracy, final train accuracy, and test accuracy. In this submission's
  run: **95.50% val / 99.55% train / 94.98% test**, best checkpoint at epoch 147/160.

### Stage 2a — `02_baseline_resnet18.ipynb`  (~160 epochs)
Trains the FP32 ResNet18 baseline (no KD, no quantization) — the "no-compression"
reference point required by the assignment.
- **Input:** none (uses the same CIFAR-10 split).
- **Output:** `checkpoints/baseline_resnet18_fp32_best.pth`, `results/baseline_training_curves.png`.
- **Expected result (this submission):** 95.56% val / 99.56% train / 95.08% test.

### Stage 2b+3 — `03_ternary_student_kd_qat.ipynb`  (~160 epochs)
The main stage: builds the ternary ResNet18 student, loads and **freezes** the Stage-1
teacher checkpoint, and trains the student with the combined KD + ternary-QAT loss
(5-epoch FP32 warm-up, then quantization on).
- **Input:** `checkpoints/teacher_resnet34_best.pth` (must exist — run Stage 1 first).
- **Output:** `checkpoints/student_ternary_resnet18_best.pth`,
  `results/student_kd_qat_training_curves.png`, `results/student_kd_loss_breakdown.png`.
- **Built-in verification:** this notebook (or `src/evaluate.py`, called from Notebook
  05) reloads the saved checkpoint and confirms every quantized layer's forward weight
  is genuinely restricted to `{-α, 0, +α}` — do not skip this check.
- **Expected result (this submission):** 95.10% val / 99.49% train / 94.63% test,
  19/19 ternary layers verified.

### Stage 4a — `04_ablation_study.ipynb`  (configurable epochs per run)
Sweeps KD temperature `T ∈ {1, 2, 4, 8}` with `λ` (`kd_alpha`) held fixed, training a
fresh student from scratch for each `T`. **This is a reduced-budget run for relative
comparison — do not treat it as a replacement for Stage 3's full 160-epoch result.**
- **Input:** `checkpoints/teacher_resnet34_best.pth`.
- **To change the per-run epoch budget** (as was done for this submission, upgrading
  from an initial 30-epoch pilot to a 100-epoch final run), edit the line in cell 5:
  ```python
  base_cfg = StudentKDQATConfig(epochs=100)   # change this value
  ```
- **Output:** `results/ablation_temperature.csv`, `results/ablation_temperature_plot.png`,
  and one checkpoint per temperature under `checkpoints/ablation_temperature_student_*`.
- **Expected result (100-epoch run, this submission):** best val. accuracy in a tight
  94.72–94.92% band across T=1/2/4/8, with T=4 marginally best.

### Stage 4b — `05_evaluation_compression.ipynb`
Loads all three final checkpoints (teacher, baseline, student), recomputes
train/val/test accuracy under an identical protocol, runs the ternary-constraint
verification, and computes parameter counts, theoretical model size, sparsity, and
compression ratios.
- **Input:** all three checkpoints from Stages 1–3.
- **Output:** `results/accuracy_comparison_plot.png`, `results/model_size_comparison_plot.png`,
  and the final comparison table (reproduced as Section 10.1 of the report).
- **Run this notebook last**, after Stages 1–3 — it is the source of the headline
  evaluation table.

### Optional — `06_Unseen_CIFAR10_Image_Testing.ipynb`
Qualitative sanity check: runs the trained student on your own CIFAR-10-style images
(not part of the train/val/test split). Requires you to supply image files locally;
it is not required to reproduce any number in the report and was not run for this
submission (no output cells were produced).

---

## 5. Regenerating the report

The report (`ATDL_Assignment1_Report.docx`) is built from the exact printed outputs
of Notebooks 01–05 and the six PNG figures they export. If you re-run the notebooks
and get different numbers (e.g. a different GPU/seed interaction), update the tables
in the report by hand rather than assuming the existing report still applies —
per the project's academic-integrity requirement, **no number in the report should
be copied forward once the underlying run has changed.**

---

## 6. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `FileNotFoundError: checkpoints/teacher_resnet34_best.pth` | Ran Stage 2/3 before Stage 1 | Run `01_teacher_training.ipynb` to completion first |
| CUDA out of memory | Batch size (128) too large for your GPU | Lower `batch_size` in the relevant `Config` object |
| Validation accuracy collapses to ~10–15% for the first few epochs of Notebook 03 | **Expected** — this is the FP32 warm-up window (`quant=OFF`) before quantization switches on at epoch 6; accuracy recovers sharply once `quant=ON` | No action needed; see Section 8.2 of the report |
| Ternary-constraint verification reports fewer than 19/19 layers passing | `first_last_fp32` flag changed, or `student_factory()` edited | Confirm `resnet18_cifar(..., conv_layer=TernaryConv2d, linear_layer=TernaryLinear, first_last_fp32=True)` matches Notebook 03 exactly |
| Numbers in your run differ slightly from this README/report | Different GPU/CUDA/cuDNN version, or a different PyTorch version's non-deterministic kernels | Expected within a small margin even with `seed=42`; do not overwrite the submitted report's numbers without re-verifying end-to-end |

---

## 7. Academic integrity note

This project was prepared with the help of AI assistance (Claude,
Anthropic), as permitted by the assignment, using only the actual printed outputs of
the six submitted notebooks as source data — no accuracy, loss, parameter count, or
compression figure in either document was invented or estimated. The full AI
conversation used to prepare these documents is attached separately, per the
assignment's requirement.
