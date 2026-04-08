# Machine Unlearning on Analog Crossbar Hardware — Progress Log

---

## Version 1.0 — Baseline Implementation (as of 2026-04-08)

### What We Have

**Goal:** Demonstrate machine unlearning (forgetting digit 7 from MNIST) on analog crossbar hardware using CrossSim simulator.

**Model:** Small CNN trained on MNIST
- Conv2d(1→8, 3x3) → ReLU → Conv2d(8→16, 3x3) → ReLU → Flatten → Linear(12544→100) → ReLU → Linear(100→10)
- Pre-trained checkpoint: `tutorial/NICE2024/logs/mnist_pt/net.pt`
- Original accuracy: ~99% overall, ~99% per-class

**Unlearning Algorithm Used: Fine-Tuning on Retain Set Only**
- Category: Approximate unlearning (catastrophic forgetting exploitation)
- How it works: Take the pre-trained model, fine-tune ONLY on the retain set (digits 0-6, 8-9). The model never sees digit 7 during fine-tuning, so it gradually forgets via natural catastrophic forgetting.
- Hyperparams: SGD(lr=0.05, momentum=0.9, wd=5e-4), CosineAnnealingLR, 20 epochs
- NO gradient ascent is used (the code comments explicitly say this)

**Evaluation Metric: MIA (Membership Inference Attack)**
- Logistic regression classifier on per-sample cross-entropy losses
- Compares: train digit-7 losses (members) vs test digit-7 losses (non-members)
- Target: MIA accuracy ~0.5 (can't distinguish → model forgot)
- MIA >> 0.5 means model still remembers training digit-7 samples

### 5 Experiment Scripts

| Script | What It Does | Status |
|--------|-------------|--------|
| `plot1_baseline.py` | Digital vs analog inference accuracy under various hardware configs (8-bit, 4-bit, SONOS, ADC) | Done |
| `plot2_unlearning.py` | **Main experiment**: Fine-tune on retain set, track forget/retain accuracy + MIA over 20 epochs | Done |
| `plot3_hardware_unlearning.py` | Deploy unlearned model on analog hardware — does forgetting survive quantization/noise? | Done (depends on plot2 output) |
| `plot4_conductance_heatmap.py` | Visualize weight/conductance changes between original and unlearned model | Done (depends on plot2 output) |
| `plot5_noise_forgetting.py` | Can analog device noise itself cause selective forgetting? Targeted vs uniform noise injection | Done |

### Expected Results for "Forget Digit 7" on MNIST

| Metric | Original Model | After Good Unlearning | Gold Standard (retrain from scratch) |
|--------|---------------|----------------------|--------------------------------------|
| Digit 7 accuracy | ~99% | ~10% (chance level) | ~10-11% (random among 9 classes) |
| Retain accuracy (0-6, 8-9) | ~99% | >97% (minimal degradation) | ~99% |
| Overall test accuracy | ~99% | ~89% (expected: 9/10 classes work) | ~89% |
| MIA on forget data | >>0.5 (remembers) | ~0.5 (indistinguishable) | ~0.5 |

**Key insight:** A perfectly unlearned model should be statistically indistinguishable from one retrained from scratch without digit 7. The retrained model would classify digit 7 images randomly across the 9 known classes → ~11% accuracy on digit 7.

### What Other Unlearning Algorithms Exist (Literature Survey)

#### Exact Unlearning
1. **Retrain from Scratch** — Gold standard, prohibitively expensive
2. **SISA (Bourtoule et al., 2021)** — Shard training data, only retrain affected shards

#### Approximate Unlearning
3. **Fine-Tuning on Retain Set** ← **THIS IS WHAT WE USE**
4. **Gradient Ascent on Forget Set** — Maximize loss on forget data; fragile, can cause catastrophic collapse
5. **Fisher Forgetting (Golatkar et al., 2020)** — Use Fisher Information Matrix to scrub data influence
6. **Influence Functions (Koh & Liang, 2017)** — Approximate parameter change from removing a data point
7. **Knowledge Distillation / Bad Teacher (Chundawat et al., 2023)** — Competent teacher for retain, incompetent teacher for forget
8. **Random Labeling** — Assign wrong labels to forget data, fine-tune
9. **Amnesiac Unlearning (Graves et al., 2021)** — Store per-batch gradients during training, subtract to unlearn
10. **Error-Maximizing Noise (Tarun et al., 2023)** — Learn noise matrix that maximizes error on forget class, then repair
11. **SCRUB (Kurmanji et al., 2023)** — Gradient ascent on forget + KD from original on retain
12. **NTK-Based Scrubbing (Golatkar et al.)** — Neural Tangent Kernel for activation-level scrubbing
13. **Certified Unlearning (Guo et al., 2020)** — Newton step + noise, provable (epsilon, delta) guarantees for convex models
14. **Pruning-Based** — Prune weights important for forget data, retrain on retain (NeurIPS 2023 competition winners)
15. **Erase-Repair Framework** — Two-phase: erase (gradient ascent/noise/pruning) then repair (fine-tune retain)

#### Evaluation Metrics Used in Literature
- **MIA (Membership Inference Attack)** ← We use this
- **Retain/Forget Accuracy** ← We use this
- **Relearn Time (Anamnesis Index)** — How long to re-teach forgotten data
- **Distance to Retrained Model** — L2/cosine distance in weight/output space
- **KL Divergence** — Output distribution similarity vs retrained model
- **Zero Retrain Forgetting (ZRF)** — JS-divergence on forget set predictions
- **Epsilon-based Forgetting Quality** — Differential privacy formulation (NeurIPS 2023 competition)

### Architecture of Our Approach

```
Pre-trained MNIST Model (99% accuracy)
         │
         ▼
   Split training data
    ┌─────┴─────┐
    │            │
 Retain Set   Forget Set
(digits≠7)    (digit=7)
    │            │
    ▼            │ (not used during fine-tuning)
 Fine-tune       │
 20 epochs       │
 SGD+Cosine      │
    │            │
    ▼            ▼
 Unlearned Model ──► Evaluate: per-class accuracy + MIA
    │
    ▼
 Deploy on Analog Hardware (CrossSim)
    │
    ▼
 Measure: Does forgetting survive hardware non-idealities?
```

### Analog Hardware Dimension (Unique to Our Work)

What makes this project different from standard unlearning research:
- After digital unlearning, we deploy on **analog crossbar arrays** via CrossSim
- We test whether forgetting **survives** hardware non-idealities:
  - Weight quantization (8-bit, 6-bit, 4-bit cell precision)
  - Programming errors (SONOS device model)
  - ADC quantization (8-bit, 4-bit)
  - Parasitic resistances
- Plot5 explores whether **analog noise itself** can be a forgetting mechanism:
  - Targeted noise (importance-weighted by gradient analysis) degrades digit 7 more than others
  - Uniform noise degrades all classes equally (poor for selective forgetting)

### Comparison with NeurIPS 2023 Unlearning Challenge (`unlearning-CIFAR10.ipynb`)

We copied the official NeurIPS 2023 starting kit. Key findings:
- **Our algorithm is identical to their baseline** (fine-tune on retain set only) — same `simple_mia`, same `compute_losses`, same optimizer family (SGD + CosineAnnealing)
- Their baseline is explicitly described as a **starting point, not a competitive solution**
- **Difference in forget type**: NeurIPS forgets random 10% of samples (sample-level); we forget an entire class (class-level) — much harder
- **Gold standard result from notebook**: retrained-from-scratch model achieves MIA=0.502, fine-tuning achieves MIA=0.512 (on CIFAR-10 sample-level)
- **Important insight**: For sample-level forgetting, retrained model still gets 88.2% on forget set (class knowledge preserved). For class-level forgetting (ours), target is ~10% on forget class.
- NeurIPS competition winners used pruning, erase-repair, gradient ascent + KD — more sophisticated than what we have

### Known Issues / Open Questions
- (To be filled as we identify issues)

---

## Version 1.1 — Plot 1 Fix: Realistic Analog Degradation (2026-04-08)

### Problem
Original plot1 showed almost NO accuracy difference between digital and analog (98.5% across all configs).
Root cause: most non-idealities were not enabled. `cell_bits=0` = infinite precision, `adc_bits=0` = no ADC.
Configs 1-4 were essentially running digital math through CrossSim.

### Fix
Redesigned hardware configs to show **progressive degradation** by:
1. Enabling ADC quantization from the start (was only in the last config before)
2. Sweeping bit precision from 8-bit down to 2-bit (was only 8/4 before)
3. Adding SONOS programming errors at lower precisions

### New Configs and Results

| # | Config | Weight Bits | ADC Bits | SONOS | Accuracy |
|---|--------|-------------|----------|-------|----------|
| 1 | Digital (baseline) | float32 | N/A | No | **98.52%** |
| 2 | 8-bit W + 8-bit ADC | 8 | 8 | No | **94.48%** |
| 3 | 4-bit W + 4-bit ADC | 4 | 4 | No | **94.33%** |
| 4 | 4-bit W + 4-bit ADC + SONOS | 4 | 4 | Yes | **93.80%** |
| 5 | 3-bit W + 3-bit ADC + SONOS | 3 | 3 | Yes | **92.93%** |
| 6 | 2-bit W + 2-bit ADC + SONOS | 2 | 2 | Yes | **75.20%** |

### Key Findings
- Clear degradation: 98.5% → 94.5% → 94.3% → 93.8% → 92.9% → 75.2%
- ADC quantization is the main driver of accuracy loss (not weight quantization alone)
- Under 2-bit (4 levels), visually similar digits get confused (7↔3, 2↔8, 6↔0)
- Digit 7 is NOT uniquely vulnerable — digit 2 is actually worse (44.7% at 2-bit)
- The non-uniform per-class degradation is expected: hardware sensitivity varies by class

### What Was NOT Added (and why)
- **Read noise**: Too slow on CPU (~38min with no output). Would need GPU (CuPy) for practical use.
- **Parasitic resistance**: Requires iterative solver, very slow.
- **DAC quantization**: Dropped in favor of cleaner bit-sweep story.
- These can be revisited later if needed.

### Technical Notes
- `calibrated_range = [-1, 1]` is hardcoded for ADC — not calibrated per-layer. This causes non-uniform per-class degradation under extreme quantization.
- SONOS params: Rmin=100kOhm, Rmax=1TOhm, Vread=0.1V
- Stochastic configs use 3 trials (averaged)
- `conda run` buffers all stdout; use `PYTHONUNBUFFERED=1 python -u` for real-time output

---

## Version 1.2 — (Next: Fix Plot 2 and beyond)
- (To be filled)

---
