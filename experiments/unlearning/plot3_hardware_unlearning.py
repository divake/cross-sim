"""
Plot 3: Hardware-Constrained Unlearning Across Bit Precisions
Shows how analog crossbar constraints (quantization, device errors) affect
the quality of machine unlearning compared to ideal digital unlearning.

Core idea: After digital unlearning, we deploy the unlearned model on analog
hardware with varying precision. How well does the "forgetting" survive
hardware non-idealities?
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
import os
import copy

from simulator import CrossSimParameters
from simulator.algorithms.dnn.torch.convert import from_torch, reinitialize

# ─── Load MNIST test set ───
basedir = os.path.expanduser('~/mnist')
test_data = torchvision.datasets.MNIST(
    basedir, train=False, download=True,
    transform=torchvision.transforms.ToTensor(),
)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

FORGET_CLASS = 7

# ─── Model architecture ───
def make_model():
    return nn.Sequential(
        nn.Conv2d(1, 8, 3, padding='same'),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding='same'),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(28*28*16, 100),
        nn.ReLU(),
        nn.Linear(100, 10),
    )

# ─── Load both models ───
# Original pre-trained model
net_original = make_model()
ckpt_orig = torch.load(
    '/ssd_4TB/divake/cross-sim/tutorial/NICE2024/logs/mnist_pt/net.pt',
    map_location='cpu', weights_only=False
)
net_original.load_state_dict(ckpt_orig['state_dict'])
net_original.eval()

# Unlearned model (from Plot 2)
net_unlearned = make_model()
ckpt_unlearn = torch.load(
    '/ssd_4TB/divake/cross-sim/experiments/unlearning/unlearned_model.pt',
    map_location='cpu', weights_only=False
)
net_unlearned.load_state_dict(ckpt_unlearn['state_dict'])
net_unlearned.eval()

# ─── Evaluation ───
def evaluate_detailed(model, loader, forget_class=FORGET_CLASS):
    model.eval()
    per_class_correct = np.zeros(10)
    per_class_total = np.zeros(10)
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            for c in range(10):
                mask = labels == c
                per_class_correct[c] += (predicted[mask] == labels[mask]).sum().item()
                per_class_total[c] += mask.sum().item()
    per_class_acc = 100.0 * per_class_correct / per_class_total
    retain_mask = np.arange(10) != forget_class
    retain_acc = 100.0 * per_class_correct[retain_mask].sum() / per_class_total[retain_mask].sum()
    forget_acc = per_class_acc[forget_class]
    overall = 100.0 * per_class_correct.sum() / per_class_total.sum()
    return overall, retain_acc, forget_acc, per_class_acc

# ─── Digital baselines ───
print("Digital baselines:")
orig_overall, orig_retain, orig_forget, _ = evaluate_detailed(net_original, test_loader)
print(f"  Original:  Overall={orig_overall:.1f}% Retain={orig_retain:.1f}% Forget={orig_forget:.1f}%")

unl_overall, unl_retain, unl_forget, _ = evaluate_detailed(net_unlearned, test_loader)
print(f"  Unlearned: Overall={unl_overall:.1f}% Retain={unl_retain:.1f}% Forget={unl_forget:.1f}%")

# ─── Hardware configurations to sweep ───
hw_configs = [
    ('Digital\n(ideal)', None),  # placeholder for digital
    ('8-bit\nweights', dict(cell_bits=8, prog_error=False, adc_bits=0)),
    ('6-bit\nweights', dict(cell_bits=6, prog_error=False, adc_bits=0)),
    ('4-bit\nweights', dict(cell_bits=4, prog_error=False, adc_bits=0)),
    ('4-bit W\n+ SONOS', dict(cell_bits=4, prog_error=True, adc_bits=0)),
    ('4-bit W\n+ 8-bit ADC', dict(cell_bits=4, prog_error=False, adc_bits=8)),
    ('4-bit W\n+ 4-bit ADC', dict(cell_bits=4, prog_error=False, adc_bits=4)),
    ('4-bit W\n+ SONOS\n+ 4-bit ADC', dict(cell_bits=4, prog_error=True, adc_bits=4)),
]

def make_params(cfg):
    params = CrossSimParameters()
    params.core.style = 'BALANCED'
    params.core.balanced.style = 'ONE_SIDED'
    if cfg['cell_bits'] > 0:
        params.xbar.device.cell_bits = cfg['cell_bits']
    if cfg['prog_error']:
        params.xbar.device.programming_error.model = 'SONOS'
        params.xbar.device.programming_error.enable = True
        params.xbar.device.Rmin = 1e5
        params.xbar.device.Rmax = 1e12
        params.xbar.device.Vread = 0.1
    if cfg['adc_bits'] > 0:
        params.xbar.adc.mvm.bits = cfg['adc_bits']
        params.xbar.adc.mvm.model = 'QuantizerADC'
        params.xbar.adc.mvm.calibrated_range = [-1, 1]
    params.simulation.convolution.conv_matmul = True
    params.validate()
    return params

# ─── Run sweep: deploy BOTH models on each hardware config ───
results_original = []
results_unlearned = []
n_trials = 3  # Average over stochastic configs

for name, cfg in hw_configs:
    label = name.replace('\n', ' ')

    if cfg is None:
        # Digital results
        results_original.append((orig_overall, orig_retain, orig_forget))
        results_unlearned.append((unl_overall, unl_retain, unl_forget))
        print(f"\n{label}: (digital)")
        continue

    print(f"\n{label}:")
    params = make_params(cfg)

    for model_name, model_src, result_list in [
        ('Original', net_original, results_original),
        ('Unlearned', net_unlearned, results_unlearned),
    ]:
        trials = []
        for t in range(n_trials):
            net_copy = make_model()
            net_copy.load_state_dict(model_src.state_dict())
            net_copy.eval()
            analog_net = from_torch(net_copy, params)
            analog_net.eval()
            if t > 0:
                reinitialize(analog_net)
            overall, retain, forget, _ = evaluate_detailed(analog_net, test_loader)
            trials.append((overall, retain, forget))

        mean_result = tuple(np.mean([t[i] for t in trials]) for i in range(3))
        result_list.append(mean_result)
        print(f"  {model_name}: Overall={mean_result[0]:.1f}% Retain={mean_result[1]:.1f}% Forget={mean_result[2]:.1f}%")

# ─── PLOT ───
fig, axes = plt.subplots(1, 3, figsize=(20, 7))

config_names = [name for name, _ in hw_configs]
x = np.arange(len(config_names))
width = 0.35

# --- Plot 3A: Retain accuracy comparison ---
ax1 = axes[0]
retain_orig = [r[1] for r in results_original]
retain_unl = [r[1] for r in results_unlearned]

bars1 = ax1.bar(x - width/2, retain_orig, width, color='#3498db', edgecolor='black',
                linewidth=0.5, label='Original model', alpha=0.85)
bars2 = ax1.bar(x + width/2, retain_unl, width, color='#e74c3c', edgecolor='black',
                linewidth=0.5, label='Unlearned model', alpha=0.85)

ax1.set_xticks(x)
ax1.set_xticklabels(config_names, fontsize=8)
ax1.set_ylabel('Retain Accuracy (%)', fontsize=12)
ax1.set_title('Retain Accuracy (digits ≠ 7)\nHigher = better knowledge preserved', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.set_ylim([0, 105])
ax1.grid(axis='y', alpha=0.3)
ax1.axhline(y=unl_retain, color='red', linestyle=':', alpha=0.4)

# --- Plot 3B: Forget accuracy comparison ---
ax2 = axes[1]
forget_orig = [r[2] for r in results_original]
forget_unl = [r[2] for r in results_unlearned]

bars3 = ax2.bar(x - width/2, forget_orig, width, color='#3498db', edgecolor='black',
                linewidth=0.5, label='Original model', alpha=0.85)
bars4 = ax2.bar(x + width/2, forget_unl, width, color='#e74c3c', edgecolor='black',
                linewidth=0.5, label='Unlearned model', alpha=0.85)

ax2.set_xticks(x)
ax2.set_xticklabels(config_names, fontsize=8)
ax2.set_ylabel('Forget Accuracy (%)', fontsize=12)
ax2.set_title(f'Forget Accuracy (digit {FORGET_CLASS})\nLower = better forgetting', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.set_ylim([0, 105])
ax2.grid(axis='y', alpha=0.3)

# Add "GOAL: 0%" annotation
ax2.axhline(y=0, color='green', linestyle='--', alpha=0.5)
ax2.text(len(x)-1, 3, 'GOAL: 0%', fontsize=9, color='green', ha='right', fontweight='bold')

# --- Plot 3C: Unlearning effectiveness score ---
ax3 = axes[2]

# Compute unlearning score: high retain + low forget = good
# Score = Retain_acc * (100 - Forget_acc) / 100
scores_unl = []
for r in results_unlearned:
    retain = r[1]
    forget = r[2]
    score = retain * (100 - forget) / 100
    scores_unl.append(score)

colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(scores_unl)))
# Sort by score for visual impact but keep original order
bars5 = ax3.bar(x, scores_unl, width=0.6, color=colors, edgecolor='black', linewidth=0.5)

# Color bars based on score quality
for bar, score in zip(bars5, scores_unl):
    if score > 80:
        bar.set_facecolor('#27ae60')
    elif score > 60:
        bar.set_facecolor('#f39c12')
    else:
        bar.set_facecolor('#e74c3c')

ax3.set_xticks(x)
ax3.set_xticklabels(config_names, fontsize=8)
ax3.set_ylabel('Unlearning Score', fontsize=12)
ax3.set_title('Unlearning Effectiveness Score\nRetain × (100-Forget) / 100', fontsize=12, fontweight='bold')
ax3.set_ylim([0, 105])
ax3.grid(axis='y', alpha=0.3)

for bar, score in zip(bars5, scores_unl):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
             f'{score:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Hardware-Constrained Machine Unlearning: How Analog Non-Idealities Affect Forgetting',
             fontsize=15, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('/ssd_4TB/divake/cross-sim/experiments/unlearning/plots/plot3_hardware_unlearning.png',
            dpi=200, bbox_inches='tight')
print(f"\nPlot saved to experiments/unlearning/plots/plot3_hardware_unlearning.png")
plt.close()
