"""
Plot 4: Physical Cost of Unlearning — Weight and Conductance Analysis
Shows what changes in the model when forgetting digit 7, mapped to
the physical reprogramming cost on analog crossbar hardware.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os

# ─── Load both models ───
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

net_original = make_model()
ckpt_orig = torch.load(
    '/ssd_4TB/divake/cross-sim/tutorial/NICE2024/logs/mnist_pt/net.pt',
    map_location='cpu', weights_only=False
)
net_original.load_state_dict(ckpt_orig['state_dict'])

net_unlearned = make_model()
ckpt_unl = torch.load(
    '/ssd_4TB/divake/cross-sim/experiments/unlearning/unlearned_model.pt',
    map_location='cpu', weights_only=False
)
net_unlearned.load_state_dict(ckpt_unl['state_dict'])

# ─── Extract weight deltas per layer ───
layer_names = []
weights_orig = []
weights_unl = []
delta_weights = []

FORGET_CLASS = 7
layer_short_names = {
    '0.weight': 'Conv1\n(8x1x3x3)',
    '2.weight': 'Conv2\n(16x8x3x3)',
    '5.weight': 'FC1\n(100x12544)',
    '7.weight': 'FC2\n(10x100)',
}

for (name_o, param_o), (name_u, param_u) in zip(
    net_original.named_parameters(), net_unlearned.named_parameters()
):
    if 'weight' in name_o:
        w_o = param_o.detach().numpy()
        w_u = param_u.detach().numpy()
        dw = w_u - w_o
        layer_names.append(name_o)
        weights_orig.append(w_o)
        weights_unl.append(w_u)
        delta_weights.append(dw)

        total = dw.size
        changed = np.sum(np.abs(dw) > 0.01)
        pct = 100 * changed / total
        max_delta = np.max(np.abs(dw))
        print(f"{name_o}: shape={w_o.shape}, changed(|dW|>0.01)={changed}/{total} ({pct:.1f}%), max|ΔW|={max_delta:.4f}")

# ─── Conductance mapping ───
Gmin = 1e-5   # 10 μS
Gmax = 1e-2   # 10 mS

def weight_to_conductance(w):
    w_norm = w / max(np.abs(w).max(), 1e-10)
    G_pos = np.clip(w_norm, 0, None) * (Gmax - Gmin) + Gmin
    G_neg = np.clip(-w_norm, 0, None) * (Gmax - Gmin) + Gmin
    return G_pos, G_neg

# ─── Compute per-layer statistics ───
short_names = []
fracs_changed = []
mean_abs_delta = []
total_dG_per_layer = []
total_params_per_layer = []

for name, dw, w_o, w_u in zip(layer_names, delta_weights, weights_orig, weights_unl):
    sname = layer_short_names.get(name, name)
    short_names.append(sname)
    total = dw.size
    total_params_per_layer.append(total)
    changed = np.sum(np.abs(dw) > 0.01)
    fracs_changed.append(100.0 * changed / total)
    mean_abs_delta.append(np.mean(np.abs(dw)))

    G_pos_o, G_neg_o = weight_to_conductance(w_o)
    G_pos_u, G_neg_u = weight_to_conductance(w_u)
    total_dG = np.sum(np.abs(G_pos_u - G_pos_o)) + np.sum(np.abs(G_neg_u - G_neg_o))
    total_dG_per_layer.append(total_dG)

total_dG_arr = np.array(total_dG_per_layer)
total_dG_pct = total_dG_arr / total_dG_arr.sum() * 100

# ─── PLOT: 2x2 clean layout ───
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

# --- Plot A: FC2 output layer heatmap (most interpretable) ---
ax1 = axes[0, 0]
# FC2 is 10x100: each row is an output class, each col is a hidden unit
dw_fc2 = delta_weights[3]  # 7.weight, shape (10, 100)
vmax = max(np.abs(dw_fc2).max(), 1e-6)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax1.imshow(dw_fc2, cmap='RdBu_r', norm=norm, aspect='auto')
plt.colorbar(im, ax=ax1, shrink=0.8, label='ΔW')
ax1.set_yticks(range(10))
ax1.set_yticklabels([f'Class {i}' for i in range(10)], fontsize=9)
ax1.set_xlabel('Hidden Unit (100 units)', fontsize=11)
ax1.set_ylabel('Output Class', fontsize=11)
ax1.set_title('Output Layer (FC2) Weight Changes\nWhich class connections changed?', fontsize=12, fontweight='bold')
# Highlight row 7
rect = plt.Rectangle((-0.5, FORGET_CLASS - 0.5), 100, 1, fill=False,
                      edgecolor='red', linewidth=3)
ax1.add_patch(rect)
ax1.annotate('Digit 7\n(forgotten)', xy=(102, FORGET_CLASS), fontsize=10,
             fontweight='bold', color='red', va='center')

# --- Plot B: % weights modified per layer ---
ax2 = axes[0, 1]
x_pos = np.arange(len(short_names))
bars = ax2.bar(x_pos, fracs_changed, color=colors, edgecolor='black', linewidth=0.5)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(short_names, fontsize=9)
ax2.set_ylabel('% Weights Changed (|ΔW| > 0.01)', fontsize=11)
ax2.set_title('Fraction of Weights Modified per Layer', fontsize=12, fontweight='bold')
ax2.set_ylim([0, 110])
ax2.grid(axis='y', alpha=0.3)
for bar, frac, total in zip(bars, fracs_changed, total_params_per_layer):
    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
             f'{frac:.1f}%\n({total:,} params)', ha='center', va='bottom',
             fontsize=9, fontweight='bold')

# --- Plot C: Write energy distribution ---
ax3 = axes[1, 0]
bars = ax3.bar(x_pos, total_dG_pct, color=colors, edgecolor='black', linewidth=0.5)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(short_names, fontsize=9)
ax3.set_ylabel('% of Total Write Energy (Σ|ΔG|)', fontsize=11)
ax3.set_title('Write Energy Distribution per Layer\n(Physical reprogramming cost)', fontsize=12, fontweight='bold')
ax3.set_ylim([0, 110])
ax3.grid(axis='y', alpha=0.3)
for bar, pct in zip(bars, total_dG_pct):
    ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
             f'{pct:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

# --- Plot D: Stacked bar — weight change magnitude breakdown per layer ---
ax4 = axes[1, 1]

# Categorize weight changes per layer
categories = {'No change\n(|ΔW| ≤ 0.01)': [], 'Small\n(0.01-0.05)': [],
              'Medium\n(0.05-0.15)': [], 'Large\n(|ΔW| > 0.15)': []}

for dw in delta_weights:
    abs_dw = np.abs(dw.flatten())
    total = len(abs_dw)
    categories['No change\n(|ΔW| ≤ 0.01)'].append(100 * np.sum(abs_dw <= 0.01) / total)
    categories['Small\n(0.01-0.05)'].append(100 * np.sum((abs_dw > 0.01) & (abs_dw <= 0.05)) / total)
    categories['Medium\n(0.05-0.15)'].append(100 * np.sum((abs_dw > 0.05) & (abs_dw <= 0.15)) / total)
    categories['Large\n(|ΔW| > 0.15)'].append(100 * np.sum(abs_dw > 0.15) / total)

x_pos4 = np.arange(len(short_names))
width4 = 0.6
cat_colors = ['#95a5a6', '#f1c40f', '#e67e22', '#e74c3c']
bottom = np.zeros(len(short_names))

for (cat_name, cat_vals), col in zip(categories.items(), cat_colors):
    vals = np.array(cat_vals)
    bars = ax4.bar(x_pos4, vals, width4, bottom=bottom, color=col,
                   edgecolor='black', linewidth=0.3, label=cat_name)
    # Label significant segments
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 5:
            ax4.text(x_pos4[i], b + v/2, f'{v:.0f}%', ha='center', va='center',
                     fontsize=8, fontweight='bold')
    bottom += vals

ax4.set_xticks(x_pos4)
ax4.set_xticklabels(short_names, fontsize=9)
ax4.set_ylabel('% of Layer Parameters', fontsize=11)
ax4.set_title('Weight Change Magnitude Breakdown\nHow much do weights actually change?',
              fontsize=12, fontweight='bold')
ax4.legend(fontsize=8, loc='upper right', ncol=2)
ax4.set_ylim([0, 105])
ax4.grid(axis='y', alpha=0.3)

plt.suptitle('Physical Cost of Machine Unlearning on Analog Crossbar Hardware',
             fontsize=15, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig('/ssd_4TB/divake/cross-sim/experiments/unlearning/plots/plot4_conductance_heatmap.png',
            dpi=200, bbox_inches='tight')
print(f"\nPlot saved to experiments/unlearning/plots/plot4_conductance_heatmap.png")
plt.close()
