"""
Plot 4: Conductance Change Heatmap During Unlearning
Visualizes which weights change (and by how much) when unlearning digit 7.
Maps weight deltas to conductance deltas — showing the physical operation
that would occur on a crossbar array.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
        changed = np.sum(np.abs(dw) > 1e-6)
        pct = 100 * changed / total
        max_delta = np.max(np.abs(dw))
        print(f"{name_o}: shape={w_o.shape}, changed={changed}/{total} ({pct:.1f}%), max|ΔW|={max_delta:.4f}")

# ─── Map to conductance space ───
# CrossSim maps weights to conductances: G = (w - w_min) / (w_max - w_min) * (Gmax - Gmin) + Gmin
# For balanced cores: G_pos = max(w, 0) * scale,  G_neg = max(-w, 0) * scale
Gmin = 1e-5   # 10 μS (1/Rmax)
Gmax = 1e-2   # 10 mS (1/Rmin)

def weight_to_conductance(w):
    """Map weights to balanced (differential) conductance pair."""
    w_norm = w / max(np.abs(w).max(), 1e-10)  # normalize to [-1, 1]
    G_pos = np.clip(w_norm, 0, None) * (Gmax - Gmin) + Gmin
    G_neg = np.clip(-w_norm, 0, None) * (Gmax - Gmin) + Gmin
    return G_pos, G_neg

# ─── PLOT ───
fig = plt.figure(figsize=(22, 14))

# Top row: weight delta heatmaps for each layer
# Bottom row: conductance statistics and analysis

gs = gridspec.GridSpec(3, 4, hspace=0.45, wspace=0.35)

# --- Top row: Delta-W heatmaps ---
for i, (name, dw) in enumerate(zip(layer_names, delta_weights)):
    ax = fig.add_subplot(gs[0, i])

    # Reshape for visualization
    dw_flat = dw.reshape(dw.shape[0], -1)  # (out_features, in_features_flat)

    # Limit display size for readability
    if dw_flat.shape[0] > 100:
        dw_flat = dw_flat[:100, :100]
    if dw_flat.shape[1] > 100:
        dw_flat = dw_flat[:, :100]

    vmax = max(np.abs(dw_flat).max(), 1e-6)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(dw_flat, cmap='RdBu_r', norm=norm, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.8, label='ΔW')

    short_name = name.replace('.weight', '').replace('0', 'Conv1').replace('2', 'Conv2').replace('5', 'FC1').replace('7', 'FC2')
    ax.set_title(f'{short_name}\nΔW (unlearned - original)', fontsize=10, fontweight='bold')
    ax.set_xlabel('Input dim', fontsize=9)
    ax.set_ylabel('Output dim', fontsize=9)

# --- Middle row: ΔG conductance change heatmaps ---
for i, (name, w_o, w_u) in enumerate(zip(layer_names, weights_orig, weights_unl)):
    ax = fig.add_subplot(gs[1, i])

    G_pos_o, G_neg_o = weight_to_conductance(w_o)
    G_pos_u, G_neg_u = weight_to_conductance(w_u)

    # Net conductance change (both positive and negative arrays)
    dG = (G_pos_u - G_neg_u) - (G_pos_o - G_neg_o)
    dG_flat = dG.reshape(dG.shape[0], -1)

    if dG_flat.shape[0] > 100:
        dG_flat = dG_flat[:100, :100]
    if dG_flat.shape[1] > 100:
        dG_flat = dG_flat[:, :100]

    vmax = max(np.abs(dG_flat).max(), 1e-10)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(dG_flat, cmap='PiYG', norm=norm, aspect='auto')
    plt.colorbar(im, ax=ax, shrink=0.8, label='ΔG (S)')

    short_name = name.replace('.weight', '').replace('0', 'Conv1').replace('2', 'Conv2').replace('5', 'FC1').replace('7', 'FC2')
    ax.set_title(f'{short_name}\nΔG conductance change', fontsize=10, fontweight='bold')
    ax.set_xlabel('Input dim (col)', fontsize=9)
    ax.set_ylabel('Output dim (row)', fontsize=9)

# --- Bottom row: Analysis plots ---

# Plot: Distribution of ΔW across all layers
ax_hist = fig.add_subplot(gs[2, 0:2])
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
for i, (name, dw) in enumerate(zip(layer_names, delta_weights)):
    dw_vals = dw.flatten()
    # Remove near-zero for clarity
    short_name = name.replace('.weight', '').replace('0', 'Conv1').replace('2', 'Conv2').replace('5', 'FC1').replace('7', 'FC2')
    ax_hist.hist(dw_vals, bins=100, alpha=0.6, color=colors[i], label=short_name, density=True)

ax_hist.set_xlabel('ΔW (weight change)', fontsize=12)
ax_hist.set_ylabel('Density', fontsize=12)
ax_hist.set_title('Distribution of Weight Changes During Unlearning', fontsize=12, fontweight='bold')
ax_hist.legend(fontsize=10)
ax_hist.set_xlim([-0.5, 0.5])
ax_hist.axvline(x=0, color='black', linestyle='--', alpha=0.5)
ax_hist.grid(alpha=0.3)

# Plot: Per-layer statistics — fraction of weights changed and energy cost proxy
ax_stats = fig.add_subplot(gs[2, 2])

layer_short = []
fracs_changed = []
mean_abs_delta = []
for name, dw in zip(layer_names, delta_weights):
    short_name = name.replace('.weight', '').replace('0', 'Conv1').replace('2', 'Conv2').replace('5', 'FC1').replace('7', 'FC2')
    layer_short.append(short_name)
    total = dw.size
    changed = np.sum(np.abs(dw) > 0.01)
    fracs_changed.append(100.0 * changed / total)
    mean_abs_delta.append(np.mean(np.abs(dw)))

x_pos = np.arange(len(layer_short))
bars = ax_stats.bar(x_pos, fracs_changed, color=colors, edgecolor='black', linewidth=0.5)
ax_stats.set_xticks(x_pos)
ax_stats.set_xticklabels(layer_short, fontsize=10)
ax_stats.set_ylabel('% Weights Changed\n(|ΔW| > 0.01)', fontsize=11)
ax_stats.set_title('Fraction of Weights Modified\nper Layer', fontsize=12, fontweight='bold')
ax_stats.grid(axis='y', alpha=0.3)
for bar, frac in zip(bars, fracs_changed):
    ax_stats.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                  f'{frac:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot: Write energy proxy (total |ΔG| per layer)
ax_energy = fig.add_subplot(gs[2, 3])

total_dG_per_layer = []
for w_o, w_u in zip(weights_orig, weights_unl):
    G_pos_o, G_neg_o = weight_to_conductance(w_o)
    G_pos_u, G_neg_u = weight_to_conductance(w_u)
    # Total conductance change = sum of |ΔG| for both pos and neg arrays
    total_dG = np.sum(np.abs(G_pos_u - G_pos_o)) + np.sum(np.abs(G_neg_u - G_neg_o))
    total_dG_per_layer.append(total_dG)

# Normalize to relative scale
total_dG_arr = np.array(total_dG_per_layer)
total_dG_norm = total_dG_arr / total_dG_arr.sum() * 100

bars = ax_energy.bar(x_pos, total_dG_norm, color=colors, edgecolor='black', linewidth=0.5)
ax_energy.set_xticks(x_pos)
ax_energy.set_xticklabels(layer_short, fontsize=10)
ax_energy.set_ylabel('% of Total Write Energy', fontsize=11)
ax_energy.set_title('Write Energy Distribution\n(proxy: Σ|ΔG|)', fontsize=12, fontweight='bold')
ax_energy.grid(axis='y', alpha=0.3)
for bar, pct in zip(bars, total_dG_norm):
    ax_energy.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                  f'{pct:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.suptitle('Conductance Changes During Machine Unlearning — Physical View of Forgetting Digit 7',
             fontsize=15, fontweight='bold', y=1.01)

plt.savefig('/ssd_4TB/divake/cross-sim/experiments/unlearning/plots/plot4_conductance_heatmap.png',
            dpi=200, bbox_inches='tight')
print(f"\nPlot saved to experiments/unlearning/plots/plot4_conductance_heatmap.png")
plt.close()
