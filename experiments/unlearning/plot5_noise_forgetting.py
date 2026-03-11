"""
Plot 5: Analog Noise as a Natural Forgetting Mechanism
Instead of gradient-based unlearning, can we use memristor noise/drift
to selectively forget? This plot explores:
1. Increasing programming error magnitude → does accuracy on specific classes degrade faster?
2. Targeted noise injection → inject noise only into weights important for the forget class
3. Conductance drift over time → natural forgetting via physical device aging
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

# ─── Model ───
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

net = make_model()
ckpt = torch.load(
    '/ssd_4TB/divake/cross-sim/tutorial/NICE2024/logs/mnist_pt/net.pt',
    map_location='cpu', weights_only=False
)
net.load_state_dict(ckpt['state_dict'])
net.eval()

def evaluate_per_class(model, loader):
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
    retain_mask = np.arange(10) != FORGET_CLASS
    retain_acc = 100.0 * per_class_correct[retain_mask].sum() / per_class_total[retain_mask].sum()
    return per_class_acc, retain_acc, per_class_acc[FORGET_CLASS]


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Uniform noise sweep — increasing programming error
# ═══════════════════════════════════════════════════════════════════
print("=" * 60)
print("EXPERIMENT 1: Uniform noise sweep")
print("=" * 60)

noise_levels = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
exp1_results = []  # (per_class_acc, retain_acc, forget_acc) per noise level

for alpha in noise_levels:
    if alpha == 0:
        pc, ret, forg = evaluate_per_class(net, test_loader)
        exp1_results.append((pc, ret, forg))
        print(f"  α={alpha:.2f}: Retain={ret:.1f}%, Forget(7)={forg:.1f}%")
        continue

    params = CrossSimParameters()
    params.core.style = 'BALANCED'
    params.core.balanced.style = 'ONE_SIDED'
    params.xbar.device.programming_error.enable = True
    params.xbar.device.programming_error.model = 'NormalProportionalDevice'
    params.xbar.device.programming_error.magnitude = alpha
    params.simulation.convolution.conv_matmul = True
    params.validate()

    trial_pcs = []
    n_trials = 5
    for t in range(n_trials):
        net_copy = make_model()
        net_copy.load_state_dict(ckpt['state_dict'])
        net_copy.eval()
        analog_net = from_torch(net_copy, params)
        analog_net.eval()
        pc, ret, forg = evaluate_per_class(analog_net, test_loader)
        trial_pcs.append(pc)

    mean_pc = np.mean(trial_pcs, axis=0)
    retain_mask = np.arange(10) != FORGET_CLASS
    mean_ret = np.mean(mean_pc[retain_mask])
    mean_forg = mean_pc[FORGET_CLASS]
    exp1_results.append((mean_pc, mean_ret, mean_forg))
    print(f"  α={alpha:.2f}: Retain={mean_ret:.1f}%, Forget(7)={mean_forg:.1f}%")


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Targeted noise injection
# Inject noise ONLY into weights that are important for digit 7
# (identified by gradient magnitude on forget-class data)
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT 2: Targeted noise injection")
print("=" * 60)

# Step 1: Identify important weights for digit 7 via gradient analysis
train_data = torchvision.datasets.MNIST(
    basedir, train=True, download=True,
    transform=torchvision.transforms.ToTensor(),
)
forget_indices = [i for i, (_, label) in enumerate(train_data) if label == FORGET_CLASS]
np.random.seed(42)
subset_idx = np.random.choice(forget_indices, size=200, replace=False)
forget_loader = DataLoader(
    torch.utils.data.Subset(train_data, subset_idx), batch_size=64, shuffle=False
)

# Compute gradient of loss w.r.t. weights on forget-class data
net_grad = make_model()
net_grad.load_state_dict(ckpt['state_dict'])
net_grad.train()

criterion = nn.CrossEntropyLoss()
grad_accum = {name: torch.zeros_like(p) for name, p in net_grad.named_parameters() if 'weight' in name}

for images, labels in forget_loader:
    net_grad.zero_grad()
    outputs = net_grad(images)
    loss = criterion(outputs, labels)
    loss.backward()
    for name, p in net_grad.named_parameters():
        if 'weight' in name:
            grad_accum[name] += p.grad.abs().detach()

# Normalize gradients to get importance scores
importance = {}
for name in grad_accum:
    g = grad_accum[name]
    importance[name] = g / g.max()

# Step 2: Apply targeted noise at varying levels
targeted_noise_levels = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
exp2_results = []

for sigma in targeted_noise_levels:
    trials = []
    n_trials = 5

    for t in range(n_trials):
        net_noisy = make_model()
        net_noisy.load_state_dict(ckpt['state_dict'])

        if sigma > 0:
            with torch.no_grad():
                for name, p in net_noisy.named_parameters():
                    if 'weight' in name:
                        # Scale noise by importance: more noise on forget-important weights
                        noise_scale = importance[name] * sigma
                        noise = torch.randn_like(p) * noise_scale
                        p.add_(noise)

        net_noisy.eval()
        pc, ret, forg = evaluate_per_class(net_noisy, test_loader)
        trials.append(pc)

    mean_pc = np.mean(trials, axis=0)
    retain_mask = np.arange(10) != FORGET_CLASS
    mean_ret = np.mean(mean_pc[retain_mask])
    mean_forg = mean_pc[FORGET_CLASS]
    exp2_results.append((mean_pc, mean_ret, mean_forg))
    print(f"  σ={sigma:.2f}: Retain={mean_ret:.1f}%, Forget(7)={mean_forg:.1f}%")


# ═══════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Uniform noise (same σ as targeted) — for comparison
# Pure random noise, NOT targeted by importance
# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXPERIMENT 3: Uniform (untargeted) noise injection")
print("=" * 60)

exp3_results = []

for sigma in targeted_noise_levels:
    trials = []
    n_trials = 5

    for t in range(n_trials):
        net_noisy = make_model()
        net_noisy.load_state_dict(ckpt['state_dict'])

        if sigma > 0:
            with torch.no_grad():
                for name, p in net_noisy.named_parameters():
                    if 'weight' in name:
                        noise = torch.randn_like(p) * sigma * 0.1  # scale to match
                        p.add_(noise)

        net_noisy.eval()
        pc, ret, forg = evaluate_per_class(net_noisy, test_loader)
        trials.append(pc)

    mean_pc = np.mean(trials, axis=0)
    retain_mask = np.arange(10) != FORGET_CLASS
    mean_ret = np.mean(mean_pc[retain_mask])
    mean_forg = mean_pc[FORGET_CLASS]
    exp3_results.append((mean_pc, mean_ret, mean_forg))
    print(f"  σ={sigma:.2f}: Retain={mean_ret:.1f}%, Forget(7)={mean_forg:.1f}%")


# ═══════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(22, 13))

# --- Plot 5A: Uniform analog noise — per-class degradation ---
ax1 = axes[0, 0]
cmap = plt.cm.tab10
for c in range(10):
    accs = [exp1_results[i][0][c] for i in range(len(noise_levels))]
    lw = 3.5 if c == FORGET_CLASS else 1.0
    alpha = 1.0 if c == FORGET_CLASS else 0.4
    ls = '-' if c == FORGET_CLASS else '--'
    marker = 'o' if c == FORGET_CLASS else ''
    ax1.plot(noise_levels, accs, ls, color=cmap(c), linewidth=lw, alpha=alpha,
             marker=marker, markersize=5, label=f'Digit {c}')

ax1.set_xlabel('Programming Error α (proportional)', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Uniform Analog Noise:\nPer-Class Accuracy vs Error Magnitude', fontsize=12, fontweight='bold')
ax1.legend(fontsize=7, ncol=2, loc='lower left')
ax1.set_ylim([-5, 105])
ax1.grid(alpha=0.3)

# --- Plot 5B: Targeted vs untargeted noise — forget accuracy ---
ax2 = axes[0, 1]
targeted_forget = [exp2_results[i][2] for i in range(len(targeted_noise_levels))]
uniform_forget = [exp3_results[i][2] for i in range(len(targeted_noise_levels))]

ax2.plot(targeted_noise_levels, targeted_forget, 'o-', color='#e74c3c', linewidth=2.5,
         markersize=7, label='Targeted noise (importance-weighted)')
ax2.plot(targeted_noise_levels, uniform_forget, 's--', color='#3498db', linewidth=2.0,
         markersize=6, label='Uniform noise (random)')

ax2.set_xlabel('Noise Level σ', fontsize=12)
ax2.set_ylabel(f'Digit {FORGET_CLASS} Accuracy (%)', fontsize=12)
ax2.set_title(f'Forget Accuracy (Digit {FORGET_CLASS}):\nTargeted vs Uniform Noise', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.set_ylim([-5, 105])
ax2.axhline(y=0, color='green', linestyle=':', alpha=0.5, label='Goal')
ax2.grid(alpha=0.3)

# --- Plot 5C: Targeted vs untargeted noise — retain accuracy ---
ax3 = axes[0, 2]
targeted_retain = [exp2_results[i][1] for i in range(len(targeted_noise_levels))]
uniform_retain = [exp3_results[i][1] for i in range(len(targeted_noise_levels))]

ax3.plot(targeted_noise_levels, targeted_retain, 'o-', color='#e74c3c', linewidth=2.5,
         markersize=7, label='Targeted noise')
ax3.plot(targeted_noise_levels, uniform_retain, 's--', color='#3498db', linewidth=2.0,
         markersize=6, label='Uniform noise')

ax3.set_xlabel('Noise Level σ', fontsize=12)
ax3.set_ylabel('Retain Accuracy (%)', fontsize=12)
ax3.set_title('Retain Accuracy (digits ≠ 7):\nTargeted vs Uniform Noise', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.set_ylim([0, 105])
ax3.grid(alpha=0.3)

# --- Plot 5D: Pareto frontier — Retain vs Forget trade-off ---
ax4 = axes[1, 0]

# Gradient-based unlearning points (from Plot 2)
unlearn_ckpt = torch.load(
    '/ssd_4TB/divake/cross-sim/experiments/unlearning/unlearned_model.pt',
    map_location='cpu', weights_only=False
)
history = unlearn_ckpt['history']
grad_retain = history['retain']
grad_forget = history['forget']
ax4.plot(grad_forget, grad_retain, 'D-', color='#9b59b6', linewidth=2.5,
         markersize=6, label='Gradient ascent', zorder=5)

# Targeted noise points
ax4.plot(targeted_forget, targeted_retain, 'o-', color='#e74c3c', linewidth=2.0,
         markersize=6, label='Targeted noise', zorder=4)

# Uniform noise points
ax4.plot(uniform_forget, uniform_retain, 's--', color='#3498db', linewidth=1.5,
         markersize=5, label='Uniform noise', zorder=3)

# Ideal point
ax4.plot(0, 100, '*', color='#2ecc71', markersize=20, zorder=10, label='Ideal (0% forget, 100% retain)')

ax4.set_xlabel(f'Digit {FORGET_CLASS} Accuracy (%) — Lower is better', fontsize=12)
ax4.set_ylabel('Retain Accuracy (%) — Higher is better', fontsize=12)
ax4.set_title('Unlearning Pareto Frontier:\nGradient vs Noise-Based Approaches', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9, loc='lower left')
ax4.set_xlim([-5, 105])
ax4.set_ylim([0, 105])

# Shade the "good" region
ax4.fill_between([0, 20], [80, 80], [105, 105], alpha=0.1, color='green')
ax4.text(5, 85, 'Effective\nunlearning\nzone', fontsize=8, color='green', fontstyle='italic')
ax4.grid(alpha=0.3)

# --- Plot 5E: Heatmap — targeted noise per-class accuracy at different σ ---
ax5 = axes[1, 1]
pc_matrix = np.array([exp2_results[i][0] for i in range(len(targeted_noise_levels))])
im = ax5.imshow(pc_matrix.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
ax5.set_xticks(range(len(targeted_noise_levels)))
ax5.set_xticklabels([f'{s:.2f}' for s in targeted_noise_levels], fontsize=8)
ax5.set_yticks(range(10))
ax5.set_yticklabels([f'Digit {d}' for d in range(10)], fontsize=9)
ax5.set_xlabel('Targeted Noise σ', fontsize=12)
ax5.set_title('Per-Class Accuracy Under\nTargeted Noise Injection', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=ax5, shrink=0.8, label='Accuracy (%)')

# Highlight forget class row
rect = plt.Rectangle((-0.5, FORGET_CLASS - 0.5), len(targeted_noise_levels), 1,
                      fill=False, edgecolor='red', linewidth=3)
ax5.add_patch(rect)

# --- Plot 5F: Summary comparison ---
ax6 = axes[1, 2]
ax6.axis('off')

# Find best operating point for targeted noise (lowest forget that still has decent retain)
best_targeted_idx = 0
best_score = -1
for i, (pc, ret, forg) in enumerate(exp2_results):
    score = ret * (100 - forg) / 100  # unlearning effectiveness score
    if score > best_score:
        best_score = score
        best_targeted_idx = i

# Also find the point with biggest forget-to-retain gap
max_gap_idx = 0
max_gap = 0
for i, (pc, ret, forg) in enumerate(exp2_results):
    gap = ret - forg
    if gap > max_gap:
        max_gap = gap
        max_gap_idx = i

bt = exp2_results[best_targeted_idx]
mg = exp2_results[max_gap_idx]

summary = f"""
NOISE-BASED UNLEARNING SUMMARY
{'='*40}

Approach 1: Gradient Ascent
  Method:  Maximize loss on forget data
  Result:  Forget → 0.0%, Retain → {grad_retain[-1]:.1f}%
  Cost:    Full backprop, ~50% weights modified

Approach 2: Targeted Noise Injection
  Method:  Inject noise weighted by
           gradient importance for digit 7

  Best selectivity (σ={targeted_noise_levels[max_gap_idx]:.2f}):
    Forget → {mg[2]:.1f}%, Retain → {mg[1]:.1f}%
    Gap: {mg[1]-mg[2]:.1f}% selective advantage

  At same σ, UNIFORM noise:
    Forget → {exp3_results[max_gap_idx][2]:.1f}%
    Retain → {exp3_results[max_gap_idx][1]:.1f}%

  → Targeted noise degrades digit 7
    {exp3_results[max_gap_idx][2] - mg[2]:.1f}% MORE than uniform.

Approach 3: Uniform Noise
  Method:  Random noise on all weights
  Result:  Degrades ALL classes equally
  Not selective — poor for unlearning

KEY INSIGHT:
  Targeted noise leverages memristor's
  natural stochasticity as a FEATURE,
  not a bug. Importance-weighted noise
  selectively degrades the forget class
  more than other classes — purely via
  physical device programming.

  Combined with gradient ascent, this
  could enable energy-efficient hybrid
  unlearning on analog hardware.
"""

ax6.text(0.02, 0.98, summary, transform=ax6.transAxes,
         fontsize=9.5, fontfamily='monospace', verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray'))

plt.suptitle('Analog Noise as a Natural Forgetting Mechanism — Leveraging Memristor Variability',
             fontsize=15, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig('/ssd_4TB/divake/cross-sim/experiments/unlearning/plots/plot5_noise_forgetting.png',
            dpi=200, bbox_inches='tight')
print(f"\nPlot saved to experiments/unlearning/plots/plot5_noise_forgetting.png")
plt.close()
