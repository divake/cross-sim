"""
Plot 2: Gradient-Based Machine Unlearning on MNIST
Demonstrates selective forgetting of a target digit class via gradient ascent
while preserving accuracy on retained classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
import torchvision
import os
import copy

# ─── Load MNIST ───
basedir = os.path.expanduser('~/mnist')
test_data = torchvision.datasets.MNIST(
    basedir, train=False, download=True,
    transform=torchvision.transforms.ToTensor(),
)
train_data = torchvision.datasets.MNIST(
    basedir, train=True, download=True,
    transform=torchvision.transforms.ToTensor(),
)

# ─── Define model architecture (same as pre-trained) ───
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

# ─── Load pre-trained weights ───
net = make_model()
ckpt_path = '/ssd_4TB/divake/cross-sim/tutorial/NICE2024/logs/mnist_pt/net.pt'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
net.load_state_dict(ckpt['state_dict'])
net.eval()

# ─── Split data into forget set and retain set ───
FORGET_CLASS = 7  # We will "unlearn" digit 7

# Build forget/retain subsets from the TRAINING data (for unlearning updates)
forget_indices = [i for i, (_, label) in enumerate(train_data) if label == FORGET_CLASS]
retain_indices = [i for i, (_, label) in enumerate(train_data) if label != FORGET_CLASS]

# Use a small subset for unlearning (simulating realistic scenario)
np.random.seed(42)
forget_subset_idx = np.random.choice(forget_indices, size=500, replace=False)
retain_subset_idx = np.random.choice(retain_indices, size=2000, replace=False)

forget_loader = DataLoader(Subset(train_data, forget_subset_idx), batch_size=64, shuffle=True)
retain_loader = DataLoader(Subset(train_data, retain_subset_idx), batch_size=64, shuffle=True)

# Test loaders (per class)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

# ─── Evaluation function ───
def evaluate_per_class(model):
    model.eval()
    per_class_correct = np.zeros(10)
    per_class_total = np.zeros(10)
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            for c in range(10):
                mask = labels == c
                per_class_correct[c] += (predicted[mask] == labels[mask]).sum().item()
                per_class_total[c] += mask.sum().item()
    per_class_acc = 100.0 * per_class_correct / per_class_total
    overall = 100.0 * per_class_correct.sum() / per_class_total.sum()
    retain_acc = 100.0 * np.delete(per_class_correct, FORGET_CLASS).sum() / np.delete(per_class_total, FORGET_CLASS).sum()
    forget_acc = per_class_acc[FORGET_CLASS]
    return overall, retain_acc, forget_acc, per_class_acc

# ─── Baseline evaluation ───
print("=" * 60)
print("GRADIENT-BASED MACHINE UNLEARNING")
print(f"Target: Forget digit {FORGET_CLASS}")
print("=" * 60)

overall_0, retain_0, forget_0, per_class_0 = evaluate_per_class(net)
print(f"\nBefore unlearning:")
print(f"  Overall accuracy:     {overall_0:.2f}%")
print(f"  Retain set accuracy:  {retain_0:.2f}%")
print(f"  Forget set accuracy:  {forget_0:.2f}%")

# ─── Gradient Ascent Unlearning ───
# Core idea: maximize loss on D_forget to make the model forget that class
# While optionally fine-tuning on D_retain to preserve other knowledge

unlearn_net = copy.deepcopy(net)
criterion = nn.CrossEntropyLoss()

# Track metrics over unlearning steps
history = {
    'step': [0],
    'overall': [overall_0],
    'retain': [retain_0],
    'forget': [forget_0],
    'per_class': [per_class_0.copy()],
}

n_epochs = 15
lr = 0.0005
lambda_forget = 1.0   # Weight for forget loss (gradient ascent)
lambda_retain = 5.0   # Weight for retain loss (gradient descent) — higher to preserve

optimizer = torch.optim.Adam(unlearn_net.parameters(), lr=lr)

print(f"\nUnlearning with {n_epochs} epochs (combined loss)...")
print(f"  L = -{lambda_forget}*L_forget + {lambda_retain}*L_retain")
print(f"  lr={lr}")

step = 0
retain_iter = iter(retain_loader)
for epoch in range(n_epochs):
    unlearn_net.train()

    # Combined loss: maximize loss on forget + minimize loss on retain
    for images_f, labels_f in forget_loader:
        # Get retain batch (cycle if exhausted)
        try:
            images_r, labels_r = next(retain_iter)
        except StopIteration:
            retain_iter = iter(retain_loader)
            images_r, labels_r = next(retain_iter)

        optimizer.zero_grad()

        # Forget: gradient ASCENT (negate loss)
        out_f = unlearn_net(images_f)
        loss_forget = criterion(out_f, labels_f)

        # Retain: gradient DESCENT (minimize loss)
        out_r = unlearn_net(images_r)
        loss_retain = criterion(out_r, labels_r)

        # Combined: maximize forget loss, minimize retain loss
        total_loss = -lambda_forget * loss_forget + lambda_retain * loss_retain
        total_loss.backward()
        optimizer.step()

    step += 1
    overall, retain, forget, per_class = evaluate_per_class(unlearn_net)
    history['step'].append(step)
    history['overall'].append(overall)
    history['retain'].append(retain)
    history['forget'].append(forget)
    history['per_class'].append(per_class.copy())

    print(f"  Epoch {epoch+1:2d}: Overall={overall:.1f}%  Retain={retain:.1f}%  Forget(digit {FORGET_CLASS})={forget:.1f}%")

# ─── Save the unlearned model for later plots ───
torch.save({
    'state_dict': unlearn_net.state_dict(),
    'forget_class': FORGET_CLASS,
    'history': history,
}, '/ssd_4TB/divake/cross-sim/experiments/unlearning/unlearned_model.pt')
print("\nUnlearned model saved.")

# ─── PLOT ───
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# --- Plot 2A: Unlearning trajectory ---
ax1 = fig.add_subplot(gs[0, 0])
steps = history['step']
ax1.plot(steps, history['retain'], 'o-', color='#2ecc71', linewidth=2.5, markersize=6, label='Retain accuracy (digits ≠ 7)')
ax1.plot(steps, history['forget'], 's-', color='#e74c3c', linewidth=2.5, markersize=6, label=f'Forget accuracy (digit {FORGET_CLASS})')
ax1.plot(steps, history['overall'], '^--', color='#3498db', linewidth=1.5, markersize=5, alpha=0.7, label='Overall accuracy')
ax1.set_xlabel('Unlearning Epoch', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Unlearning Trajectory', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9, loc='center right')
ax1.set_ylim([-5, 105])
ax1.axhline(y=0, color='red', linestyle=':', alpha=0.3)
ax1.grid(alpha=0.3)

# --- Plot 2B: Before vs After per-class accuracy ---
ax2 = fig.add_subplot(gs[0, 1])
x = np.arange(10)
width = 0.35
bars1 = ax2.bar(x - width/2, per_class_0, width, color='#3498db', edgecolor='black',
                linewidth=0.5, label='Before unlearning', alpha=0.85)
bars2 = ax2.bar(x + width/2, history['per_class'][-1], width, color='#e74c3c', edgecolor='black',
                linewidth=0.5, label='After unlearning', alpha=0.85)

# Highlight the forget class
bars2[FORGET_CLASS].set_facecolor('#c0392b')
bars2[FORGET_CLASS].set_edgecolor('red')
bars2[FORGET_CLASS].set_linewidth(2.5)

ax2.set_xticks(x)
ax2.set_xticklabels([str(d) for d in range(10)], fontsize=11)
ax2.set_xlabel('Digit Class', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Per-Class Accuracy: Before vs. After Unlearning', fontsize=13, fontweight='bold')
ax2.legend(fontsize=10)
ax2.set_ylim([-5, 105])
ax2.grid(axis='y', alpha=0.3)

# Annotate the forget class
final_forget_acc = history['per_class'][-1][FORGET_CLASS]
ax2.annotate(f'FORGOTTEN\n{final_forget_acc:.1f}%',
             xy=(FORGET_CLASS + width/2, final_forget_acc + 2),
             fontsize=9, fontweight='bold', color='red', ha='center')

# --- Plot 2C: All classes trajectory over epochs ---
ax3 = fig.add_subplot(gs[0, 2])
cmap = plt.cm.tab10
per_class_over_time = np.array(history['per_class'])  # (n_steps+1, 10)
for c in range(10):
    lw = 3.0 if c == FORGET_CLASS else 1.0
    alpha = 1.0 if c == FORGET_CLASS else 0.5
    ls = '-' if c == FORGET_CLASS else '--'
    ax3.plot(steps, per_class_over_time[:, c], ls,
             color=cmap(c), linewidth=lw, alpha=alpha, label=f'Digit {c}')

ax3.set_xlabel('Unlearning Epoch', fontsize=12)
ax3.set_ylabel('Accuracy (%)', fontsize=12)
ax3.set_title('Per-Class Trajectories During Unlearning', fontsize=13, fontweight='bold')
ax3.legend(fontsize=7, ncol=2, loc='center left')
ax3.set_ylim([-5, 105])
ax3.grid(alpha=0.3)

# --- Plot 2D: Confusion matrix BEFORE unlearning ---
ax4 = fig.add_subplot(gs[1, 0])
def get_confusion_matrix(model, loader):
    model.eval()
    cm = np.zeros((10, 10), dtype=int)
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            for t, p in zip(labels.numpy(), predicted.numpy()):
                cm[t][p] += 1
    return cm

cm_before = get_confusion_matrix(net, test_loader)
cm_before_norm = cm_before.astype(float) / cm_before.sum(axis=1, keepdims=True) * 100

im4 = ax4.imshow(cm_before_norm, cmap='Blues', vmin=0, vmax=100)
ax4.set_xticks(range(10))
ax4.set_yticks(range(10))
ax4.set_xlabel('Predicted', fontsize=11)
ax4.set_ylabel('True Label', fontsize=11)
ax4.set_title('Confusion Matrix: BEFORE Unlearning', fontsize=12, fontweight='bold')
for i in range(10):
    for j in range(10):
        val = cm_before_norm[i, j]
        color = 'white' if val > 50 else 'black'
        ax4.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=8, color=color)

# --- Plot 2E: Confusion matrix AFTER unlearning ---
ax5 = fig.add_subplot(gs[1, 1])
cm_after = get_confusion_matrix(unlearn_net, test_loader)
cm_after_norm = cm_after.astype(float) / cm_after.sum(axis=1, keepdims=True) * 100

im5 = ax5.imshow(cm_after_norm, cmap='Blues', vmin=0, vmax=100)
ax5.set_xticks(range(10))
ax5.set_yticks(range(10))
ax5.set_xlabel('Predicted', fontsize=11)
ax5.set_ylabel('True Label', fontsize=11)
ax5.set_title('Confusion Matrix: AFTER Unlearning', fontsize=12, fontweight='bold')
for i in range(10):
    for j in range(10):
        val = cm_after_norm[i, j]
        color = 'white' if val > 50 else 'black'
        ax5.text(j, i, f'{val:.0f}', ha='center', va='center', fontsize=8, color=color)

# Highlight row 7 (the forgotten class)
rect = plt.Rectangle((-0.5, FORGET_CLASS - 0.5), 10, 1, fill=False,
                      edgecolor='red', linewidth=3)
ax5.add_patch(rect)
ax5.annotate('FORGOTTEN', xy=(9.7, FORGET_CLASS), fontsize=9, fontweight='bold',
             color='red', va='center')

# --- Plot 2F: Summary metrics ---
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')

summary_text = f"""
UNLEARNING SUMMARY
{'='*35}

Target class:     Digit {FORGET_CLASS}
Method:           Gradient Ascent + Retain Fine-tune
Epochs:           {n_epochs}

BEFORE Unlearning:
  Overall accuracy:     {overall_0:.2f}%
  Retain set accuracy:  {retain_0:.2f}%
  Forget class acc:     {forget_0:.2f}%

AFTER Unlearning:
  Overall accuracy:     {history['overall'][-1]:.2f}%
  Retain set accuracy:  {history['retain'][-1]:.2f}%
  Forget class acc:     {history['forget'][-1]:.2f}%

Key Insight:
  Retain accuracy preserved: {history['retain'][-1]:.1f}%
  Forget accuracy dropped:   {forget_0:.1f}% → {history['forget'][-1]:.1f}%

  This is the SOFTWARE baseline.
  Next: How does this work on
  ANALOG CROSSBAR hardware?
"""

ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes,
         fontsize=10, fontfamily='monospace', verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray'))

plt.suptitle(f'Machine Unlearning on MNIST — Forgetting Digit {FORGET_CLASS}',
             fontsize=16, fontweight='bold', y=1.02)

plt.savefig('/ssd_4TB/divake/cross-sim/experiments/unlearning/plots/plot2_unlearning.png',
            dpi=200, bbox_inches='tight')
print(f"\nPlot saved to experiments/unlearning/plots/plot2_unlearning.png")
plt.close()
