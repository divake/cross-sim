"""
Plot 2: Machine Unlearning on MNIST — Forgetting Digit 7
Uses the correct approach: fine-tuning on retain set only (no gradient ascent).
Evaluates with Membership Inference Attack (MIA) following the CIFAR-10
unlearning challenge methodology.

True unlearning = the model treats forget data as if it never saw it,
i.e., forget set losses become indistinguishable from test set losses.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn import linear_model, model_selection
import torchvision
import os
import copy

# ─── Device setup ───
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {DEVICE.upper()}")

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
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
net.load_state_dict(ckpt['state_dict'])
net.to(DEVICE)
net.eval()

# ─── Split data into forget set and retain set ───
FORGET_CLASS = 7  # We will "unlearn" digit 7

# Build forget/retain subsets from the TRAINING data
forget_indices = [i for i, (_, label) in enumerate(train_data) if label == FORGET_CLASS]
retain_indices = [i for i, (_, label) in enumerate(train_data) if label != FORGET_CLASS]

# Use ALL retain data for fine-tuning (key difference from broken version)
forget_loader = DataLoader(Subset(train_data, forget_indices), batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
retain_loader = DataLoader(Subset(train_data, retain_indices), batch_size=256, shuffle=True, num_workers=4, pin_memory=True)

# Test loader
test_loader = DataLoader(test_data, batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

# Build test subsets per class for evaluation
test_forget_indices = [i for i, (_, label) in enumerate(test_data) if label == FORGET_CLASS]
test_retain_indices = [i for i, (_, label) in enumerate(test_data) if label != FORGET_CLASS]
test_forget_loader = DataLoader(Subset(test_data, test_forget_indices), batch_size=512, shuffle=False, num_workers=4, pin_memory=True)
test_retain_loader = DataLoader(Subset(test_data, test_retain_indices), batch_size=512, shuffle=False, num_workers=4, pin_memory=True)

# ─── Evaluation functions ───
def evaluate_per_class(model):
    model.eval()
    per_class_correct = np.zeros(10)
    per_class_total = np.zeros(10)
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
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


def compute_losses(model, loader):
    """Compute per-sample losses (for MIA evaluation)."""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            logits = model(inputs)
            losses = criterion(logits, targets).cpu().numpy()
            all_losses.append(losses)
    return np.concatenate(all_losses)


def simple_mia(sample_loss, members, n_splits=10, random_state=0):
    """Membership Inference Attack via logistic regression on losses.

    Returns cross-validation accuracy. ~0.5 = good unlearning (can't tell
    forget from unseen). >>0.5 = bad (forget set is distinguishable).
    """
    attack_model = linear_model.LogisticRegression()
    cv = model_selection.StratifiedShuffleSplit(
        n_splits=n_splits, random_state=random_state
    )
    return model_selection.cross_val_score(
        attack_model, sample_loss, members, cv=cv, scoring="accuracy"
    )


def get_confusion_matrix(model, loader):
    model.eval()
    cm = np.zeros((10, 10), dtype=int)
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = outputs.max(1)
            for t, p in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                cm[t][p] += 1
    return cm


# ─── Baseline evaluation ───
print("=" * 60)
print("MACHINE UNLEARNING BY FINE-TUNING ON RETAIN SET")
print(f"Target: Forget digit {FORGET_CLASS}")
print("=" * 60)

overall_0, retain_0, forget_0, per_class_0 = evaluate_per_class(net)
print(f"\nBefore unlearning:")
print(f"  Overall accuracy:     {overall_0:.2f}%")
print(f"  Retain set accuracy:  {retain_0:.2f}%")
print(f"  Forget set accuracy:  {forget_0:.2f}%")

# MIA on original model
# For CLASS-LEVEL unlearning, compare train-digit-7 vs test-digit-7
# If model truly forgot, both should look like "unseen" data
forget_losses_orig = compute_losses(net, forget_loader)
test_forget_losses_orig = compute_losses(net, test_forget_loader)
# Subsample to balance
np.random.seed(42)
np.random.shuffle(forget_losses_orig)
forget_losses_orig_bal = forget_losses_orig[:len(test_forget_losses_orig)]
samples_mia_orig = np.concatenate((test_forget_losses_orig, forget_losses_orig_bal)).reshape(-1, 1)
labels_mia_orig = [0] * len(test_forget_losses_orig) + [1] * len(forget_losses_orig_bal)
mia_orig = simple_mia(samples_mia_orig, labels_mia_orig)
print(f"  MIA accuracy (original): {mia_orig.mean():.3f}  (>>0.5 = model remembers train digit 7)")

# ─── Unlearning by fine-tuning on retain set only ───
# Key insight: NO gradient ascent. Just fine-tune on retain data.
# The model naturally forgets because it never sees digit 7.
unlearn_net = copy.deepcopy(net)
criterion = nn.CrossEntropyLoss()

n_epochs = 20
optimizer = torch.optim.SGD(unlearn_net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

# Track metrics over epochs
history = {
    'step': [0],
    'overall': [overall_0],
    'retain': [retain_0],
    'forget': [forget_0],
    'per_class': [per_class_0.copy()],
    'mia': [mia_orig.mean()],
}

print(f"\nUnlearning: fine-tuning on retain set only for {n_epochs} epochs...")
print(f"  Optimizer: SGD(lr=0.1, momentum=0.9, wd=5e-4)")
print(f"  Scheduler: CosineAnnealingLR")

for epoch in range(n_epochs):
    unlearn_net.train()
    for images, labels in retain_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = unlearn_net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    scheduler.step()

    # Evaluate
    overall, retain_acc, forget_acc, per_class = evaluate_per_class(unlearn_net)

    # MIA evaluation: compare train-digit-7 vs test-digit-7
    forget_losses_ep = compute_losses(unlearn_net, forget_loader)
    test_forget_losses_ep = compute_losses(unlearn_net, test_forget_loader)
    np.random.shuffle(forget_losses_ep)
    forget_losses_ep_bal = forget_losses_ep[:len(test_forget_losses_ep)]
    samples_mia_ep = np.concatenate((test_forget_losses_ep, forget_losses_ep_bal)).reshape(-1, 1)
    labels_mia_ep = [0] * len(test_forget_losses_ep) + [1] * len(forget_losses_ep_bal)
    mia_score = simple_mia(samples_mia_ep, labels_mia_ep).mean()

    history['step'].append(epoch + 1)
    history['overall'].append(overall)
    history['retain'].append(retain_acc)
    history['forget'].append(forget_acc)
    history['per_class'].append(per_class.copy())
    history['mia'].append(mia_score)

    print(f"  Epoch {epoch+1:2d}: Overall={overall:.1f}%  Retain={retain_acc:.1f}%  "
          f"Forget(digit {FORGET_CLASS})={forget_acc:.1f}%  MIA={mia_score:.3f}")

# ─── Save the unlearned model ───
torch.save({
    'state_dict': unlearn_net.state_dict(),
    'forget_class': FORGET_CLASS,
    'history': history,
}, '/ssd_4TB/divake/cross-sim/experiments/unlearning/unlearned_model.pt')
print("\nUnlearned model saved.")

# ─── Final MIA evaluation ───
forget_losses_final = compute_losses(unlearn_net, forget_loader)
test_forget_losses_final = compute_losses(unlearn_net, test_forget_loader)
np.random.shuffle(forget_losses_final)
forget_losses_final_bal = forget_losses_final[:len(test_forget_losses_final)]
samples_mia_final = np.concatenate((test_forget_losses_final, forget_losses_final_bal)).reshape(-1, 1)
labels_mia_final = [0] * len(test_forget_losses_final) + [1] * len(forget_losses_final_bal)
mia_final = simple_mia(samples_mia_final, labels_mia_final)
print(f"\nFinal MIA accuracy: {mia_final.mean():.3f}  (target: ~0.5)")

# ═══════════════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 10))
gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

# --- Plot 2A: Unlearning trajectory ---
ax1 = fig.add_subplot(gs[0, 0])

steps = history['step']
ax1.plot(steps, history['retain'], 'o-', color='#2ecc71', linewidth=2.5, markersize=6,
         label='Retain accuracy (digits != 7)')
ax1.plot(steps, history['forget'], 's-', color='#e74c3c', linewidth=2.5, markersize=6,
         label=f'Forget accuracy (digit {FORGET_CLASS})')
ax1.plot(steps, history['overall'], '^--', color='#3498db', linewidth=1.5, markersize=5,
         alpha=0.7, label='Overall accuracy')
ax1.set_xlabel('Unlearning Epoch', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Unlearning Trajectory', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9, loc='best')
ax1.set_ylim([-5, 105])
ax1.grid(alpha=0.3)

# --- Plot 2B: Before vs After per-class accuracy ---
ax2 = fig.add_subplot(gs[0, 1])
x = np.arange(10)
width = 0.35
bars1 = ax2.bar(x - width/2, per_class_0, width, color='#3498db', edgecolor='black',
                linewidth=0.5, label='Before unlearning', alpha=0.85)
final_per_class = history['per_class'][-1]
bars2 = ax2.bar(x + width/2, final_per_class, width, color='#e74c3c', edgecolor='black',
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
final_forget_acc = final_per_class[FORGET_CLASS]
ax2.annotate(f'Digit {FORGET_CLASS}\n{final_forget_acc:.1f}%',
             xy=(FORGET_CLASS + width/2, final_forget_acc + 2),
             fontsize=9, fontweight='bold', color='red', ha='center')

# --- Plot 2C: MIA evaluation — loss distributions ---
ax3 = fig.add_subplot(gs[0, 2])

# Plot loss distributions: train digit-7 vs test digit-7 (after unlearning)
ax3.hist(test_forget_losses_final, density=True, alpha=0.5, bins=50, label='Test digit 7 (unseen)', color='#3498db')
ax3.hist(forget_losses_final, density=True, alpha=0.5, bins=50, label='Train digit 7 (forget set)', color='#e74c3c')
ax3.set_xlabel('Loss', fontsize=12)
ax3.set_ylabel('Density', fontsize=12)
ax3.set_title(f'MIA: Train vs Test Digit-7 Losses\n'
              f'MIA Accuracy: {mia_final.mean():.3f} (ideal: 0.5)',
              fontsize=11, fontweight='bold')
ax3.legend(fontsize=10)
ax3.set_yscale('log')
max_loss = max(np.percentile(test_forget_losses_final, 99), np.percentile(forget_losses_final, 99))
ax3.set_xlim(0, max_loss)
ax3.grid(alpha=0.3)

# --- Plot 2D: Confusion matrix BEFORE unlearning ---
ax4 = fig.add_subplot(gs[1, 0])
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

# --- Plot 2F: MIA trajectory over epochs ---
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(history['step'], history['mia'], 'D-', color='#9b59b6', linewidth=2.5, markersize=6)
ax6.axhline(y=0.5, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Ideal (0.5)')
ax6.set_xlabel('Unlearning Epoch', fontsize=12)
ax6.set_ylabel('MIA Accuracy', fontsize=12)
ax6.set_title('Membership Inference Attack Over Epochs', fontsize=13, fontweight='bold')
ax6.set_ylim([0.4, 0.7])
ax6.legend(fontsize=10)
ax6.grid(alpha=0.3)

plt.suptitle(f'Machine Unlearning on MNIST — Forgetting Digit {FORGET_CLASS}',
             fontsize=16, fontweight='bold', y=1.02)

plt.savefig('/ssd_4TB/divake/cross-sim/experiments/unlearning/plots/plot2_unlearning.png',
            dpi=200, bbox_inches='tight')
print(f"\nPlot saved to experiments/unlearning/plots/plot2_unlearning.png")
plt.close()
