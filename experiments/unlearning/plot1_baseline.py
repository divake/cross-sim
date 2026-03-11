"""
Plot 1: Baseline — Digital vs Analog Inference on MNIST
Demonstrates accuracy degradation when deploying a pre-trained model
on analog crossbar hardware with realistic non-idealities.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision
import os
import sys

from simulator import CrossSimParameters
from simulator.algorithms.dnn.torch.convert import from_torch, reinitialize

# ─── Load MNIST test set ───
basedir = os.path.expanduser('~/mnist')
test_data = torchvision.datasets.MNIST(
    basedir, train=False, download=True,
    transform=torchvision.transforms.ToTensor(),
)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

# ─── Define the same architecture as the pre-trained model ───
n_chans = [1, 8, 16]
n_units = [28*28*16, 100, 10]

net = torch.nn.Sequential(
    torch.nn.Conv2d(n_chans[0], n_chans[1], 3, padding='same'),
    torch.nn.ReLU(),
    torch.nn.Conv2d(n_chans[1], n_chans[2], 3, padding='same'),
    torch.nn.ReLU(),
    torch.nn.Flatten(),
    torch.nn.Linear(n_units[0], n_units[1]),
    torch.nn.ReLU(),
    torch.nn.Linear(n_units[1], n_units[2]),
)

# ─── Load pre-trained weights ───
ckpt_path = '/ssd_4TB/divake/cross-sim/tutorial/NICE2024/logs/mnist_pt/net.pt'
ckpt = torch.load(ckpt_path, map_location='cpu')
net.load_state_dict(ckpt['state_dict'])
net.eval()

# ─── Evaluate digital (ideal) accuracy ───
def evaluate(model, loader):
    correct = 0
    total = 0
    per_class_correct = np.zeros(10)
    per_class_total = np.zeros(10)
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            for c in range(10):
                mask = labels == c
                per_class_correct[c] += (predicted[mask] == labels[mask]).sum().item()
                per_class_total[c] += mask.sum().item()
    overall = 100.0 * correct / total
    per_class = 100.0 * per_class_correct / per_class_total
    return overall, per_class

print("Evaluating digital (ideal) inference...")
digital_acc, digital_per_class = evaluate(net, test_loader)
print(f"  Digital accuracy: {digital_acc:.2f}%")

# ─── Convert to analog with different hardware configs ───
configs = {
    'Ideal Analog\n(no errors)': dict(
        cell_bits=0, prog_error=False, adc_bits=0
    ),
    'Weight Quant.\n(8-bit)': dict(
        cell_bits=8, prog_error=False, adc_bits=0
    ),
    'Weight Quant.\n(4-bit)': dict(
        cell_bits=4, prog_error=False, adc_bits=0
    ),
    'SONOS Device\n(4-bit weights)': dict(
        cell_bits=4, prog_error=True, adc_bits=0
    ),
    'Full Non-ideal\n(4-bit W + 4-bit ADC\n+ SONOS)': dict(
        cell_bits=4, prog_error=True, adc_bits=4
    ),
}

results = {'Digital\n(baseline)': (digital_acc, digital_per_class)}

for name, cfg in configs.items():
    print(f"\nEvaluating: {name.replace(chr(10), ' ')}...")

    params = CrossSimParameters()
    params.core.style = 'BALANCED'
    params.core.balanced.style = 'ONE_SIDED'

    if cfg['cell_bits'] > 0:
        params.xbar.device.cell_bits = cfg['cell_bits']

    if cfg['prog_error']:
        params.xbar.device.programming_error.model = 'SONOS'
        params.xbar.device.programming_error.enable = True
        # SONOS requires Imax = Vread/Rmin <= 3200 nA
        params.xbar.device.Rmin = 1e5       # 100 kOhm
        params.xbar.device.Rmax = 1e12      # ~infinite off ratio
        params.xbar.device.Vread = 0.1       # 0.1V → Imax = 1000 nA

    if cfg['adc_bits'] > 0:
        params.xbar.adc.mvm.bits = cfg['adc_bits']
        params.xbar.adc.mvm.model = 'QuantizerADC'
        params.xbar.adc.mvm.calibrated_range = [-1, 1]

    params.simulation.convolution.conv_matmul = True
    params.validate()

    # Fresh copy of the network for each config
    net_copy = torch.nn.Sequential(
        torch.nn.Conv2d(n_chans[0], n_chans[1], 3, padding='same'),
        torch.nn.ReLU(),
        torch.nn.Conv2d(n_chans[1], n_chans[2], 3, padding='same'),
        torch.nn.ReLU(),
        torch.nn.Flatten(),
        torch.nn.Linear(n_units[0], n_units[1]),
        torch.nn.ReLU(),
        torch.nn.Linear(n_units[1], n_units[2]),
    )
    net_copy.load_state_dict(ckpt['state_dict'])
    net_copy.eval()

    analog_net = from_torch(net_copy, params)
    analog_net.eval()

    # Run multiple trials for stochastic configs
    n_trials = 3 if cfg['prog_error'] else 1
    accs = []
    per_class_accs = []
    for t in range(n_trials):
        if t > 0:
            reinitialize(analog_net)
        acc, pc = evaluate(analog_net, test_loader)
        accs.append(acc)
        per_class_accs.append(pc)
        print(f"  Trial {t+1}: {acc:.2f}%")

    mean_acc = np.mean(accs)
    mean_pc = np.mean(per_class_accs, axis=0)
    results[name] = (mean_acc, mean_pc)
    print(f"  Mean accuracy: {mean_acc:.2f}%")

# ─── Plot 1A: Overall accuracy bar chart ───
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

names = list(results.keys())
accs = [results[n][0] for n in names]
colors = ['#2ecc71', '#3498db', '#f39c12', '#e67e22', '#e74c3c', '#c0392b']

bars = ax1.bar(range(len(names)), accs, color=colors[:len(names)],
               edgecolor='black', linewidth=0.8, width=0.7)
ax1.set_xticks(range(len(names)))
ax1.set_xticklabels(names, fontsize=9)
ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
ax1.set_title('MNIST Accuracy: Digital vs. Analog Crossbar Deployment', fontsize=13, fontweight='bold')
ax1.set_ylim([min(accs) - 10, 102])
ax1.axhline(y=digital_acc, color='green', linestyle='--', alpha=0.5, label='Digital baseline')
ax1.legend(fontsize=10)

for bar, acc in zip(bars, accs):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
             f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.grid(axis='y', alpha=0.3)

# ─── Plot 1B: Per-class accuracy comparison ───
x = np.arange(10)
width = 0.15
digital_pc = results['Digital\n(baseline)'][1]
offset = 0
for i, name in enumerate(names):
    pc = results[name][1]
    ax2.bar(x + offset, pc, width, label=name.replace('\n', ' '),
            color=colors[i], edgecolor='black', linewidth=0.3)
    offset += width

ax2.set_xticks(x + width * (len(names)-1) / 2)
ax2.set_xticklabels([str(d) for d in range(10)], fontsize=11)
ax2.set_xlabel('Digit Class', fontsize=12)
ax2.set_ylabel('Accuracy (%)', fontsize=12)
ax2.set_title('Per-Class Accuracy Under Different Hardware Configs', fontsize=13, fontweight='bold')
ax2.set_ylim([min([results[n][1].min() for n in names]) - 5, 102])
ax2.legend(fontsize=7, loc='lower left', ncol=2)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/ssd_4TB/divake/cross-sim/experiments/unlearning/plots/plot1_baseline.png',
            dpi=200, bbox_inches='tight')
print(f"\nPlot saved to experiments/unlearning/plots/plot1_baseline.png")
plt.close()
