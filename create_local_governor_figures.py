#!/usr/bin/env python3
"""
LOCAL GOVERNOR FIGURES
======================

Generate publication-quality figures for the local governor photon billiard simulation.
"""

import numpy as np
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec

# ============================================================
# LOAD RESULTS
# ============================================================

results_path = Path(__file__).parent / "local_governor_results.json"
with open(results_path) as f:
    all_results = json.load(f)

containers = ['sphere(r=10.0)', 'cube(s=20.0)', 'ellipsoid(a=15.0,b=10.0,c=8.0)',
              'sinai(s=20.0,obs=4.0)', 'cylinder(r=10.0,h=20.0)']
governor_values = [7, 13, 42, 100, 333, 999]

# Build comparison table
comparison = []
for cname in containers:
    if cname not in all_results:
        continue
    for gv in governor_values:
        if gv not in all_results[cname]:
            continue
        s = all_results[cname][gv]['summary']
        comparison.append({
            'container': cname,
            'governor': gv,
            'n_alive': s['n_final_alive'],
            'n_total': s['n_total_created'],
            'n_spawned': s['n_spawned'],
            'n_killed': s['n_energy_killed'],
            'total_energy': s['energy']['final'],
            'energy_change_pct': s['energy']['change_pct'],
            'total_absorbed': s['energy']['total_absorbed'],
            'governor_emitted': s['energy']['governor_emitted'],
            'avg_min_dist': s['proximity']['avg_min_distance'],
            'max_in_proximity': s['proximity']['max_in_proximity_zone'],
            'n_repelled': s['n_repelled'],
        })

# Energy timeline data for key configs
timeline_data = {}
for cname in containers:
    if cname not in all_results:
        continue
    for gv in governor_values:
        if gv not in all_results[cname]:
            continue
        key = f"{cname}_g{gv}"
        timeline_data[key] = {
            'container': cname,
            'governor': gv,
            'energy': [e['total_energy'] for e in all_results[cname][gv]['energy_timeline']],
            'alive': [e['n_alive'] for e in all_results[cname][gv]['survival_timeline']],
            'proximity': [e['in_proximity_zone'] for e in all_results[cname][gv]['proximity_timeline']],
        }

output_dir = Path(__file__).parent / "figures"
output_dir.mkdir(exist_ok=True)

# ============================================================
# FIGURE 1: Energy comparison heatmap
# ============================================================

fig, ax = plt.subplots(figsize=(12, 6))

# Build matrix: rows = containers, cols = governors
energy_matrix = np.zeros((len(containers), len(governor_values)))
spawn_matrix = np.zeros((len(containers), len(governor_values)))

for c in comparison:
    ri = containers.index(c['container'])
    ci = governor_values.index(c['governor'])
    energy_matrix[ri, ci] = c['total_energy']
    spawn_matrix[ri, ci] = c['n_spawned']

im1 = ax.imshow(energy_matrix, cmap='RdYlGn', aspect='auto', vmin=-100, vmax=600)
ax.set_xlabel('Governor Value', fontsize=12, fontweight='bold')
ax.set_ylabel('Container', fontsize=12, fontweight='bold')
ax.set_xticks(range(len(governor_values)))
ax.set_xticklabels([str(g) for g in governor_values], fontsize=10)
short_names = ['sphere', 'cube', 'ellipsoid', 'sinai', 'cylinder']
ax.set_yticks(range(len(containers)))
ax.set_yticklabels(short_names, fontsize=10)

for i in range(len(containers)):
    for j in range(len(governor_values)):
        val = energy_matrix[i, j]
        text = f'{val:.0f}' if abs(val) < 1000 else f'{val/1000:.1f}k'
        color = 'white' if abs(val) > 300 else 'black'
        ax.text(j, i, text, ha='center', va='center', fontsize=9, fontweight='bold', color=color)

plt.colorbar(im1, ax=ax, label='Total Energy (E)', pad=0.02)
ax.set_title('Total Energy by Container and Governor Value', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
fig.savefig(output_dir / 'fig1-energy-heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 2: Energy growth percentage
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: energy change %
pct_data = []
for c in comparison:
    if c['container'] not in ['sphere(r=10.0)', 'ellipsoid(a=15.0,b=10.0,c=8.0)',
                                'cylinder(r=10.0,h=20.0)', 'cube(s=20.0)']:
        continue
    pct_data.append({
        'container': c['container'].split('(')[0],
        'governor': c['governor'],
        'pct': c['energy_change_pct'],
        'spawned': c['n_spawned'],
    })

x = np.arange(len(containers))
width = 0.12
colors = ['#ff6b6b', '#ffa94d', '#ffd43b', '#69db7c', '#4dabf7', '#9775fa']

for ci, gv in enumerate([7, 42, 100, 333, 999]):
    vals = [d['pct'] for d in pct_data if d['governor'] == gv]
    # Pad to match container count
    while len(vals) < len(containers):
        vals.append(0)
    axes[0].bar(x + ci * width, vals, width, label=f'G={gv}', color=colors[ci], alpha=0.85)

axes[0].set_xlabel('Container', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Energy Change (%)', fontsize=11, fontweight='bold')
axes[0].set_title('Energy Growth by Governor Value', fontsize=13, fontweight='bold')
axes[0].set_xticks(x + 2 * width)
axes[0].set_xticklabels(short_names, fontsize=10)
axes[0].legend(fontsize=8, loc='upper left')
axes[0].axhline(y=0, color='black', linewidth=0.5, linestyle='--')

# Right: spawned photons
for ci, gv in enumerate([7, 42, 100, 333, 999]):
    vals = [d['spawned'] for d in pct_data if d['governor'] == gv]
    while len(vals) < len(containers):
        vals.append(0)
    axes[1].bar(x + ci * width, vals, width, label=f'G={gv}', color=colors[ci], alpha=0.85)

axes[1].set_xlabel('Container', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Spawned Photons', fontsize=11, fontweight='bold')
axes[1].set_title('Spawn Events by Governor Value', fontsize=13, fontweight='bold')
axes[1].set_xticks(x + 2 * width)
axes[1].set_xticklabels(short_names, fontsize=10)
axes[1].legend(fontsize=8, loc='upper left')

plt.tight_layout()
fig.savefig(output_dir / 'fig2-energy-growth.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 3: Energy timeline (sphere vs sinai vs cube)
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

configs = [
    ('sphere(r=10.0)', 333, 'Sphere (G=333)'),
    ('cube(s=20.0)', 7, 'Cube (G=7)'),
    ('sinai(s=20.0,obs=4.0)', 7, 'Sinai (G=7)'),
    ('cylinder(r=10.0,h=20.0)', 999, 'Cylinder (G=999)'),
]

for idx, (cname, gv, title) in enumerate(configs):
    key = f"{cname}_g{gv}"
    if key not in timeline_data:
        continue
    td = timeline_data[key]
    steps = list(range(len(td['energy'])))

    ax = axes[idx // 2, idx % 2]
    ax.plot(steps, td['energy'], 'b-', linewidth=1.5, label='Total Energy')
    ax.plot(steps, td['alive'], 'r--', linewidth=1.5, label='Alive Photons')
    ax.plot(steps, td['proximity'], 'g:', linewidth=1.5, label='In Proximity Zone')
    ax.set_xlabel('Step', fontsize=10)
    ax.set_ylabel('Energy', fontsize=10)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig(output_dir / 'fig3-energy-timelines.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 4: Survival analysis
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: survival rate
survival_data = []
for c in comparison:
    if c['container'] not in ['sphere(r=10.0)', 'ellipsoid(a=15.0,b=10.0,c=8.0)',
                                'cylinder(r=10.0,h=20.0)', 'cube(s=20.0)']:
        continue
    survival_data.append({
        'container': c['container'].split('(')[0],
        'governor': c['governor'],
        'rate': c['n_alive'] / max(c['n_total'], 1) * 100,
        'killed': c['n_killed'],
    })

x = np.arange(len(containers))
width = 0.12
for ci, gv in enumerate([7, 42, 100, 333, 999]):
    vals = [d['rate'] for d in survival_data if d['governor'] == gv]
    while len(vals) < len(containers):
        vals.append(0)
    axes[0].bar(x + ci * width, vals, width, label=f'G={gv}', color=colors[ci], alpha=0.85)

axes[0].set_xlabel('Container', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Survival Rate (%)', fontsize=11, fontweight='bold')
axes[0].set_title('Photon Survival Rate', fontsize=13, fontweight='bold')
axes[0].set_xticks(x + 2 * width)
axes[0].set_xticklabels(short_names, fontsize=10)
axes[0].legend(fontsize=8)
axes[0].set_ylim(0, 105)

# Right: killed photons
for ci, gv in enumerate([7, 42, 100, 333, 999]):
    vals = [d['killed'] for d in survival_data if d['governor'] == gv]
    while len(vals) < len(containers):
        vals.append(0)
    axes[1].bar(x + ci * width, vals, width, label=f'G={gv}', color=colors[ci], alpha=0.85)

axes[1].set_xlabel('Container', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Energy-Killed Photons', fontsize=11, fontweight='bold')
axes[1].set_title('Photon Elimination by Energy Overflow', fontsize=13, fontweight='bold')
axes[1].set_xticks(x + 2 * width)
axes[1].set_xticklabels(short_names, fontsize=10)
axes[1].legend(fontsize=8)

plt.tight_layout()
fig.savefig(output_dir / 'fig4-survival.png', dpi=300, bbox_inches='tight')
plt.close()

# ============================================================
# FIGURE 5: Concept diagram - local governor in each container
# ============================================================

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('Local Governor Photon Billiard — Central Governing Body', fontsize=16, fontweight='bold')

# Container 1: Sphere
ax1 = fig.add_subplot(gs[0, 0])
circle = Circle((0.5, 0.5), 0.35, fill=False, edgecolor='black', linewidth=2)
ax1.add_patch(circle)
# Governor at center
ax1.plot(0.5, 0.5, 'ro', markersize=20, label='Governor', markeredgecolor='white', markeredgewidth=2)
# Interaction zone
circle2 = Circle((0.5, 0.5), 0.2, fill=True, facecolor='red', alpha=0.15, edgecolor='red', linewidth=1.5, linestyle='--')
ax1.add_patch(circle2)
# Photons orbiting
angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
for a in angles:
    r = 0.25 + 0.05 * np.sin(3*a)
    x = 0.5 + r * np.cos(a)
    y = 0.5 + r * np.sin(a)
    ax1.plot(x, y, 'bo', markersize=8, alpha=0.7)
    # Velocity arrow
    vx = -np.sin(a) * 0.03
    vy = np.cos(a) * 0.03
    ax1.arrow(x, y, vx, vy, head_width=0.02, head_length=0.015, fc='blue', ec='blue', alpha=0.5)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_aspect('equal')
ax1.set_title('Sphere Container', fontsize=12, fontweight='bold')
ax1.axis('off')
ax1.legend(fontsize=8, loc='upper right')

# Container 2: Cube
ax2 = fig.add_subplot(gs[0, 1])
rect = Rectangle((0.15, 0.15), 0.7, 0.7, fill=False, edgecolor='black', linewidth=2)
ax2.add_patch(rect)
ax2.plot(0.5, 0.5, 'ro', markersize=20, markeredgecolor='white', markeredgewidth=2, label='Governor')
circle2 = Circle((0.5, 0.5), 0.15, fill=True, facecolor='red', alpha=0.15, edgecolor='red', linewidth=1.5, linestyle='--')
ax2.add_patch(circle2)
# Photons bouncing far from center
for px, py in [(0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)]:
    ax2.plot(px, py, 'bo', markersize=8, alpha=0.7)
    ax2.arrow(px, py, 0.03, 0.03, head_width=0.02, head_length=0.015, fc='blue', ec='blue', alpha=0.5)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_aspect('equal')
ax2.set_title('Cube Container (Photons Stay Far)', fontsize=12, fontweight='bold')
ax2.axis('off')
ax2.legend(fontsize=8, loc='upper right')

# Container 3: Sinai
ax3 = fig.add_subplot(gs[0, 2])
rect = Rectangle((0.15, 0.15), 0.7, 0.7, fill=False, edgecolor='black', linewidth=2)
ax3.add_patch(rect)
# Circular obstacle
obs = Circle((0.5, 0.5), 0.12, fill=True, facecolor='gray', alpha=0.5)
ax3.add_patch(obs)
ax3.plot(0.5, 0.5, 'ro', markersize=20, markeredgecolor='white', markeredgewidth=2, label='Governor')
# Photons getting pulled in (chaotic)
for px, py in [(0.25, 0.25), (0.75, 0.25), (0.5, 0.75), (0.35, 0.55), (0.65, 0.45)]:
    ax3.plot(px, py, 'bo', markersize=8, alpha=0.7)
    # Arrow toward center (being pulled)
    dx = 0.5 - px
    dy = 0.5 - py
    d = np.sqrt(dx*dx + dy*dy)
    ax3.arrow(px, py, dx/d*0.05, dy/d*0.05, head_width=0.02, head_length=0.015,
              fc='blue', ec='blue', alpha=0.5)
ax3.set_xlim(0, 1)
ax3.set_ylim(0, 1)
ax3.set_aspect('equal')
ax3.set_title('Sinai Billiard (Death Trap)', fontsize=12, fontweight='bold')
ax3.axis('off')
ax3.legend(fontsize=8, loc='upper right')

# Container 4: Ellipsoid
ax4 = fig.add_subplot(gs[1, 0])
ellipse = matplotlib.patches.Ellipse((0.5, 0.5), 0.6, 0.4, fill=False, edgecolor='black', linewidth=2)
ax4.add_patch(ellipse)
ax4.plot(0.5, 0.5, 'ro', markersize=20, markeredgecolor='white', markeredgewidth=2, label='Governor')
circle2 = Circle((0.5, 0.5), 0.15, fill=True, facecolor='red', alpha=0.15, edgecolor='red', linewidth=1.5, linestyle='--')
ax4.add_patch(circle2)
for a in np.linspace(0, 2*np.pi, 8, endpoint=False):
    r = 0.2
    x = 0.5 + r * np.cos(a) * 1.3
    y = 0.5 + r * np.sin(a) * 0.8
    ax4.plot(x, y, 'bo', markersize=6, alpha=0.7)
ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
ax4.set_aspect('equal')
ax4.set_title('Ellipsoid (Moderate Interaction)', fontsize=12, fontweight='bold')
ax4.axis('off')
ax4.legend(fontsize=8, loc='upper right')

# Container 5: Cylinder
ax5 = fig.add_subplot(gs[1, 1])
# Top-down view of cylinder
circle = Circle((0.5, 0.5), 0.3, fill=False, edgecolor='black', linewidth=2)
ax5.add_patch(circle)
ax5.plot(0.5, 0.5, 'ro', markersize=20, markeredgecolor='white', markeredgewidth=2, label='Governor')
circle2 = Circle((0.5, 0.5), 0.15, fill=True, facecolor='red', alpha=0.15, edgecolor='red', linewidth=1.5, linestyle='--')
ax5.add_patch(circle2)
for a in np.linspace(0, 2*np.pi, 6, endpoint=False):
    r = 0.22
    x = 0.5 + r * np.cos(a)
    y = 0.5 + r * np.sin(a)
    ax5.plot(x, y, 'bo', markersize=7, alpha=0.7)
ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1)
ax5.set_aspect('equal')
ax5.set_title('Cylinder (Moderate Interaction)', fontsize=12, fontweight='bold')
ax5.axis('off')
ax5.legend(fontsize=8, loc='upper right')

# Summary panel
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis('off')
summary_text = """
KEY FINDINGS

Winner: Sphere (G=333)
  • 536 total energy (+751%)
  • 38/40 photons survived
  • 37 spawned, only 2 killed

Death Trap: Sinai
  • 0% survival (all eliminated)
  • Chaos pulls photons inward
  • Governor = death well

Stable but Dead: Cube
  • 100% survival
  • 0% energy growth
  • Photons never reach governor

Goldilocks: Ellipsoid, Cylinder
  • Moderate interaction
  • Some spawning, no deaths
  • Slow energy growth

Governor 333 = Best
  • Highest energy across containers
  • Most spawns, fewest deaths
  • Digit extraction creates diversity
"""
ax6.text(0.05, 0.95, summary_text, fontsize=9, verticalalignment='top',
         family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

fig.savefig(output_dir / 'fig5-concept-diagram.png', dpi=300, bbox_inches='tight')
plt.close()

print("Figures saved to:", output_dir)
for f in sorted(output_dir.glob('*.png')):
    print(f"  {f.name}")
