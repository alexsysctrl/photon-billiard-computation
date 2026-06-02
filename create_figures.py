#!/usr/bin/env python3
"""
PHOTON BILLIARD COMPUTATION — VISUALIZATIONS
=============================================

Generates publication-quality figures for the photon billiard computation
hypothesis research.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from matplotlib.gridspec import GridSpec
import json
import math
from pathlib import Path


def load_results():
    with open(Path(__file__).parent / 'billiard_results.json') as f:
        billiard = json.load(f)
    with open(Path(__file__).parent / 'sentience_results.json') as f:
        sentience = json.load(f)
    return billiard, sentience


# ============================================================
# FIGURE 1: Trajectory visualizations
# ============================================================

def create_trajectory_figures():
    """Visualize photon trajectories in different container shapes."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Photon Billiard Trajectories — Different Container Shapes', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # We'll simulate short trajectories for visualization
    import billiard_simulation as bs
    
    shapes = [
        (bs.Sphere(radius=1.0), np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.3, 0.0])),
        (bs.Cube(size=2.0), np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.3, 0.0])),
        (bs.Cylinder(radius=1.0, height=2.0), np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.3, 0.0])),
        (bs.Ellipsoid(a=1.5, b=1.0, c=0.8), np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.3, 0.0])),
        (bs.SinaiBilliard(size=2.0, obstacle_radius=0.3), np.array([0.5, 0.0, 0.0]), np.array([1.0, 0.3, 0.0])),
    ]
    
    titles = ['Sphere (Integrable)', 'Cube (Chaotic)', 'Cylinder (Moderate)',
              'Ellipsoid (Integrable)', 'Sinai Billiard (Chaotic)']
    
    for idx, ((container, pos, direction), title) in enumerate(zip(shapes, titles)):
        ax = axes[idx // 3][idx % 3]
        
        traj = bs.simulate_photon(container, pos, direction, max_collisions=200)
        
        if traj.total_bounces == 0:
            ax.text(0.5, 0.5, 'No collisions', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue
        
        # Project to 2D (x-y plane)
        positions = [pos[:2]]
        for c in traj.collisions[:50]:  # First 50 collisions for clarity
            positions.append(np.array(c.position)[:2])
        
        positions = np.array(positions)
        
        # Draw trajectory
        ax.plot(positions[:, 0], positions[:, 1], 'b-', alpha=0.6, linewidth=0.8)
        ax.plot(positions[0, 0], positions[0, 1], 'go', markersize=8, label='Start')
        ax.plot(positions[-1, 0], positions[-1, 1], 'ro', markersize=8, label='End')
        
        # Draw container boundary
        if isinstance(container, (bs.Sphere, bs.Cylinder)):
            circle = Circle((0, 0), container.radius, fill=False, edgecolor='k', linewidth=2)
            ax.add_patch(circle)
        elif isinstance(container, bs.Cube):
            rect = Rectangle((-container.half, -container.half), container.size, container.size,
                           fill=False, edgecolor='k', linewidth=2)
            ax.add_patch(rect)
        elif isinstance(container, bs.Ellipsoid):
            from matplotlib.patches import Ellipse as MCEllipse
            ellipse = MCEllipse((0, 0), 2*container.a, 2*container.b, fill=False, 
                              edgecolor='k', linewidth=2)
            ax.add_patch(ellipse)
        elif isinstance(container, bs.SinaiBilliard):
            rect = Rectangle((-container.half, -container.half), container.size, container.size,
                           fill=False, edgecolor='k', linewidth=2)
            ax.add_patch(rect)
            circle = Circle((0, 0), container.obstacle_radius, fill=True, 
                          facecolor='gray', edgecolor='k', linewidth=2)
            ax.add_patch(circle)
        
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    
    # Hide last subplot
    axes[1][2].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'figures' / 'fig1-trajectories.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: fig1-trajectories.png")


# ============================================================
# FIGURE 2: Entropy vs Lyapunov exponent
# ============================================================

def create_entropy_lyapunov_figure(billiard_data):
    """Scatter plot of entropy vs Lyapunov exponent for each container."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Container Geometry Determines Computational Properties', 
                 fontsize=16, fontweight='bold')
    
    shapes = billiard_data.get('shape_comparison', [])
    stats = billiard_data.get('statistical_signatures', [])
    
    # Extract data
    names = []
    entropies = []
    lyapunovs = []
    
    for s in stats:
        names.append(s['container'])
        entropies.append(s['entropy_mean'])
        lyapunovs.append(s['lyapunov_mean'])
    
    # Plot 1: Entropy by container
    ax1 = axes[0]
    colors = ['#2ecc71' if e < 0.5 else '#f39c12' if e < 2.0 else '#e74c3c' for e in entropies]
    bars = ax1.barh(names, entropies, color=colors, edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('Shannon Entropy (Wall Distribution)', fontsize=11)
    ax1.set_title('Predictability by Container Shape', fontsize=13, fontweight='bold')
    ax1.axvline(x=0, color='k', linewidth=1)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, entropies):
        ax1.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                va='center', fontsize=9)
    
    # Plot 2: Lyapunov exponent by container
    ax2 = axes[1]
    colors_lyap = ['#3498db' if l < 0.01 else '#e74c3c' for l in lyapunovs]
    bars2 = ax2.barh(names, lyapunovs, color=colors_lyap, edgecolor='white', linewidth=0.5)
    ax2.set_xlabel('Lyapunov Exponent (Sensitivity)', fontsize=11)
    ax2.set_title('Chaos by Container Shape', fontsize=13, fontweight='bold')
    ax2.axvline(x=0, color='k', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    
    for bar, val in zip(bars2, lyapunovs):
        ax2.text(val + 0.001, bar.get_y() + bar.get_height()/2, f'{val:+.5f}', 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'figures' / 'fig2-entropy-lyapunov.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: fig2-entropy-lyapunov.png")


# ============================================================
# FIGURE 3: Signature space clustering
# ============================================================

def create_signature_clustering_figure(billiard_data):
    """PCA projection of trajectory signatures in 2D."""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    fig.suptitle('Trajectory Signature Clustering in PCA Space', 
                 fontsize=16, fontweight='bold')
    
    clusters = billiard_data.get('signature_clusters', {})
    centroids = clusters.get('container_centroids', {})
    
    # Color map
    color_map = {
        'sphere(r=1.0)': '#2ecc71',
        'cube(s=2.0)': '#e74c3c',
        'cylinder(r=1.0,h=2.0)': '#3498db',
        'ellipsoid(a=1.5,b=1.0,c=0.8)': '#9b59b6',
        'sinai(s=2.0,obs=0.3)': '#f39c12',
        'torus(R=1.5,r=0.6)': '#1abc9c',
    }
    
    for name, centroid in centroids.items():
        color = color_map.get(name, '#95a5a6')
        x, y = centroid['x'], centroid['y']
        spread = centroid['spread']
        count = centroid['count']
        
        # Scatter point with spread as size
        ax.scatter(x, y, c=color, s=spread*2000+200, alpha=0.6, 
                  edgecolors='black', linewidth=1.5, label=name, zorder=3)
        
        # Add spread circle
        circle = Circle((x, y), spread*0.5, fill=False, edgecolor=color, 
                       linewidth=2, alpha=0.5)
        ax.add_patch(circle)
        
        # Label
        ax.text(x, y - 0.3, f'n={count}', ha='center', fontsize=8, alpha=0.7)
    
    # Variance explained
    variance = clusters.get('pca_variance_ratio', [])
    ax.text(0.02, 0.95, f'PCA variance: {variance[0]:.1%} + {variance[1]:.1%} = {(variance[0]+variance[1]):.1%}',
           transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel('PC1 (Position-Direction Coupling)', fontsize=11)
    ax.set_ylabel('PC2 (Entropy vs Angle Change)', fontsize=11)
    ax.legend(fontsize=8, loc='lower right', ncol=2)
    ax.grid(alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'figures' / 'fig3-signature-clustering.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: fig3-signature-clustering.png")


# ============================================================
# FIGURE 4: Sentience properties comparison
# ============================================================

def create_sentience_figure(sentience_data):
    """Compare sentience properties across container shapes."""
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Sentience-Like Properties by Container Shape', 
                 fontsize=16, fontweight='bold')
    
    gs = GridSpec(3, 3, figure=fig)
    
    scores = sentience_data.get('sentience_scores', [])
    
    # Filter out NaN scores
    valid_scores = [s for s in scores if not math.isnan(s.get('composite_sentience_score', float('nan')))]
    
    names = [s['container'] for s in valid_scores]
    mem_scores = [s['memory'] for s in valid_scores]
    phi_scores = [s['phi'] for s in valid_scores]
    adapt_scores = [s['adaptation'] for s in valid_scores]
    composite = [s['composite_sentience_score'] for s in valid_scores]
    
    # Plot 1: Memory accuracy
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.barh(names, mem_scores, color='#2ecc71', edgecolor='white')
    ax1.set_xlabel('Self-Prediction Accuracy')
    ax1.set_title('Memory (Self-Prediction)', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 1.1)
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Phi (Information Integration)
    ax2 = fig.add_subplot(gs[0, 1])
    colors_phi = ['#3498db' if p > -0.2 else '#e74c3c' for p in phi_scores]
    ax2.barh(names, phi_scores, color=colors_phi, edgecolor='white')
    ax2.set_xlabel('Phi (Normalized Information Integration)')
    ax2.set_title('Information Integration (Phi)', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='k', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    
    # Plot 3: Adaptation rate
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.barh(names, adapt_scores, color='#e67e22', edgecolor='white')
    ax3.set_xlabel('TOF Adaptation Rate')
    ax3.set_title('Adaptation', fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # Plot 4: Composite score (bar chart)
    ax4 = fig.add_subplot(gs[1, :])
    colors_comp = ['#2ecc71' if c > 0.5 else '#f39c12' if c > 0.3 else '#e74c3c' for c in composite]
    bars = ax4.bar(names, composite, color=colors_comp, edgecolor='white', linewidth=1)
    ax4.set_ylabel('Composite Sentience Score')
    ax4.set_title('Overall Sentience Score (Memory 30% + Phi 30% + Differentiation 20% + Adaptation 20%)', 
                  fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 1.1)
    ax4.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, composite):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    
    # Plot 5: Memory vs Adaptation scatter
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.scatter(mem_scores, adapt_scores, s=100, alpha=0.7, edgecolors='black', linewidth=1)
    for i, name in enumerate(names):
        ax5.annotate(name.split('(')[0], (mem_scores[i], adapt_scores[i]), 
                    fontsize=7, ha='left')
    ax5.set_xlabel('Memory Accuracy')
    ax5.set_ylabel('Adaptation Rate')
    ax5.set_title('Memory vs Adaptation Trade-off', fontsize=11, fontweight='bold')
    ax5.grid(alpha=0.3)
    
    # Plot 6: Phi vs Composite
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.scatter(phi_scores, composite, s=100, alpha=0.7, edgecolors='black', linewidth=1)
    for i, name in enumerate(names):
        ax6.annotate(name.split('(')[0], (phi_scores[i], composite[i]), 
                    fontsize=7, ha='left')
    ax6.set_xlabel('Phi')
    ax6.set_ylabel('Composite Score')
    ax6.set_title('Phi vs Composite Score', fontsize=11, fontweight='bold')
    ax6.grid(alpha=0.3)
    
    # Plot 7: Periodicity by feedback strength (from recursion experiments)
    ax7 = fig.add_subplot(gs[2, 2])
    recursion = sentience_data.get('recursion_experiments', [])
    feedback_levels = sorted(set(r['feedback_strength'] for r in recursion))
    
    for container_name in set(r['container'] for r in recursion):
        data = [r['periodicity_mean'] for r in recursion if r['container'] == container_name]
        short_name = container_name.split('(')[0]
        ax7.plot(feedback_levels, data, 'o-', label=short_name, markersize=6)
    
    ax7.set_xlabel('Feedback Strength')
    ax7.set_ylabel('Periodicity Score')
    ax7.set_title('Recursion vs Feedback Strength', fontsize=11, fontweight='bold')
    ax7.legend(fontsize=7)
    ax7.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'figures' / 'fig4-sentience-properties.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: fig4-sentience-properties.png")


# ============================================================
# FIGURE 5: Conceptual diagram
# ============================================================

def create_concept_figure():
    """Conceptual diagram of the photon billiard computation hypothesis."""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(5, 9.5, 'Photon Billiard Computation Hypothesis', 
           ha='center', fontsize=20, fontweight='bold')
    ax.text(5, 9.0, 'Container Geometry → Trajectory Dynamics → Computational Signatures',
           ha='center', fontsize=12, style='italic')
    
    # Step 1: Container shapes
    y_pos = 7.5
    ax.text(5, y_pos, 'STEP 1: Container Geometry', ha='center', fontsize=14, fontweight='bold')
    
    shapes_y = [6.2, 5.0]
    shape_labels = [
        ('Sphere\nIntegrable\nZero Entropy', '#2ecc71'),
        ('Cube\nChaotic\nHigh Entropy', '#e74c3c'),
        ('Sinai\nHighly Chaotic\nMax Entropy', '#f39c12'),
    ]
    
    for i, (label, color) in enumerate(shape_labels):
        x = 2 + i * 3
        rect = Rectangle((x-0.8, shapes_y[i%2]-0.5), 1.6, 1.0, 
                        facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, shapes_y[i%2], label, ha='center', va='center', fontsize=9)
    
    # Arrow down
    ax.annotate('', xy=(5, 4.3), xytext=(5, 4.7),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(5.5, 4.5, 'Photon Entry', fontsize=10, style='italic')
    
    # Step 2: Trajectory
    ax.text(5, 4.0, 'STEP 2: Photon Trajectory (Bouncing)', ha='center', fontsize=14, fontweight='bold')
    
    # Draw a sample trajectory
    traj_x = np.linspace(1, 9, 100)
    traj_y = 2.5 + 0.5 * np.sin(3 * traj_x) * np.exp(-0.1 * traj_x)
    ax.plot(traj_x, traj_y, 'b-', alpha=0.6, linewidth=1.5)
    ax.plot(1, 2.5 + 0.5 * np.sin(3), 'go', markersize=10)
    ax.text(1, 2.0, 'Start', ha='center', fontsize=9)
    ax.plot(9, 2.5 + 0.5 * np.sin(27) * np.exp(-0.9), 'ro', markersize=10)
    ax.text(9, 2.0, 'End', ha='center', fontsize=9)
    
    # Arrow down
    ax.annotate('', xy=(5, 1.3), xytext=(5, 1.7),
               arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.text(5.5, 1.5, 'Extract Signatures', fontsize=10, style='italic')
    
    # Step 3: Computational signatures
    ax.text(5, 1.0, 'STEP 3: Computational Signatures', ha='center', fontsize=14, fontweight='bold')
    
    sig_labels = [
        ('Entropy\n(Predictability)', 0.5),
        ('Lyapunov\n(Chaos)', 2.0),
        ('Phi\n(Info Integration)', 3.5),
        ('Periodicity\n(Recursion)', 5.0),
        ('Adaptation\n(Learning)', 6.5),
        ('Differentiation\n(Discrimination)', 8.0),
    ]
    
    for label, x in sig_labels:
        ax.text(x, 1.5, label, ha='center', va='center', fontsize=8,
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent / 'figures' / 'fig5-concept-diagram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: fig5-concept-diagram.png")


# ============================================================
# MAIN
# ============================================================

def create_all_figures():
    """Create all figures."""
    
    figures_dir = Path(__file__).parent / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    print("Loading results...")
    billiard, sentience = load_results()
    
    print("\nCreating figures...")
    create_trajectory_figures()
    create_entropy_lyapunov_figure(billiard)
    create_signature_clustering_figure(billiard)
    create_sentience_figure(sentience)
    create_concept_figure()
    
    print("\nAll figures created in: figures/")
    for f in figures_dir.glob('*.png'):
        print(f"  {f.name}")


if __name__ == "__main__":
    create_all_figures()
