# Photon Billiard Computation Hypothesis

A computational investigation into whether photon billiard systems in complex containers can produce emergent computational behaviors with properties associated with cognition and sentience.

## Overview

This project explores a novel computational hypothesis: that a photon (point particle) confined to a closed container and bouncing off walls via specular reflection can encode mathematical information through its collision trajectory. The core question is whether **container geometry + initial conditions → trajectory signature** constitutes a computational primitive.

## Key Findings

### 1. Container Geometry Determines Computational Properties

| Container | Entropy | Lyapunov Exponent | Classification |
|-----------|---------|-------------------|----------------|
| Sphere | 0.0000 | -0.0005 | Integrable (predictable) |
| Ellipsoid | 0.0000 | +0.0003 | Integrable (predictable) |
| Cylinder | 1.2389 | -0.0002 | Moderately chaotic |
| Cube | 2.3608 | +0.0228 | Chaotic |
| Sinai | 2.7415 | +0.0745 | Highly chaotic |

**Finding:** Integrable shapes (sphere, ellipsoid) produce zero-entropy, stable trajectories. Chaotic shapes (cube, Sinai) produce high-entropy, sensitive trajectories. The container geometry is the dominant factor in trajectory behavior (99.4% variance explained by 3 PCA dimensions).

### 2. Signature Space Clustering

Trajectories from different container shapes form distinct, non-overlapping clusters in signature space. This means the container shape is **readable** from the trajectory alone — a form of computational encoding.

### 3. Sentience-Like Properties

We measured six properties associated with cognition:

| Property | Sphere | Cube | Cylinder | Ellipsoid | Sinai |
|----------|--------|------|----------|-----------|-------|
| Memory (self-prediction) | 1.000 | 0.191 | 0.571 | 1.000 | 0.104 |
| Phi (info integration) | -0.426 | -0.081 | -0.254 | -0.270 | -0.092 |
| Adaptation | 0.025 | 0.055 | 0.054 | 0.021 | NaN |
| **Composite Score** | **0.591** | **0.395** | **0.483** | **0.613** | **NaN** |

**Finding:** Integrable shapes (sphere, ellipsoid) show high memory accuracy (perfect self-prediction) but negative phi. Chaotic shapes (cube, Sinai) show low memory but higher phi and adaptation.

### 4. Recursion and Feedback

Feedback strength increases periodicity in chaotic containers:
- Sphere: always periodic (1.0) regardless of feedback
- Cube: periodicity increases from 0.63 to 1.0 as feedback increases
- Sinai: periodicity increases from 0.29 to 0.60 with feedback

## Architecture

```
Photon_Billiard_Computation/
├── billiard_simulation.py      # Core simulation engine
├── sentience_experiments.py    # Sentience property measurement
├── create_figures.py           # Figure generation
├── billiard_results.json       # Full simulation results
├── sentience_results.json      # Sentience experiment results
├── figures/                    # Publication-quality figures
│   ├── fig1-trajectories.png
│   ├── fig2-entropy-lyapunov.png
│   ├── fig3-signature-clustering.png
│   ├── fig4-sentience-properties.png
│   └── fig5-concept-diagram.png
├── manuscript.md               # Full manuscript
└── README.md                   # This file
```

## Computational Pipeline

1. **Container Definition** — Define closed 3D shape (sphere, cube, cylinder, ellipsoid, Sinai billiard)
2. **Photon Initialization** — Set initial position and velocity vector inside container
3. **Trajectory Simulation** — Ray-cast through container, specular reflection at each wall collision
4. **Signature Extraction** — Compute entropy, Lyapunov exponent, Phi, periodicity, adaptation, differentiation
5. **Clustering** — Embed signatures in PCA space, identify container-specific clusters

## Mathematical Foundations

This work connects to several established areas:

- **Billiard Dynamics** — Classical mechanical systems with hard-wall boundaries (Tabor, 1989)
- **Spectral Geometry** — "Can you hear the shape of a drum?" (Kac, 1966)
- **Chaotic Billiards** — Sinai billiard, stadium billiard (Sinai, 1970; Lochak, 1984)
- **Integrated Information Theory** — Phi measure of consciousness (Tononi, 2004)
- **Periodic Orbit Theory** — Gutzwiller trace formula (Gutzwiller, 1971)
- **Random Matrix Theory** — Quantum chaos, energy level statistics (Haake, 2010)

## Novel Contributions

1. **First systematic comparison** of photon billiard signatures across multiple container geometries
2. **Signature space clustering** demonstrating that container shape is readable from trajectory alone
3. **Sentience property measurement** applied to billiard systems (memory, integration, recursion, differentiation, adaptation, phi)
4. **Feedback-modified billiard dynamics** showing how output can influence future input
5. **Computational primitive proposal** — billiard trajectories as information encoding mechanism

## Running the Simulation

```bash
# Run billiard experiments
python3 billiard_simulation.py

# Run sentience experiments
python3 sentience_experiments.py

# Generate figures
python3 create_figures.py
```

**Requirements:** Python 3.10+, numpy, matplotlib

## Limitations

- Simulations use classical (non-quantum) photons
- 3D containers approximated with simplified collision detection
- Phi measure is an approximation (not the full IIT calculation)
- Sentience properties are operational definitions, not proof of consciousness
- No experimental validation — purely computational

## Future Work

- Quantum billiard simulations (wave function evolution)
- Higher-dimensional containers (4D+)
- Adaptive containers (geometry changes based on trajectory)
- Machine learning classification of containers from trajectories
- Experimental validation with optical billiard systems
- Connection to quantum computing (billiard-based quantum gates)

## References

- Kac, M. (1966). "Can One Hear the Shape of a Drum?" American Mathematical Monthly
- Sinai, Y. G. (1970). "Dynamics of a hard sphere particle in a periodic array of fixed spherical centers"
- Gutzwiller, M. C. (1971). "Classical chaos and quantum spectra"
- Tononi, G. (2004). "An information integration theory of consciousness"
- Tabor, M. (1989). "Chaos and Integrability in Nonlinear Dynamics"
- Lochak, P. (1984). "Classical and quantum dynamics in chaotic billiards"
- Haake, F. (2010). "Quantum Signatures of Chaos"

## License

MIT — feel free to use, modify, and build upon this research.

---

*Built with computational physics, dynamical systems theory, and curiosity about the nature of computation itself.*
