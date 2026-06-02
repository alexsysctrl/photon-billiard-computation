# Local Governor Photon Billiard Simulation

A computational investigation of proximity-based governance in photon billiard systems. Each universe (container) has a single governing body at its center that only affects photons when they come close — creating a natural Goldilocks zone where photons must dance around the governor: close enough to feed, far enough to survive.

## Architecture

### The Governor
A central governing body with a numerical value (7, 13, 42, 100, 333, 999) that sits at the origin of each universe. It has:
- **Interaction radius**: 5.0 units — photons within this range are affected
- **Proximity threshold**: 2.0 units — photons within this range receive maximum effect
- **Energy field**: absorbs/emits energy to photons based on proximity
- **Digit extraction**: extracts the thousandths digit from `distance / governor_value` to determine photon size class

### Photons
Extremely small particles (radius ~0.001) that:
- Start on a ring 3.0 units from the governor with tangential velocity
- Move at high speed (5-15 units per step)
- Absorb energy from the governor when close (growing in size)
- Can spawn new photons when they grow too large (radius > 0.12)
- Can be eliminated if they absorb too much energy (KE > 50)
- Get repelled if they get too close to the governor (< 0.6 units)

### Simulation Mechanics
- **Micro-steps**: Each macro-step = 5 micro-steps, allowing photons to pass through the interaction zone
- **Container walls**: Only checked at macro-step boundaries
- **Governor interaction**: Happens during every micro-step when photons are in range

### Containers (Universes)
| Container | Geometry | Behavior |
|-----------|----------|----------|
| **Sphere** | r=10 | Photons orbit freely, strong interaction, cascading spawning |
| **Cube** | s=20 | Photons bounce off walls, rarely reach center, minimal interaction |
| **Ellipsoid** | a=15,b=10,c=8 | Moderate interaction, some spawning, no deaths |
| **Sinai** | s=20, obs=4 | Chaotic obstacle pulls photons inward, total extinction |
| **Cylinder** | r=10,h=20 | Moderate interaction, stable spawning ecosystem |

## Results

### Energy Production (Total Energy at Step 2000)

| Container | Best Governor | Total Energy | Change | Survivors |
|-----------|--------------|--------------|--------|-----------|
| **Sphere** | **333** | **536.01** | **+751%** | **38/40** |
| Sphere | 100 | 470.14 | +671% | 23/26 |
| Sphere | 999 | 427.33 | +619% | 20/22 |
| Sphere | 42 | 375.88 | +556% | 15/18 |
| Sphere | 7 | 241.80 | +394% | 24/27 |
| Ellipsoid | 333 | -1.44 | +98% | 9/9 |
| Ellipsoid | 999 | -2.68 | +97% | 10/10 |
| Cylinder | 999 | -64.70 | +21% | 8/8 |
| Cube | 7 | -74.29 | +10% | 3/3 |
| **Sinai** | **Any** | **0.00** | **100%** | **0/3** |

### Key Findings

1. **Sphere (G=333) wins** — 536 total energy, +751% growth, 38/40 photons survived. The sphere allows photons to orbit freely and interact with the governor repeatedly, creating cascading spawning.

2. **Sinai is a death trap** — 0% survival across all governors. The chaotic circular obstacle deflects photons toward the center where they absorb too much energy and are eliminated.

3. **Cube is stable but dead** — 100% survival, but only +10% energy growth. Photons bounce off walls and rarely reach the interaction zone.

4. **Governor 333 is optimal** — highest energy across all containers that survive, most spawns (37 in sphere), fewest deaths (2). The digit extraction from `distance/333` creates rich variation in photon sizes.

5. **Digit diversity matters** — Governor 333 divides distance by 333, producing varied thousandths digits (0-9) that create diverse photon sizes. Governor 7 divides by 7, producing coarser digit resolution that leads to uniform sizes and eventual elimination.

6. **Goldilocks containers** — Ellipsoid and cylinder provide moderate interaction: some spawning, no deaths, slow but steady energy growth.

### The Goldilocks Principle

The system demonstrates a fundamental Goldilocks principle:
- **Too chaotic** (Sinai): photons pulled into governor → death
- **Too stable** (Cube): photons never reach governor → stagnation
- **Just right** (Sphere): photons orbit, feed, spawn, survive → exponential growth
- **Moderate** (Ellipsoid, Cylinder): limited interaction, slow growth, high survival

## Files

- `local_governor_simulation.py` — Main simulation code
- `create_local_governor_figures.py` — Figure generation
- `local_governor_results.json` — Full simulation results
- `figures/` — Publication-quality figures

## Figures

1. **fig1-energy-heatmap.png** — Energy production heatmap across containers and governors
2. **fig2-energy-growth.png** — Energy growth percentage and spawn events
3. **fig3-energy-timelines.png** — Energy, survival, and proximity timelines
4. **fig4-survival.png** — Survival rates and elimination analysis
5. **fig5-concept-diagram.png** — Visual overview of each container's dynamics

## Mathematical Foundation

### Digit Extraction
```
digit = floor(|distance / governor_value| × 1000) mod 10
```
This creates a discrete size class (0-9) from continuous distance, encoding the continuous billiard dynamics into a discrete computational layer.

### Proximity Factor
```
pf = ((interaction_radius - dist) / (interaction_radius - proximity_threshold))²
```
Quadratic falloff from 1.0 at proximity threshold to 0.0 at interaction radius.

### Energy Absorption
```
energy_absorbed = pf × absorption_rate × speed × 0.02
```
Proportional to proximity, speed, and governor absorption rate.

### Size Growth
```
new_radius = (0.001 + digit × 0.001) + energy_absorbed × 0.05 + growth_steps × 0.0005
```
Base size from digit + energy contribution + cumulative growth.

## Reproducing Results

```bash
python3 local_governor_simulation.py
python3 create_local_governor_figures.py
```

## Citation

```bibtex
@misc{photon_billiard_local_governor,
  title = {Local Governor Photon Billiard Simulation},
  author = {Alex},
  year = {2025},
  url = {https://github.com/alexsysctrl/local-governor-photon-billiard}
}
```

## License

MIT
