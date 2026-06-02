#!/usr/bin/env python3
"""
EVOLUTIONARY PHOTON BILLIARD SIMULATION
========================================

Extends the photon billiard with an evolutionary game layer:

- 3+ photons bouncing continuously in a 3D container
- Each photon carries full physical state: KE, PE, speed, velocity, momentum, mass, frequency
- At random intervals, the system "reads" a photon's 3D position
- That position encodes a perturbation value that changes how everything interacts
- Perturbations can: alter time flow, change photon energies, shift trajectories,
  scale photon size, eliminate photons, spawn new ones, change container shape
- The system evolves over time — we look for patterns in survival, energy distribution,
  trajectory clustering, and emergent behavior

This is a computational model of how a physical system can encode
evolutionary dynamics through position-dependent perturbation rules.
"""

import numpy as np
import json
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import random


# ============================================================
# VECTOR MATH
# ============================================================

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-15 else v

def dot(a, b):
    return np.dot(a, b)

def reflect(direction, normal):
    return direction - 2.0 * dot(direction, normal) * normal


# ============================================================
# CONTAINERS
# ============================================================

class Sphere:
    def __init__(self, radius=1.0):
        self.radius = radius
        self.name = f"sphere(r={radius})"
        self.base_radius = radius

    def get_intersections(self, pos, direction, max_dist=100.0):
        op = pos
        a = dot(direction, direction)
        b = 2.0 * dot(op, direction)
        c = dot(op, op) - self.radius ** 2
        disc = b * b - 4 * a * c
        if disc < 0:
            return []
        sqrt_d = math.sqrt(disc)
        results = []
        for t in [(-b - sqrt_d) / (2 * a), (-b + sqrt_d) / (2 * a)]:
            if t > 1e-10 and t < max_dist:
                hp = pos + t * direction
                normal = normalize(hp)
                results.append((t, hp, normal, "sphere"))
        return results

    def is_inside(self, point):
        return np.linalg.norm(point) < self.radius - 1e-10

    def volume(self):
        return 4.0 / 3.0 * math.pi * self.radius ** 3


class Cube:
    def __init__(self, size=2.0):
        self.size = size
        self.half = size / 2
        self.name = f"cube(s={size})"
        self.base_size = size

    def get_intersections(self, pos, direction, max_dist=100.0):
        results = []
        faces = [(0, 1, self.half, "x+"), (0, -1, -self.half, "x-"),
                 (1, 1, self.half, "y+"), (1, -1, -self.half, "y-"),
                 (2, 1, self.half, "z+"), (2, -1, -self.half, "z-")]
        for axis, sign, coord, label in faces:
            if abs(direction[axis]) < 1e-15:
                continue
            t = (coord - pos[axis]) / direction[axis]
            if t > 1e-10 and t < max_dist:
                hp = pos + t * direction
                others = [i for i in range(3) if i != axis]
                if all(-self.half <= hp[i] <= self.half for i in others):
                    normal = np.array([0.0, 0.0, 0.0])
                    normal[axis] = sign
                    results.append((t, hp, normal, label))
        return results

    def is_inside(self, point):
        return all(-self.half < p < self.half for p in point)

    def volume(self):
        return self.size ** 3


class Ellipsoid:
    def __init__(self, a=1.5, b=1.0, c=0.8):
        self.a, self.b, self.c = a, b, c
        self.name = f"ellipsoid(a={a},b={b},c={c})"
        self.base_a, self.base_b, self.base_c = a, b, c

    def get_intersections(self, pos, direction, max_dist=100.0):
        ax = direction[0] / self.a
        ay = direction[1] / self.b
        az = direction[2] / self.c
        ox = pos[0] / self.a
        oy = pos[1] / self.b
        oz = pos[2] / self.c
        a_q = ax**2 + ay**2 + az**2
        b_q = 2 * (ox*ax + oy*ay + oz*az)
        c_q = ox**2 + oy**2 + oz**2 - 1
        disc = b_q**2 - 4 * a_q * c_q
        if disc < 0:
            return []
        sqrt_d = math.sqrt(disc)
        results = []
        for t in [(-b_q - sqrt_d) / (2*a_q), (-b_q + sqrt_d) / (2*a_q)]:
            if t > 1e-10 and t < max_dist:
                hp = pos + t * direction
                normal = np.array([hp[0]/self.a**2, hp[1]/self.b**2, hp[2]/self.c**2])
                nl = np.linalg.norm(normal)
                if nl > 1e-15:
                    normal /= nl
                results.append((t, hp, normal, "ellipsoid"))
        return results

    def is_inside(self, point):
        return (point[0]/self.a)**2 + (point[1]/self.b)**2 + (point[2]/self.c)**2 < 1.0 - 1e-10

    def volume(self):
        return 4.0/3.0 * math.pi * self.a * self.b * self.c


class SinaiBilliard:
    def __init__(self, size=2.0, obs_r=0.4):
        self.size = size
        self.half = size / 2
        self.obs_r = obs_r
        self.name = f"sinai(s={size},obs={obs_r})"
        self.base_size = size
        self.base_obs_r = obs_r

    def get_intersections(self, pos, direction, max_dist=100.0):
        results = []
        faces = [(0, 1, self.half, "x+"), (0, -1, -self.half, "x-"),
                 (1, 1, self.half, "y+"), (1, -1, -self.half, "y-"),
                 (2, 1, self.half, "z+"), (2, -1, -self.half, "z-")]
        for axis, sign, coord, label in faces:
            if abs(direction[axis]) < 1e-15:
                continue
            t = (coord - pos[axis]) / direction[axis]
            if t > 1e-10 and t < max_dist:
                hp = pos + t * direction
                others = [i for i in range(3) if i != axis]
                if all(-self.half <= hp[i] <= self.half for i in others):
                    if np.hypot(hp[0], hp[1]) > self.obs_r + 1e-10:
                        normal = np.array([0, 0, 0])
                        normal[axis] = sign
                        results.append((t, hp, normal, label))
        r_op = np.array([pos[0], pos[1], 0])
        r_dir = np.array([direction[0], direction[1], 0])
        a = dot(r_dir, r_dir)
        b = 2 * dot(r_op, r_dir)
        c = dot(r_op, r_op) - self.obs_r**2
        disc = b*b - 4*a*c
        if disc >= 0 and a > 1e-15:
            sqrt_d = math.sqrt(disc)
            for t in [(-b - sqrt_d)/(2*a), (-b + sqrt_d)/(2*a)]:
                if t > 1e-10 and t < max_dist:
                    hp = pos + t * direction
                    if abs(hp[2]) < self.half - 1e-10:
                        normal = np.array([hp[0], hp[1], 0])
                        nl = np.linalg.norm(normal)
                        if nl > 1e-15:
                            normal /= nl
                        results.append((t, hp, normal, "sinai-obs"))
        return results

    def is_inside(self, point):
        if not all(-self.half < p < self.half for p in point):
            return False
        return np.hypot(point[0], point[1]) > self.obs_r + 1e-10

    def volume(self):
        return self.size**2 * self.size - math.pi * self.obs_r**2 * self.size


# ============================================================
# PHOTON WITH FULL PHYSICAL STATE
# ============================================================

@dataclass
class PhotonState:
    """Complete physical state of a photon."""
    # Position
    pos: List[float]                        # [x, y, z]
    # Velocity / direction
    velocity: List[float]                   # [vx, vy, vz] (normalized direction * speed)
    speed: float                            # magnitude of velocity
    # Energy
    kinetic_energy: float                   # E = hf
    potential_energy: float                 # depends on container depth
    total_energy: float                     # KE + PE
    # Momentum
    momentum: List[float]                   # m * v
    momentum_magnitude: float               # |p|
    # Mass (effective mass from E=mc^2)
    mass: float                             # E/c^2
    # Wave properties
    frequency: float                        # Hz
    wavelength: float                       # m
    wave_vector: List[float]                # [kx, ky, kz]
    # Size (effective cross-section for interactions)
    radius: float                           # effective interaction radius
    # Identity
    id: int
    birth_step: int                         # when this photon was created
    alive: bool                             # is this photon still bouncing?
    # Interaction history
    collisions: int                         # total wall bounces
    perturbations_received: int             # how many times it was perturbed
    perturbations_applied: int              # how many times it perturbed others
    total_path_length: float                # cumulative distance traveled

    def to_dict(self):
        return asdict(self)


def create_photon(pid, pos, velocity, container, rng):
    """Create a photon with full physical state."""
    speed = np.linalg.norm(velocity)
    if speed < 1e-15:
        speed = 1.0
        velocity = velocity / np.linalg.norm(velocity)

    # Photon energy: E = h * f (using arbitrary units, h=1)
    # Speed of light c = 1 in our units
    # Effective mass: m = E/c^2 = E
    frequency = 1.0 + rng.random() * 9.0  # 1-10 Hz
    kinetic_energy = frequency  # E = hf, h=1
    mass = kinetic_energy  # m = E/c^2, c=1
    momentum = velocity * mass
    momentum_mag = np.linalg.norm(momentum)

    # Wavelength: lambda = c/f
    wavelength = 1.0 / frequency if frequency > 1e-15 else 1.0
    wave_vector = velocity * (2 * math.pi / wavelength)

    # Potential energy: depends on position in container
    if isinstance(container, Sphere):
        r = np.linalg.norm(pos)
        pe = 0.5 * (r / container.radius) ** 2
    elif isinstance(container, (Cube, SinaiBilliard)):
        max_dim = container.half if hasattr(container, 'half') else container.size / 2
        r = np.linalg.norm(pos)
        pe = 0.5 * (r / max_dim) ** 2
    elif isinstance(container, Ellipsoid):
        r_sq = (pos[0]/container.a)**2 + (pos[1]/container.b)**2 + (pos[2]/container.c)**2
        pe = 0.5 * r_sq
    else:
        pe = 0.0

    return PhotonState(
        pos=[round(p, 8) for p in pos],
        velocity=[round(v, 8) for v in velocity],
        speed=round(speed, 8),
        kinetic_energy=round(kinetic_energy, 8),
        potential_energy=round(pe, 8),
        total_energy=round(kinetic_energy + pe, 8),
        momentum=[round(m, 8) for m in momentum],
        momentum_magnitude=round(momentum_mag, 8),
        mass=round(mass, 8),
        frequency=round(frequency, 8),
        wavelength=round(wavelength, 8),
        wave_vector=[round(w, 8) for w in wave_vector],
        radius=0.01 + rng.random() * 0.04,  # 0.01-0.05 effective radius
        id=pid,
        birth_step=0,
        alive=True,
        collisions=0,
        perturbations_received=0,
        perturbations_applied=0,
        total_path_length=0.0
    )


# ============================================================
# POSITION-TO-PERTURBATION ENCODING
# ============================================================

def encode_position_to_perturbation(photon, step, rng):
    """
    Read a photon's 3D position and encode it as a perturbation value.

    The position [x, y, z] is mapped through a nonlinear transformation
    to produce a perturbation vector that affects the system.

    This is the "computation" — the photon's position determines
    what kind of evolutionary pressure is applied.
    """
    x, y, z = photon.pos

    # 1. Time dilation factor (how fast/slow time flows)
    time_factor = 0.5 + 0.5 * math.sin(x * math.pi) * math.cos(y * math.pi)

    # 2. Energy perturbation (add/subtract energy from all photons)
    energy_perturb = math.sin(z * math.pi * 2) * 0.5

    # 3. Trajectory deflection (change photon directions)
    deflection_x = math.sin(x * math.pi * 3) * 0.3
    deflection_y = math.cos(y * math.pi * 3) * 0.3
    deflection_z = math.sin(z * math.pi * 3) * 0.3

    # 4. Scaling factor (change photon sizes)
    scale_factor = 0.5 + 0.5 * math.cos(x * y * math.pi)

    # 5. Elimination probability (chance to kill a photon)
    elimination_prob = max(0, math.sin(x * math.pi) * math.sin(y * math.pi) * math.sin(z * math.pi))

    # 6. Spawn probability (chance to create a new photon)
    spawn_prob = max(0, math.cos(x * math.pi * 2) * math.cos(y * math.pi * 2) * math.cos(z * math.pi * 2))

    # 7. Container deformation (change container shape)
    container_perturb_x = math.sin(x * math.pi * 2) * 0.1
    container_perturb_y = math.cos(y * math.pi * 2) * 0.1

    # 8. Interaction strength (how strongly photons affect each other)
    interaction_strength = 0.5 + 0.5 * math.sin((x + y + z) * math.pi)

    # 9. Random perturbation component (stochasticity)
    random_ke = rng.uniform(-0.3, 0.3)
    random_angle = rng.uniform(-0.1, 0.1)
    random_scale = rng.uniform(0.9, 1.1)

    return {
        'time_factor': time_factor,
        'energy_perturb': energy_perturb,
        'deflection': [deflection_x, deflection_y, deflection_z],
        'scale_factor': scale_factor,
        'elimination_prob': elimination_prob,
        'spawn_prob': spawn_prob,
        'container_perturb': [container_perturb_x, container_perturb_y],
        'interaction_strength': interaction_strength,
        'random_ke': random_ke,
        'random_angle': random_angle,
        'random_scale': random_scale,
        'source_photon_id': photon.id,
        'step': step,
        'source_position': photon.pos.copy(),
    }


# ============================================================
# APPLY PERTURBATIONS
# ============================================================

def apply_perturbations(perturbation, photons, container, step, rng):
    """
    Apply a perturbation to the system based on encoded position.

    Returns list of events that occurred (energy changes, eliminations, spawns, etc.)
    """
    events = []

    ep = perturbation['energy_perturb']
    defl = perturbation['deflection']
    sf = perturbation['scale_factor']
    elim_p = perturbation['elimination_prob']
    spawn_p = perturbation['spawn_prob']
    ip = perturbation['container_perturb']
    ist = perturbation['interaction_strength']
    rke = perturbation['random_ke']
    rang = perturbation['random_angle']
    rsf = perturbation['random_scale']

    for photon in photons:
        if not photon.alive:
            continue

        # 1. Energy perturbation: shift all photons' energies
        if ep != 0 or rke != 0:
            old_ke = photon.kinetic_energy
            photon.kinetic_energy += ep * photon.mass + rke * photon.mass
            photon.kinetic_energy = max(0.1, photon.kinetic_energy)
            photon.total_energy = photon.kinetic_energy + photon.potential_energy
            photon.momentum = np.array(photon.velocity) * photon.mass
            photon.momentum_magnitude = np.linalg.norm(photon.momentum)
            photon.frequency = photon.kinetic_energy  # E = hf
            photon.wavelength = 1.0 / photon.frequency if photon.frequency > 1e-15 else 1.0
            photon.wave_vector = np.array(photon.velocity) * (2 * math.pi / photon.wavelength)
            photon.perturbations_received += 1
            if abs(old_ke - photon.kinetic_energy) > 0.01:
                events.append({
                    'type': 'energy_shift',
                    'photon_id': photon.id,
                    'old_ke': round(old_ke, 6),
                    'new_ke': round(photon.kinetic_energy, 6),
                    'delta': round(photon.kinetic_energy - old_ke, 6),
                })

        # 2. Trajectory deflection: nudge photon direction
        if any(abs(d) > 0.01 for d in defl) or abs(rang) > 0.01:
            v = np.array(photon.velocity)
            v += np.array(defl) * photon.radius  # deflection proportional to size
            # Add random angular perturbation
            angle = rang * photon.radius
            v[0] += math.sin(angle) * 0.1
            v[1] += math.cos(angle) * 0.1
            v = normalize(v)
            photon.velocity = [round(vi, 8) for vi in v]
            photon.perturbations_received += 1

        # 3. Scaling: change photon size
        photon.radius *= sf * rsf
        photon.radius = max(0.005, min(0.1, photon.radius))  # clamp
        photon.perturbations_received += 1

        # 4. Elimination: probabilistic death
        if elim_p > 0.2 and rng.random() < elim_p * 0.15:
            photon.alive = False
            events.append({
                'type': 'elimination',
                'photon_id': photon.id,
                'step': step,
                'reason': 'position-encoded_elimination',
                'source_position': perturbation['source_position'],
            })

    # 5. Spawn: create new photon (if enough alive photons)
    alive_photons = [p for p in photons if p.alive]
    if spawn_p > 0.4 and len(alive_photons) >= 2 and step % 100 == 0:
        # Pick a random alive photon to be the parent
        parent = rng.choice(alive_photons)
        spawn_pos = np.array(parent.pos) + rng.uniform(-0.15, 0.15, 3)
        spawn_dir = rng.uniform(-1, 1, 3)
        spawn_dir = normalize(spawn_dir)
        # Verify inside container
        if not container.is_inside(spawn_pos):
            spawn_pos = np.array(parent.pos)
        new_photon = create_photon(
            pid=max(p.id for p in photons) + 1,
            pos=spawn_pos,
            velocity=spawn_dir,
            container=container,
            rng=rng
        )
        new_photon.birth_step = step
        photons.append(new_photon)
        parent.perturbations_applied += 1
        events.append({
            'type': 'spawn',
            'new_photon_id': new_photon.id,
            'parent_id': parent.id,
            'step': step,
            'spawn_position': [round(p, 4) for p in spawn_pos],
        })

    # 6. Container deformation (for ellipsoid)
    if isinstance(container, Ellipsoid) and any(abs(p) > 0.001 for p in ip):
        container.a += ip[0]
        container.b += ip[1]
        container.a = max(0.5, min(3.0, container.a))
        container.b = max(0.5, min(3.0, container.b))
        container.c = container.b * 0.8  # keep aspect ratio
        events.append({
            'type': 'container_deform',
            'new_a': round(container.a, 4),
            'new_b': round(container.b, 4),
            'new_c': round(container.c, 4),
        })

    # 7. Photon-photon interaction (based on interaction strength)
    if ist > 0.5:
        alive_photons = [p for p in photons if p.alive]
        for i in range(len(alive_photons)):
            for j in range(i + 1, len(alive_photons)):
                p1 = alive_photons[i]
                p2 = alive_photons[j]
                dist = np.linalg.norm(np.array(p1.pos) - np.array(p2.pos))
                # If photons are close, they interact
                if dist < (p1.radius + p2.radius) * 5:  # interaction range
                    # Energy exchange
                    energy_diff = (p1.kinetic_energy - p2.kinetic_energy) * ist * 0.1
                    p1.kinetic_energy -= energy_diff
                    p2.kinetic_energy += energy_diff
                    p1.kinetic_energy = max(0.1, p1.kinetic_energy)
                    p2.kinetic_energy = max(0.1, p2.kinetic_energy)
                    p1.total_energy = p1.kinetic_energy + p1.potential_energy
                    p2.total_energy = p2.kinetic_energy + p2.potential_energy
                    p1.perturbations_received += 1
                    p2.perturbations_received += 1
                    p1.perturbations_applied += 1
                    p2.perturbations_applied += 1
                    events.append({
                        'type': 'photon_interaction',
                        'photon_1': p1.id,
                        'photon_2': p2.id,
                        'distance': round(dist, 6),
                        'energy_transfer': round(energy_diff, 6),
                    })

    return events


# ============================================================
# COLLISION DETECTION
# ============================================================

def find_closest_wall(pos, direction, container, max_dist=100.0):
    hits = container.get_intersections(pos, direction, max_dist)
    if not hits:
        return None
    best = min(hits, key=lambda h: h[0])
    return best


# ============================================================
# MAIN SIMULATION LOOP
# ============================================================

def run_evolutionary_simulation(
    container,
    n_photons=3,
    max_steps=5000,
    read_interval=50,  # read position every N steps
    rng_seed=42,
):
    """
    Run the evolutionary photon billiard simulation.
    """
    rng = np.random.RandomState(rng_seed)
    random.seed(rng_seed)

    # Create photons
    photons = []
    for i in range(n_photons):
        while True:
            pos = rng.uniform(-0.5, 0.5, 3)
            if container.is_inside(pos):
                break
        vel = rng.uniform(-1, 1, 3)
        vel = normalize(vel)
        photons.append(create_photon(i, pos, vel, container, rng))

    # Tracking
    step = 0
    history = {
        'container': container.name,
        'n_initial_photons': n_photons,
        'max_steps': max_steps,
        'read_interval': read_interval,
        'events': [],
        'photon_states': [],
        'energy_timeline': [],
        'survival_timeline': [],
        'perturbation_summary': [],
    }

    print(f"Starting evolutionary simulation:")
    print(f"  Container: {container.name}")
    print(f"  Photons: {n_photons}")
    print(f"  Max steps: {max_steps}")
    print(f"  Read interval: every {read_interval} steps")
    print()

    while step < max_steps:
        # ---- Move all photons and detect collisions ----
        for photon in photons:
            if not photon.alive:
                continue

            hit = find_closest_wall(
                np.array(photon.pos),
                np.array(photon.velocity),
                container,
                max_dist=100.0
            )

            if hit:
                t, hp, normal, wall = hit
                photon.pos = [round(p, 8) for p in hp]
                v = reflect(np.array(photon.velocity), normal)
                v = normalize(v)
                photon.velocity = [round(vi, 8) for vi in v]
                photon.pos = [round(p + vi * 1e-8, 8) for p, vi in zip(photon.pos, v)]
                photon.collisions += 1
                photon.total_path_length += t

        step += 1

        # ---- Periodic position read -> perturbation ----
        if step % read_interval == 0:
            alive = [p for p in photons if p.alive]
            if not alive:
                print(f"  Step {step}: All photons eliminated. Stopping.")
                break

            source = rng.choice(alive)
            perturbation = encode_position_to_perturbation(source, step, rng)

            events = apply_perturbations(perturbation, photons, container, step, rng)

            history['perturbation_summary'].append({
                'step': step,
                'source_id': source.id,
                'source_pos': source.pos,
                'time_factor': round(perturbation['time_factor'], 4),
                'energy_perturb': round(perturbation['energy_perturb'], 4),
                'elimination_prob': round(perturbation['elimination_prob'], 4),
                'spawn_prob': round(perturbation['spawn_prob'], 4),
                'interaction_strength': round(perturbation['interaction_strength'], 4),
                'n_events': len(events),
            })

            for e in events:
                e['step'] = step
                history['events'].append(e)

        # ---- Record snapshot ----
        alive_photons = [p for p in photons if p.alive]
        total_ke = sum(p.kinetic_energy for p in alive_photons)
        total_pe = sum(p.potential_energy for p in alive_photons)
        total_energy = total_ke + total_pe

        history['energy_timeline'].append({
            'step': step,
            'n_alive': len(alive_photons),
            'total_ke': round(total_ke, 6),
            'total_pe': round(total_pe, 6),
            'total_energy': round(total_energy, 6),
            'avg_ke': round(total_ke / max(len(alive_photons), 1), 6),
            'max_ke': round(max((p.kinetic_energy for p in alive_photons), default=0), 6),
            'min_ke': round(min((p.kinetic_energy for p in alive_photons), default=0), 6),
        })

        history['survival_timeline'].append({
            'step': step,
            'n_alive': len(alive_photons),
            'n_total': len(photons),
        })

        # Record photon states (every 100 steps to save space)
        if step % 100 == 0 or step == max_steps:
            for p in photons:
                history['photon_states'].append({
                    'step': step,
                    'photon': p.to_dict(),
                })

        # Progress reporting
        if step % 500 == 0:
            print(f"  Step {step:5d} | alive={len(alive_photons)}/{len(photons)} | "
                  f"total_E={total_energy:.4f} | events_so_far={len(history['events'])}")

    # ---- Final snapshot ----
    history['final_step'] = step
    history['final_photons'] = [p.to_dict() for p in photons]

    # ---- Summary statistics ----
    n_total_created = len(photons)
    n_final_alive = sum(1 for p in photons if p.alive)
    n_eliminated = n_total_created - n_final_alive
    n_spawned = sum(1 for e in history['events'] if e.get('type') == 'spawn')

    if history['energy_timeline']:
        initial_E = history['energy_timeline'][0]['total_energy']
        final_E = history['energy_timeline'][-1]['total_energy']
        max_E = max(e['total_energy'] for e in history['energy_timeline'])
        min_E = min(e['total_energy'] for e in history['energy_timeline'])
    else:
        initial_E = final_E = max_E = min_E = 0

    event_counts = Counter(e.get('type') for e in history['events'])

    survival_data = [s['n_alive'] for s in history['survival_timeline']]
    half_life = None
    if survival_data and survival_data[0] > 0:
        target = survival_data[0] / 2
        for i, s in enumerate(survival_data):
            if s <= target:
                half_life = i
                break

    if history['perturbation_summary']:
        avg_elim_prob = np.mean([p['elimination_prob'] for p in history['perturbation_summary']])
        avg_spawn_prob = np.mean([p['spawn_prob'] for p in history['perturbation_summary']])
        avg_interaction = np.mean([p['interaction_strength'] for p in history['perturbation_summary']])
        avg_time_factor = np.mean([p['time_factor'] for p in history['perturbation_summary']])
    else:
        avg_elim_prob = avg_spawn_prob = avg_interaction = avg_time_factor = 0

    summary = {
        'container': container.name,
        'n_initial_photons': n_photons,
        'n_total_created': n_total_created,
        'n_final_alive': n_final_alive,
        'n_eliminated': n_eliminated,
        'n_spawned': n_spawned,
        'max_steps': step,
        'read_interval': read_interval,
        'energy': {
            'initial': round(initial_E, 6),
            'final': round(final_E, 6),
            'max': round(max_E, 6),
            'min': round(min_E, 6),
            'change': round(final_E - initial_E, 6),
            'change_pct': round((final_E - initial_E) / max(initial_E, 1e-10) * 100, 2),
        },
        'survival': {
            'half_life_step': half_life,
            'final_alive_pct': round(n_final_alive / max(n_total_created, 1) * 100, 2),
        },
        'events': dict(event_counts),
        'perturbation_stats': {
            'avg_elimination_prob': round(avg_elim_prob, 4),
            'avg_spawn_prob': round(avg_spawn_prob, 4),
            'avg_interaction_strength': round(avg_interaction, 4),
            'avg_time_factor': round(avg_time_factor, 4),
            'n_perturbations': len(history['perturbation_summary']),
        },
        'container_deformations': sum(1 for e in history['events'] if e.get('type') == 'container_deform'),
    }

    history['summary'] = summary

    print()
    print("=" * 60)
    print(f"SIMULATION COMPLETE — {container.name}")
    print("=" * 60)
    print(f"  Steps: {step}")
    print(f"  Photons created: {n_total_created}")
    print(f"  Photons spawned: {n_spawned}")
    print(f"  Photons alive: {n_final_alive}/{n_total_created}")
    print(f"  Energy: {initial_E:.4f} -> {final_E:.4f} ({summary['energy']['change_pct']:+.1f}%)")
    print(f"  Events: {dict(event_counts)}")
    print(f"  Container deformations: {summary['container_deformations']}")
    if half_life is not None:
        print(f"  Survival half-life: step {half_life}")
    print()

    return history


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_all_experiments():
    """Run evolutionary simulations across multiple containers and configurations."""
    containers = [
        Sphere(radius=1.0),
        Cube(size=2.0),
        Ellipsoid(a=1.5, b=1.0, c=0.8),
        SinaiBilliard(size=2.0, obs_r=0.3),
    ]

    all_results = {}

    print("=" * 70)
    print("EVOLUTIONARY PHOTON BILLIARD SIMULATION")
    print("=" * 70)

    for container in containers:
        print(f"\n{'='*70}")
        print(f"CONTAINER: {container.name}")
        print(f"{'='*70}")

        # Run with 3 photons (base case)
        result = run_evolutionary_simulation(
            container=container,
            n_photons=3,
            max_steps=5000,
            read_interval=50,
            rng_seed=42,
        )
        all_results[container.name] = result

    # ---- Cross-container comparison ----
    print("\n" + "=" * 70)
    print("CROSS-CONTAINER COMPARISON")
    print("=" * 70)

    comparison = []
    for name, result in all_results.items():
        s = result['summary']
        comparison.append({
            'container': name,
            'n_initial': s['n_initial_photons'],
            'n_final_alive': s['n_final_alive'],
            'n_spawned': s['n_spawned'],
            'n_eliminated': s['n_eliminated'],
            'energy_change_pct': s['energy']['change_pct'],
            'events': s['events'],
            'half_life_step': s['survival']['half_life_step'],
            'perturbation_stats': s['perturbation_stats'],
        })
        print(f"  {name:30s} | alive={s['n_final_alive']:3d}/{s['n_total_created']:3d} | "
              f"spawned={s['n_spawned']:3d} | E={s['energy']['change_pct']:+6.1f}% | "
              f"HL={s['survival']['half_life_step']}")

    # ---- Pattern analysis ----
    print("\n" + "=" * 70)
    print("PATTERN ANALYSIS")
    print("=" * 70)

    # 1. Survival patterns: which containers keep photons alive longest?
    print("\n[1] Survival patterns:")
    for c in comparison:
        hl = c['half_life_step']
        if hl is not None:
            print(f"  {c['container']:30s} | half-life at step {hl} | "
                  f"{c['n_final_alive']}/{c['n_initial']} survive")
        else:
            print(f"  {c['container']:30s} | no photons eliminated (all survived)")

    # 2. Energy patterns: which containers increase/decrease total energy?
    print("\n[2] Energy dynamics:")
    for c in comparison:
        print(f"  {c['container']:30s} | {c['energy_change_pct']:+6.1f}% energy change | "
              f"events={c['events']}")

    # 3. Event distribution patterns
    print("\n[3] Event distribution:")
    for c in comparison:
        print(f"  {c['container']:30s} | {c['events']}")

    # 4. Perturbation correlation analysis
    print("\n[4] Perturbation correlations:")
    for name, result in all_results.items():
        ps = result['summary']['perturbation_stats']
        print(f"  {name:30s} | avg_elim={ps['avg_elimination_prob']:.4f} | "
              f"avg_spawn={ps['avg_spawn_prob']:.4f} | "
              f"avg_interact={ps['avg_interaction_strength']:.4f} | "
              f"avg_time={ps['avg_time_factor']:.4f}")

    # ---- Save results ----
    output_path = Path(__file__).parent / "evolutionary_results.json"

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return obj

    json_results = json.loads(json.dumps(all_results, default=convert))
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=convert)

    print(f"\nResults saved to: {output_path}")
    return json_results


if __name__ == "__main__":
    results = run_all_experiments()
