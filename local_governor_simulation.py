#!/usr/bin/env python3
"""
LOCAL GOVERNOR PHOTON BILLIARD SIMULATION v3
==============================================

The governor is at the center. Photons orbit around it.
The key insight: interaction happens DURING movement, not after.
We subdivide each step into micro-steps so photons can pass through
the interaction zone and get affected.

Design:
- Photons spawn at radius 3.0 from center with tangential velocity
- Each "step" = 50 micro-steps
- During micro-steps, photons can enter the interaction zone
- Strong gravitational pull creates orbital dynamics
- Energy absorption, digit extraction, size growth happen during micro-steps
- Container walls only checked at macro-step boundaries
"""

import numpy as np
import json
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import random


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
    def __init__(self, radius=10.0):
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
    def __init__(self, size=20.0):
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
    def __init__(self, a=15.0, b=10.0, c=8.0):
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
    def __init__(self, size=20.0, obs_r=4.0):
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


class Cylinder:
    def __init__(self, radius=10.0, height=20.0):
        self.radius = radius
        self.height = height
        self.half_h = height / 2
        self.name = f"cylinder(r={radius},h={height})"
        self.base_radius = radius
        self.base_height = height

    def get_intersections(self, pos, direction, max_dist=100.0):
        results = []
        r_op = np.array([pos[0], pos[1], 0.0])
        r_dir = np.array([direction[0], direction[1], 0.0])
        a = dot(r_dir, r_dir)
        b = 2.0 * dot(r_op, r_dir)
        c = dot(r_op, r_op) - self.radius ** 2
        disc = b * b - 4 * a * c
        if disc >= 0 and a > 1e-15:
            sqrt_d = math.sqrt(disc)
            for t in [(-b - sqrt_d) / (2 * a), (-b + sqrt_d) / (2 * a)]:
                if t > 1e-10 and t < max_dist:
                    hit_point = pos + t * direction
                    if abs(hit_point[2]) <= self.half_h + 1e-10:
                        normal = np.array([hit_point[0], hit_point[1], 0.0])
                        nlen = np.linalg.norm(normal)
                        if nlen > 1e-15:
                            normal /= nlen
                        results.append((t, hit_point, normal, "cyl-side"))
        if direction[2] > 1e-15:
            t = (self.half_h - pos[2]) / direction[2]
            if t > 1e-10 and t < max_dist:
                hit = pos + t * direction
                if np.hypot(hit[0], hit[1]) <= self.radius + 1e-10:
                    results.append((t, hit, np.array([0, 0, 1]), "cyl-top"))
        if direction[2] < -1e-15:
            t = (-self.half_h - pos[2]) / direction[2]
            if t > 1e-10 and t < max_dist:
                hit = pos + t * direction
                if np.hypot(hit[0], hit[1]) <= self.radius + 1e-10:
                    results.append((t, hit, np.array([0, 0, -1]), "cyl-bottom"))
        return results

    def is_inside(self, point):
        return (np.hypot(point[0], point[1]) < self.radius - 1e-10 and
                abs(point[2]) < self.half_h - 1e-10)

    def volume(self):
        return math.pi * self.radius ** 2 * self.height


# ============================================================
# DIGIT EXTRACTION
# ============================================================

def extract_thousandths_digit(dividend, divisor):
    if abs(divisor) < 1e-15:
        divisor = 1e-15 if divisor >= 0 else -1e-15
    result = dividend / divisor
    abs_result = abs(result)
    thousandths = int(abs_result * 1000) % 10
    return thousandths


# ============================================================
# GOVERNOR
# ============================================================

@dataclass
class LocalGovernor:
    value: int
    position: List[float]
    interaction_radius: float
    proximity_threshold: float
    energy_field: float
    absorption_rate: float
    repulsion_strength: float
    total_emitted: float
    n_proximity_events: int

    def distance_to(self, pos):
        return np.linalg.norm(np.array(pos) - np.array(self.position))

    def proximity_factor(self, dist):
        if dist >= self.interaction_radius:
            return 0.0
        if dist <= self.proximity_threshold:
            return 1.0
        ratio = (self.interaction_radius - dist) / (self.interaction_radius - self.proximity_threshold)
        return ratio ** 2


# ============================================================
# PHOTON
# ============================================================

@dataclass
class LocalPhoton:
    pos: List[float]
    velocity: List[float]
    speed: float
    kinetic_energy: float
    potential_energy: float
    total_energy: float
    momentum: List[float]
    momentum_magnitude: float
    mass: float
    frequency: float
    wavelength: float
    wave_vector: List[float]
    radius: float
    id: int
    birth_step: int
    alive: bool
    collisions: int
    current_digit: int
    digit_history: List[int]
    size_class: int
    perturbations_received: int
    perturbations_applied: int
    total_path_length: float
    size_growth_steps: int
    size_shrink_steps: int
    proximity_events: int
    total_absorbed_energy: float
    last_proximity_dist: float
    orbit_count: int

    def to_dict(self):
        return asdict(self)


def create_local_photon(pid, pos, velocity, governor, rng):
    speed = 5.0 + rng.random() * 10.0
    velocity = normalize(velocity) * speed

    frequency = 0.1 + rng.random() * 0.9
    kinetic_energy = frequency * speed ** 2 / 2
    mass = kinetic_energy / (speed ** 2) if speed > 1e-15 else 0.1
    momentum = velocity * mass
    momentum_mag = np.linalg.norm(momentum)
    wavelength = 1.0 / frequency if frequency > 1e-15 else 1.0
    wave_vector = velocity * (2 * math.pi / wavelength)

    gov_dist = governor.distance_to(pos)
    pe = -governor.energy_field / max(gov_dist, 0.1)

    initial_radius = 0.001 + rng.random() * 0.001

    if gov_dist < governor.proximity_threshold:
        initial_digit = extract_thousandths_digit(gov_dist, governor.value)
    else:
        initial_digit = 0

    return LocalPhoton(
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
        radius=round(initial_radius, 8),
        id=pid,
        birth_step=0,
        alive=True,
        collisions=0,
        current_digit=initial_digit,
        digit_history=[initial_digit],
        size_class=initial_digit,
        perturbations_received=0,
        perturbations_applied=0,
        total_path_length=0.0,
        size_growth_steps=0,
        size_shrink_steps=0,
        proximity_events=0,
        total_absorbed_energy=0.0,
        last_proximity_dist=gov_dist,
        orbit_count=0,
    )


def apply_governor_micro(photon, governor, rng):
    """Apply governor effects during a micro-step."""
    events = []
    dist = governor.distance_to(photon.pos)

    if dist >= governor.interaction_radius:
        return events

    pf = governor.proximity_factor(dist)
    if pf <= 0:
        return events

    photon.proximity_events += 1
    photon.last_proximity_dist = dist

    if dist < governor.proximity_threshold:
        photon.orbit_count += 1

    # DIGIT
    digit = extract_thousandths_digit(dist, governor.value)
    photon.current_digit = digit
    photon.digit_history.append(digit)
    photon.size_class = digit

    # ENERGY
    energy_absorbed = pf * governor.absorption_rate * photon.speed * 0.02
    photon.kinetic_energy += energy_absorbed
    photon.total_absorbed_energy += energy_absorbed
    governor.total_emitted += energy_absorbed

    photon.mass = photon.kinetic_energy / (photon.speed ** 2) if photon.speed > 1e-15 else photon.mass
    photon.momentum = np.array(photon.velocity) * photon.mass
    photon.momentum_magnitude = np.linalg.norm(photon.momentum)
    photon.frequency = photon.kinetic_energy
    photon.wavelength = 1.0 / photon.frequency if photon.frequency > 1e-15 else 1.0
    photon.wave_vector = np.array(photon.velocity) * (2 * math.pi / photon.wavelength)

    # SIZE
    base_size = 0.001 + digit * 0.001
    growth = energy_absorbed * 0.05
    new_radius = base_size + growth + photon.size_growth_steps * 0.0005
    old_radius = photon.radius
    photon.radius = max(0.0005, min(0.2, new_radius))
    if photon.radius > old_radius:
        photon.size_growth_steps += 1
    else:
        photon.size_shrink_steps += 1

    # GRAVITY
    if dist > governor.proximity_threshold:
        direction_to_center = -np.array(photon.pos) / dist
        force = pf * 0.002 / max(dist, 0.1)
        photon.velocity = np.array(photon.velocity) + direction_to_center * force
        photon.velocity = normalize(photon.velocity) * photon.speed
        photon.velocity = [round(v, 8) for v in photon.velocity]

    # REPULSION
    if dist < governor.proximity_threshold * 0.3:
        direction_away = np.array(photon.pos) / dist
        repulsion = governor.repulsion_strength * (1.0 - dist / (governor.proximity_threshold * 0.3))
        photon.velocity = np.array(photon.velocity) + direction_away * repulsion
        photon.velocity = normalize(photon.velocity) * photon.speed
        photon.velocity = [round(v, 8) for v in photon.velocity]
        events.append({'type': 'repulsion', 'photon_id': photon.id, 'distance': round(dist, 6)})

    # ELIMINATION
    if photon.kinetic_energy > 50.0 and rng.random() < 0.01:
        photon.alive = False
        events.append({'type': 'energy_elimination', 'photon_id': photon.id,
                       'kinetic_energy': round(photon.kinetic_energy, 4), 'distance': round(dist, 4)})

    return events


def spawn_photon_from_large(parent, container, governor, rng, photon_list):
    """Spawn a new photon from a large photon."""
    spawn_pos = np.array(parent.pos) + rng.uniform(-0.5, 0.5, 3)
    spawn_dir = rng.uniform(-1, 1, 3)
    spawn_dir = normalize(spawn_dir)
    if not container.is_inside(spawn_pos):
        spawn_pos = np.array(parent.pos)
    new_photon = create_local_photon(
        pid=max((p.id for p in photon_list), default=-1) + 1,
        pos=spawn_pos,
        velocity=spawn_dir,
        governor=governor,
        rng=rng,
    )
    photon_list.append(new_photon)
    parent.perturbations_applied += 1
    return new_photon


# ============================================================
# GLOBALS
# ============================================================
rng = None
all_photons = []


# ============================================================
# SIMULATION
# ============================================================

def run_local_governor_simulation(
    container, governor_value, n_photons=3, max_steps=10000,
    interaction_radius=5.0, proximity_threshold=2.0,
    absorption_rate=1.0, repulsion_strength=1.0, rng_seed=42,
):
    global rng, all_photons
    rng = np.random.RandomState(rng_seed)
    random.seed(rng_seed)
    all_photons = []

    governor = LocalGovernor(
        value=governor_value, position=[0.0, 0.0, 0.0],
        interaction_radius=interaction_radius, proximity_threshold=proximity_threshold,
        energy_field=absorption_rate * 100, absorption_rate=absorption_rate,
        repulsion_strength=repulsion_strength, total_emitted=0.0, n_proximity_events=0,
    )

    # Spawn photons on a ring around center with tangential velocity
    photons = []
    for i in range(n_photons):
        angle = rng.uniform(0, 2*math.pi, 3)
        pos = np.array([
            3.0 * math.sin(angle[1]) * math.cos(angle[0]),
            3.0 * math.sin(angle[1]) * math.sin(angle[0]),
            3.0 * math.cos(angle[1]),
        ])
        radial = pos / np.linalg.norm(pos)
        tangent = np.array([-radial[1], radial[0], rng.uniform(-0.5, 0.5)])
        tangent = normalize(tangent)
        radial_component = rng.uniform(-0.3, 0.3)
        vel = tangent + radial * radial_component
        vel = normalize(vel)
        photons.append(create_local_photon(i, pos, vel, governor, rng))

    history = {
        'container': container.name, 'governor_value': governor_value,
        'n_initial_photons': n_photons, 'max_steps': max_steps,
        'interaction_radius': interaction_radius, 'proximity_threshold': proximity_threshold,
        'events': [], 'photon_states': [], 'energy_timeline': [],
        'survival_timeline': [], 'governor_timeline': [], 'proximity_timeline': [],
    }

    MICRO_STEPS_PER_MACRO = 5
    micro_step = 0
    macro_step = 0

    print(f"Starting local governor simulation:")
    print(f"  Container: {container.name}")
    print(f"  Governor: {governor_value}")
    print(f"  Photons: {n_photons}")
    print(f"  Max steps: {max_steps}")
    print(f"  Interaction radius: {interaction_radius}")
    print(f"  Proximity threshold: {proximity_threshold}")
    print(f"  Micro-steps per macro: {MICRO_STEPS_PER_MACRO}")
    print()

    while macro_step < max_steps:
        # ---- MICRO-STEPS (governor interaction happens here) ----
        for _ in range(MICRO_STEPS_PER_MACRO):
            for photon in photons:
                if not photon.alive:
                    continue

                # Apply governor interaction
                gov_events = apply_governor_micro(photon, governor, rng)
                for e in gov_events:
                    e['step'] = macro_step
                    history['events'].append(e)

                # Move photon
                step_size = 0.2  # micro-step size
                new_pos = np.array(photon.pos) + np.array(photon.velocity) * step_size

                # Check container bounds
                if not container.is_inside(new_pos):
                    # Find closest wall and reflect
                    hit = container.get_intersections(
                        np.array(photon.pos), np.array(photon.velocity), max_dist=step_size
                    )
                    if hit:
                        t, hp, normal, wall = min(hit, key=lambda h: h[0])
                        photon.pos = [round(p, 8) for p in hp]
                        v = reflect(np.array(photon.velocity), normal)
                        v = normalize(v)
                        photon.velocity = [round(vi, 8) for vi in v]
                        photon.pos = [round(p + vi * 1e-8, 8) for p, vi in zip(photon.pos, v)]
                        photon.collisions += 1
                        photon.total_path_length += t
                    else:
                        # Push back inside
                        center_dir = -np.array(new_pos) / np.linalg.norm(new_pos)
                        photon.pos = [round(p, 8) for p in photon.pos + center_dir * 0.1]
                else:
                    photon.pos = [round(p, 8) for p in new_pos]

            micro_step += 1

        macro_step += 1

        # ---- SPAWNING CHECK ----
        for photon in photons:
            if photon.alive and photon.radius > 0.12 and macro_step % 200 == 0:
                spawn_photon_from_large(photon, container, governor, rng, photons)
                history['events'].append({
                    'type': 'size_spawn',
                    'new_photon_id': photons[-1].id,
                    'parent_id': photon.id,
                    'step': macro_step,
                    'parent_radius': round(photon.radius, 4),
                })

        # ---- SNAPSHOTS ----
        alive_photons = [p for p in photons if p.alive]
        total_ke = sum(p.kinetic_energy for p in alive_photons)
        total_pe = sum(p.potential_energy for p in alive_photons)
        total_energy = total_ke + total_pe
        total_absorbed = sum(p.total_absorbed_energy for p in alive_photons)

        if alive_photons:
            distances = [governor.distance_to(p.pos) for p in alive_photons]
        else:
            distances = [0]

        history['energy_timeline'].append({
            'step': macro_step, 'n_alive': len(alive_photons),
            'total_ke': round(total_ke, 6), 'total_pe': round(total_pe, 6),
            'total_energy': round(total_energy, 6),
            'total_absorbed': round(total_absorbed, 6),
            'avg_ke': round(total_ke / max(len(alive_photons), 1), 6),
            'max_ke': round(max((p.kinetic_energy for p in alive_photons), default=0), 6),
            'min_ke': round(min((p.kinetic_energy for p in alive_photons), default=0), 6),
        })

        history['survival_timeline'].append({
            'step': macro_step, 'n_alive': len(alive_photons), 'n_total': len(photons),
        })

        in_interaction = sum(1 for p in alive_photons if governor.distance_to(p.pos) < governor.interaction_radius)
        in_proximity = sum(1 for p in alive_photons if governor.distance_to(p.pos) < governor.proximity_threshold)
        history['proximity_timeline'].append({
            'step': macro_step, 'n_alive': len(alive_photons),
            'in_interaction_zone': in_interaction, 'in_proximity_zone': in_proximity,
            'avg_distance': round(float(np.mean(distances)), 4),
            'min_distance': round(float(min(distances)), 4),
        })

        if macro_step % 500 == 0:
            history['governor_timeline'].append({
                'step': macro_step,
                'total_emitted': round(governor.total_emitted, 4),
                'n_proximity_events': sum(p.proximity_events for p in photons),
            })

        if macro_step % 200 == 0 or macro_step == max_steps:
            for p in photons:
                history['photon_states'].append({'step': macro_step, 'photon': p.to_dict()})

        if macro_step % 1000 == 0:
            print(f"  Step {macro_step:5d} | alive={len(alive_photons):2d}/{len(photons):2d} | "
                  f"E={total_energy:.2f} | abs={total_absorbed:.2f} | "
                  f"avg_dist={np.mean(distances):.2f} | in_zone={in_interaction}")

    # ---- FINAL ----
    history['final_step'] = macro_step
    history['final_photons'] = [p.to_dict() for p in photons]
    history['governor_final'] = {'value': governor.value, 'total_emitted': round(governor.total_emitted, 4)}

    n_total = len(photons)
    n_alive = sum(1 for p in photons if p.alive)
    n_spawned = sum(1 for e in history['events'] if e.get('type') == 'size_spawn')
    n_killed_energy = sum(1 for e in history['events'] if e.get('type') == 'energy_elimination')
    n_repelled = sum(1 for e in history['events'] if e.get('type') == 'repulsion')

    if history['energy_timeline']:
        initial_E = history['energy_timeline'][0]['total_energy']
        final_E = history['energy_timeline'][-1]['total_energy']
        max_E = max(e['total_energy'] for e in history['energy_timeline'])
        total_absorbed_final = history['energy_timeline'][-1]['total_absorbed']
    else:
        initial_E = final_E = max_E = total_absorbed_final = 0

    if history['proximity_timeline']:
        avg_min_dist = np.mean([p['min_distance'] for p in history['proximity_timeline']])
        max_in_proximity = max(p['in_proximity_zone'] for p in history['proximity_timeline'])
    else:
        avg_min_dist = max_in_proximity = 0

    summary = {
        'container': container.name, 'governor_value': governor_value,
        'n_initial_photons': n_photons, 'n_total_created': n_total,
        'n_final_alive': n_alive, 'n_spawned': n_spawned,
        'n_energy_killed': n_killed_energy, 'n_repelled': n_repelled,
        'max_steps': macro_step,
        'energy': {
            'initial': round(initial_E, 6), 'final': round(final_E, 6),
            'max': round(max_E, 6), 'change': round(final_E - initial_E, 6),
            'change_pct': round((final_E - initial_E) / max(abs(initial_E), 1e-10) * 100, 2),
            'total_absorbed': round(total_absorbed_final, 6),
            'governor_emitted': round(governor.total_emitted, 6),
        },
        'proximity': {
            'avg_min_distance': round(avg_min_dist, 4),
            'max_in_proximity_zone': max_in_proximity,
        },
        'events': dict(Counter(e.get('type') for e in history['events'])),
    }

    history['summary'] = summary

    print()
    print("=" * 60)
    print(f"SIMULATION COMPLETE — {container.name} (governor={governor_value})")
    print("=" * 60)
    print(f"  Steps: {macro_step}")
    print(f"  Photons: {n_alive}/{n_total} alive | spawned={n_spawned} | energy-killed={n_killed_energy} | repelled={n_repelled}")
    print(f"  Energy: {initial_E:.2f} -> {final_E:.2f} ({summary['energy']['change_pct']:+.1f}%)")
    print(f"  Total absorbed: {total_absorbed_final:.2f} | Governor emitted: {governor.total_emitted:.2f}")
    print(f"  Avg min distance: {avg_min_dist:.2f} | Max in proximity: {max_in_proximity}")
    print(f"  Events: {dict(Counter(e.get('type') for e in history['events']))}")
    print()

    return history


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_all_experiments():
    containers = [
        Sphere(radius=10.0),
        Cube(size=20.0),
        Ellipsoid(a=15.0, b=10.0, c=8.0),
        SinaiBilliard(size=20.0, obs_r=4.0),
        Cylinder(radius=10.0, height=20.0),
    ]

    governor_values = [7, 13, 42, 100, 333, 999]

    all_results = {}

    print("=" * 70)
    print("LOCAL GOVERNOR PHOTON BILLIARD SIMULATION v3")
    print("Photons spawn in a ring around the central governor.")
    print("Micro-steps allow interaction during movement.")
    print("=" * 70)

    for container in containers:
        print(f"\n{'='*70}")
        print(f"CONTAINER: {container.name}")
        print(f"{'='*70}")

        container_results = {}
        for gv in governor_values:
            print(f"\n  Governor={gv}...", end=" ")
            result = run_local_governor_simulation(
                container=container, governor_value=gv, n_photons=3, max_steps=2000,
                interaction_radius=5.0, proximity_threshold=2.0,
                absorption_rate=1.0, repulsion_strength=1.0, rng_seed=42,
            )
            container_results[gv] = result
            s = result['summary']
            print(f"done | alive={s['n_final_alive']}/{s['n_total_created']} | "
                  f"E={s['energy']['final']:.2f} | abs={s['energy']['total_absorbed']:.2f}")

        all_results[container.name] = container_results

    # ---- COMPARISON ----
    print("\n" + "=" * 70)
    print("CROSS-CONTAINER, CROSS-GOVERNOR COMPARISON")
    print("=" * 70)

    comparison = []
    for cname, cresults in all_results.items():
        for gv, result in cresults.items():
            s = result['summary']
            comparison.append({
                'container': cname, 'governor': gv,
                'n_alive': s['n_final_alive'], 'n_total': s['n_total_created'],
                'n_spawned': s['n_spawned'], 'n_killed': s['n_energy_killed'],
                'total_energy': s['energy']['final'], 'energy_change_pct': s['energy']['change_pct'],
                'total_absorbed': s['energy']['total_absorbed'],
                'governor_emitted': s['energy']['governor_emitted'],
                'avg_min_dist': s['proximity']['avg_min_distance'],
                'max_in_proximity': s['proximity']['max_in_proximity_zone'],
                'events': s['events'],
            })

    print(f"\n{'Container':30s} {'Gov':>5s} {'Alive':>6s} {'Spawn':>6s} {'Kill':>5s} "
          f"{'TotalE':>8s} {'E%':>8s} {'Absorbed':>8s} {'Emit':>8s} {'AvgDist':>8s}")
    print("-" * 105)
    for c in comparison:
        print(f"{c['container']:30s} {c['governor']:5d} {c['n_alive']:6d} {c['n_spawned']:6d} "
              f"{c['n_killed']:5d} {c['total_energy']:8.2f} {c['energy_change_pct']:8.1f} "
              f"{c['total_absorbed']:8.2f} {c['governor_emitted']:8.2f} {c['avg_min_dist']:8.2f}")

    # ---- PATTERN ANALYSIS ----
    print("\n" + "=" * 70)
    print("PATTERN ANALYSIS")
    print("=" * 70)

    print("\n[1] MOST ENERGY PER CONTAINER:")
    for cname in sorted(set(c['container'] for c in comparison)):
        cc = [c for c in comparison if c['container'] == cname]
        best = max(cc, key=lambda x: x['total_energy'])
        print(f"  {cname:30s} | gov={best['governor']} | E={best['total_energy']:.2f} | "
              f"absorbed={best['total_absorbed']:.2f} | {best['n_alive']}/{best['n_total']} alive")

    print("\n[2] MOST ABSORBED ENERGY PER CONTAINER:")
    for cname in sorted(set(c['container'] for c in comparison)):
        cc = [c for c in comparison if c['container'] == cname]
        best = max(cc, key=lambda x: x['total_absorbed'])
        print(f"  {cname:30s} | gov={best['governor']} | absorbed={best['total_absorbed']:.2f} | "
              f"emitted={best['governor_emitted']:.2f} | {best['n_alive']}/{best['n_total']} alive")

    print("\n[3] BEST SURVIVAL PER CONTAINER:")
    for cname in sorted(set(c['container'] for c in comparison)):
        cc = [c for c in comparison if c['container'] == cname]
        best = max(cc, key=lambda x: x['n_alive'] / max(x['n_total'], 1))
        print(f"  {cname:30s} | gov={best['governor']} | {best['n_alive']}/{best['n_total']} ({best['n_alive']/best['n_total']*100:.0f}%) | "
              f"E={best['total_energy']:.2f}")

    print("\n[4] GOVERNOR VALUE ANALYSIS:")
    for gv in governor_values:
        gc = [c for c in comparison if c['governor'] == gv]
        avg_energy = np.mean([c['total_energy'] for c in gc])
        avg_absorbed = np.mean([c['total_absorbed'] for c in gc])
        avg_alive = np.mean([c['n_alive'] for c in gc])
        print(f"  Governor={gv:4d} | avg_E={avg_energy:8.2f} | avg_absorbed={avg_absorbed:8.2f} | "
              f"avg_alive={avg_alive:5.1f}")

    # ---- SAVE ----
    output_path = Path(__file__).parent / "local_governor_results.json"

    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        elif isinstance(obj, (np.floating,)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, np.bool_): return bool(obj)
        return obj

    json_results = json.loads(json.dumps(all_results, default=convert))
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=convert)

    print(f"\nResults saved to: {output_path}")
    return json_results


if __name__ == "__main__":
    results = run_all_experiments()
