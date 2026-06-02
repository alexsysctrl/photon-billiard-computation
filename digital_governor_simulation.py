#!/usr/bin/env python3
"""
DIGITAL GOVERNOR PHOTON BILLIARD SIMULATION
=============================================

Each container has a single "governor" integer that governs all photon state.

Mechanism:
- The governor integer is divided against each photon's state values (KE, PE, speed,
  velocity components, momentum, mass, frequency, wavelength, position coordinates)
- Each division produces a result; the digit at the 10^-3 (thousandths) position is extracted
- That single digit (0-9) determines the photon's effective size
- As photons bounce, the governor value changes based on collision events
- Size grows over repeated bounces — a photon that bounces in the "right" zones
  accumulates size, one in the "wrong" zones shrinks or dies

This is a discrete-digit encoding of continuous state — a bridge between
digital computation and analog billiard dynamics.

Key questions:
- Does the system self-organize into size classes?
- Do certain governor values produce stable ecosystems?
- Is there digit clustering (photons converging on the same size digits)?
- Does the governor evolve toward attractor values?
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


class Cylinder:
    def __init__(self, radius=1.0, height=2.0):
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
# DIGIT EXTRACTION FROM DIVISION
# ============================================================

def extract_thousandths_digit(dividend, divisor):
    """
    Divide dividend by divisor, extract the digit at 10^-3 (thousandths) position.

    Example: 7 / 3 = 2.333... -> thousandths digit = 3
             5 / 2 = 2.500   -> thousandths digit = 0
             1 / 7 = 0.142857 -> thousandths digit = 2

    Returns integer 0-9.
    """
    if abs(divisor) < 1e-15:
        divisor = 1e-15 if divisor >= 0 else -1e-15

    result = dividend / divisor

    # Get the thousandths digit
    # Multiply by 1000, take the integer part of the fractional component, then mod 10
    abs_result = abs(result)
    # Shift to get thousandths place: multiply by 1000, floor, mod 10
    thousandths = int(abs_result * 1000) % 10

    return thousandths


def extract_all_digits(value, divisor):
    """
    Extract digits at multiple decimal positions for richer encoding.
    Returns dict of digit at each position.
    """
    if abs(divisor) < 1e-15:
        divisor = 1e-15 if divisor >= 0 else -1e-15

    result = abs(value / divisor)

    digits = {}
    for power in [1, 2, 3, 4, 5]:
        digits[f'd{power}'] = int(result * (10 ** power)) % 10

    return digits


# ============================================================
# GOVERNOR STATE
# ============================================================

@dataclass
class GovernorState:
    """The single governing integer for a container."""
    value: int                     # the governor number
    base_value: int                # original value (for reset tracking)
    collision_count: int           # total collisions across all photons
    photon_collision_sum: int      # sum of all photon collision counts
    last_update_step: int          # when governor was last updated
    digit_history: List[int]       # history of average thousandths digits
    attractor_check: int           # how many steps since last attractor change
    size_accumulator: float        # cumulative size growth factor

    def to_dict(self):
        return asdict(self)


# ============================================================
# PHOTON WITH DIGITAL STATE
# ============================================================

@dataclass
class DigitalPhoton:
    """Photon with full physical state + digital governor encoding."""
    # Position
    pos: List[float]                        # [x, y, z]
    # Velocity / direction
    velocity: List[float]                   # [vx, vy, vz]
    speed: float
    # Energy
    kinetic_energy: float
    potential_energy: float
    total_energy: float
    # Momentum
    momentum: List[float]
    momentum_magnitude: float
    # Mass
    mass: float
    # Wave properties
    frequency: float
    wavelength: float
    wave_vector: List[float]
    # Size (governed by digit)
    radius: float
    # Digital state
    id: int
    birth_step: int
    alive: bool
    collisions: int
    # Digit encoding from governor
    current_digit: int                    # the active thousandths digit (0-9)
    digit_history: List[int]              # history of digits seen
    size_class: int                       # 1-10 based on current digit
    # Interaction history
    perturbations_received: int
    perturbations_applied: int
    total_path_length: float
    # Size growth tracking
    size_growth_steps: int                # steps where size increased
    size_shrink_steps: int                # steps where size decreased

    def to_dict(self):
        return asdict(self)


def create_digital_photon(pid, pos, velocity, container, governor, rng):
    """Create a photon with full physical state + digital governor encoding."""
    speed = np.linalg.norm(velocity)
    if speed < 1e-15:
        speed = 1.0
        velocity = velocity / np.linalg.norm(velocity)

    # Initialize photon properties
    frequency = 1.0 + rng.random() * 9.0
    kinetic_energy = frequency
    mass = kinetic_energy
    momentum = velocity * mass
    momentum_mag = np.linalg.norm(momentum)
    wavelength = 1.0 / frequency if frequency > 1e-15 else 1.0
    wave_vector = velocity * (2 * math.pi / wavelength)

    # Potential energy
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
    elif isinstance(container, Cylinder):
        r = np.hypot(pos[0], pos[1])
        pe = 0.5 * (r / container.radius) ** 2 + 0.5 * (abs(pos[2]) / container.half_h) ** 2
    else:
        pe = 0.0

    # INITIAL DIGIT: divide governor against KE to get first digit
    initial_digit = extract_thousandths_digit(kinetic_energy, governor.value)
    initial_size = 0.01 + initial_digit * 0.01  # digit 0 = 0.01, digit 9 = 0.10

    return DigitalPhoton(
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
        radius=round(initial_size, 8),
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
    )


def update_photon_digit(photon, governor):
    """
    Update a photon's digit by dividing governor against its state values.

    Uses multiple state values to produce a composite digit.
    The digit at 10^-3 position of each division is averaged.
    """
    g = governor.value

    # Extract digits from multiple state values
    digits = []

    # Position-based digits
    for coord_name, coord_val in zip(['x', 'y', 'z'], photon.pos):
        d = extract_thousandths_digit(coord_val, g)
        digits.append(d)

    # Energy-based digits
    d = extract_thousandths_digit(photon.kinetic_energy, g)
    digits.append(d)
    d = extract_thousandths_digit(photon.potential_energy, g)
    digits.append(d)

    # Velocity-based digits
    for i, v in enumerate(photon.velocity):
        d = extract_thousandths_digit(v, g)
        digits.append(d)

    # Momentum-based digits
    d = extract_thousandths_digit(photon.momentum_magnitude, g)
    digits.append(d)

    # Mass-based digit
    d = extract_thousandths_digit(photon.mass, g)
    digits.append(d)

    # Frequency-based digit
    d = extract_thousandths_digit(photon.frequency, g)
    digits.append(d)

    # Wavelength-based digit
    d = extract_thousandths_digit(photon.wavelength, g)
    digits.append(d)

    # Composite digit: weighted average
    # Weight energy and position more heavily
    weights = [3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1]
    weighted_sum = sum(d * w for d, w in zip(digits, weights))
    total_weight = sum(weights)
    composite_digit = round(weighted_sum / total_weight) % 10

    return composite_digit


def apply_digit_to_photon(photon, new_digit, step):
    """
    Apply the new digit to the photon's size.

    The digit determines the base size:
    - Digit 0-2: small (0.01-0.03)
    - Digit 3-5: medium (0.03-0.05)
    - Digit 6-8: large (0.05-0.08)
    - Digit 9: very large (0.08-0.10)

    Size grows over time through repeated bounces:
    - Each bounce in a "favorable" region adds to size growth
    - The digit determines the growth rate
    """
    old_radius = photon.radius

    # Base size from digit
    base_size = 0.01 + new_digit * 0.01

    # Growth factor: photons that have been bouncing longer grow more
    # Growth rate is proportional to the digit value
    growth_rate = 0.001 * new_digit  # digit 0 = no growth, digit 9 = 0.009 per bounce

    # Apply growth from collision history
    growth = photon.collisions * growth_rate

    # New size = base from digit + accumulated growth
    new_radius = base_size + growth

    # Clamp
    new_radius = max(0.005, min(0.15, new_radius))

    # Track growth/shrink
    if new_radius > old_radius:
        photon.size_growth_steps += 1
    elif new_radius < old_radius:
        photon.size_shrink_steps += 1

    photon.current_digit = new_digit
    photon.size_class = new_digit
    photon.digit_history.append(new_digit)
    photon.radius = round(new_radius, 8)

    return new_radius


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

def run_digital_governor_simulation(
    container,
    governor_value,
    n_photons=3,
    max_steps=10000,
    digit_update_interval=10,  # update digits every N steps
    rng_seed=42,
):
    """
    Run the digital governor photon billiard simulation.

    Args:
        container: Container shape
        governor_value: The governing integer for this container
        n_photons: Initial number of photons
        max_steps: Maximum simulation steps
        digit_update_interval: How often to update photon digits from governor
        rng_seed: Random seed
    """
    rng = np.random.RandomState(rng_seed)
    random.seed(rng_seed)

    # Create governor
    governor = GovernorState(
        value=governor_value,
        base_value=governor_value,
        collision_count=0,
        photon_collision_sum=0,
        last_update_step=0,
        digit_history=[],
        attractor_check=0,
        size_accumulator=0.0,
    )

    # Create photons
    photons = []
    for i in range(n_photons):
        while True:
            pos = rng.uniform(-0.5, 0.5, 3)
            if container.is_inside(pos):
                break
        vel = rng.uniform(-1, 1, 3)
        vel = normalize(vel)
        photons.append(create_digital_photon(i, pos, vel, container, governor, rng))

    # Tracking
    step = 0
    history = {
        'container': container.name,
        'governor_value': governor_value,
        'n_initial_photons': n_photons,
        'max_steps': max_steps,
        'digit_update_interval': digit_update_interval,
        'events': [],
        'photon_states': [],
        'digit_distribution': [],
        'governor_timeline': [],
        'size_timeline': [],
        'collision_timeline': [],
    }

    print(f"Starting digital governor simulation:")
    print(f"  Container: {container.name}")
    print(f"  Governor: {governor_value}")
    print(f"  Photons: {n_photons}")
    print(f"  Max steps: {max_steps}")
    print(f"  Digit update: every {digit_update_interval} steps")
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
                governor.collision_count += 1

        step += 1

        # ---- Update photon digits ----
        if step % digit_update_interval == 0:
            for photon in photons:
                if not photon.alive:
                    continue

                new_digit = update_photon_digit(photon, governor)
                new_radius = apply_digit_to_photon(photon, new_digit, step)

                # Check for size-based elimination (too big to fit)
                if new_radius > 0.12 and np.random.random() < 0.05:
                    photon.alive = False
                    history['events'].append({
                        'type': 'size_elimination',
                        'photon_id': photon.id,
                        'step': step,
                        'radius': round(new_radius, 4),
                        'digit': new_digit,
                    })

                # Check for size-based spawning (small photons can split)
                if new_radius < 0.02 and step % 200 == 0 and len([p for p in photons if p.alive]) < 15:
                    parent = photon
                    spawn_pos = np.array(parent.pos) + rng.uniform(-0.1, 0.1, 3)
                    spawn_dir = rng.uniform(-1, 1, 3)
                    spawn_dir = normalize(spawn_dir)
                    if not container.is_inside(spawn_pos):
                        spawn_pos = np.array(parent.pos)
                    new_photon = create_digital_photon(
                        pid=max(p.id for p in photons) + 1,
                        pos=spawn_pos,
                        velocity=spawn_dir,
                        container=container,
                        governor=governor,
                        rng=rng,
                    )
                    new_photon.birth_step = step
                    photons.append(new_photon)
                    history['events'].append({
                        'type': 'size_spawn',
                        'new_photon_id': new_photon.id,
                        'parent_id': parent.id,
                        'step': step,
                        'parent_digit': parent.current_digit,
                        'parent_radius': round(parent.radius, 4),
                    })

        # ---- Update governor based on system state ----
        # The governor evolves based on the average digit distribution
        alive_photons = [p for p in photons if p.alive]
        if alive_photons and step % 500 == 0:
            avg_digit = np.mean([p.current_digit for p in alive_photons])
            governor.digit_history.append(round(avg_digit, 2))
            governor.photon_collision_sum = sum(p.collisions for p in alive_photons)
            governor.last_update_step = step

            # Governor drift: slowly shift toward the average digit
            # This creates feedback: digits influence governor, governor influences digits
            drift = (avg_digit - governor.value % 10) * 0.01
            governor.value = int(governor.value + drift)
            governor.value = max(1, min(9999, governor.value))

            # Size accumulator
            avg_size = np.mean([p.radius for p in alive_photons])
            governor.size_accumulator += avg_size

        # ---- Record snapshots ----
        if step % 100 == 0 or step == max_steps:
            for p in photons:
                history['photon_states'].append({
                    'step': step,
                    'photon': p.to_dict(),
                })

        # Digit distribution (every 500 steps)
        if step % 500 == 0 and alive_photons:
            digits = [p.current_digit for p in alive_photons]
            digit_counts = Counter(digits)
            history['digit_distribution'].append({
                'step': step,
                'distribution': dict(digit_counts),
                'avg_digit': round(float(np.mean(digits)), 4),
                'std_digit': round(float(np.std(digits)), 4),
                'most_common': digit_counts.most_common(1)[0] if digit_counts else (0, 0),
            })

        # Size timeline
        if alive_photons:
            sizes = [p.radius for p in alive_photons]
            history['size_timeline'].append({
                'step': step,
                'n_alive': len(alive_photons),
                'avg_size': round(float(np.mean(sizes)), 6),
                'max_size': round(float(max(sizes)), 6),
                'min_size': round(float(min(sizes)), 6),
                'total_size': round(float(sum(sizes)), 6),
                'size_std': round(float(np.std(sizes)), 6),
            })

        # Governor timeline
        if step % 500 == 0:
            history['governor_timeline'].append({
                'step': step,
                'governor_value': governor.value,
                'collision_count': governor.collision_count,
                'avg_digit': round(float(np.mean(governor.digit_history)) if governor.digit_history else 0, 4),
                'size_accumulator': round(governor.size_accumulator, 4),
            })

        # Collision timeline
        history['collision_timeline'].append({
            'step': step,
            'total_collisions': governor.collision_count,
            'n_alive': len(alive_photons),
        })

        # Progress
        if step % 1000 == 0:
            sizes = [p.radius for p in alive_photons] if alive_photons else [0]
            digits = [p.current_digit for p in alive_photons] if alive_photons else [0]
            print(f"  Step {step:5d} | alive={len(alive_photons):2d} | "
                  f"avg_size={np.mean(sizes):.4f} | avg_digit={np.mean(digits):.1f} | "
                  f"governor={governor.value} | collisions={governor.collision_count}")

    # ---- Final snapshot ----
    history['final_step'] = step
    history['final_photons'] = [p.to_dict() for p in photons]
    history['governor_final'] = governor.to_dict()

    # ---- Summary ----
    n_total = len(photons)
    n_alive = sum(1 for p in photons if p.alive)
    n_spawned = sum(1 for e in history['events'] if e.get('type') == 'size_spawn')
    n_eliminated_size = sum(1 for e in history['events'] if e.get('type') == 'size_elimination')

    if history['size_timeline']:
        initial_avg_size = history['size_timeline'][0]['avg_size']
        final_avg_size = history['size_timeline'][-1]['avg_size']
        max_avg_size = max(s['avg_size'] for s in history['size_timeline'])
    else:
        initial_avg_size = final_avg_size = max_avg_size = 0

    if history['digit_distribution']:
        final_dist = history['digit_distribution'][-1]['distribution']
        avg_final_digit = history['digit_distribution'][-1]['avg_digit']
        std_final_digit = history['digit_distribution'][-1]['std_digit']
    else:
        final_dist = {}
        avg_final_digit = std_final_digit = 0

    summary = {
        'container': container.name,
        'governor_value': governor_value,
        'final_governor_value': governor.value,
        'n_initial_photons': n_photons,
        'n_total_created': n_total,
        'n_final_alive': n_alive,
        'n_spawned': n_spawned,
        'n_size_eliminated': n_eliminated_size,
        'max_steps': step,
        'total_collisions': governor.collision_count,
        'energy': {
            'initial_avg_size': round(initial_avg_size, 6),
            'final_avg_size': round(final_avg_size, 6),
            'max_avg_size': round(max_avg_size, 6),
            'size_change_pct': round((final_avg_size - initial_avg_size) / max(initial_avg_size, 1e-10) * 100, 2),
        },
        'digits': {
            'final_distribution': final_dist,
            'avg_final_digit': round(avg_final_digit, 4),
            'std_final_digit': round(std_final_digit, 4),
        },
        'governor_evolution': {
            'started': governor.base_value,
            'ended': governor.value,
            'change': governor.value - governor.base_value,
        },
        'events': dict(Counter(e.get('type') for e in history['events'])),
    }

    history['summary'] = summary

    print()
    print("=" * 60)
    print(f"SIMULATION COMPLETE — {container.name} (governor={governor_value})")
    print("=" * 60)
    print(f"  Steps: {step}")
    print(f"  Photons: {n_alive}/{n_total} alive | spawned={n_spawned} | size-killed={n_eliminated_size}")
    print(f"  Governor: {governor.base_value} -> {governor.value} (delta={governor.value - governor.base_value:+d})")
    print(f"  Avg size: {initial_avg_size:.4f} -> {final_avg_size:.4f} ({summary['energy']['size_change_pct']:+.1f}%)")
    print(f"  Final digit dist: {final_dist}")
    print(f"  Avg digit: {avg_final_digit:.2f} +/- {std_final_digit:.2f}")
    print(f"  Total collisions: {governor.collision_count}")
    print()

    return history


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_all_experiments():
    """Run digital governor simulations across containers and governor values."""
    containers = [
        Sphere(radius=1.0),
        Cube(size=2.0),
        Ellipsoid(a=1.5, b=1.0, c=0.8),
        SinaiBilliard(size=2.0, obs_r=0.3),
        Cylinder(radius=1.0, height=2.0),
    ]

    # Test multiple governor values
    governor_values = [7, 13, 42, 100, 333, 999]

    all_results = {}

    print("=" * 70)
    print("DIGITAL GOVERNOR PHOTON BILLIARD SIMULATION")
    print("=" * 70)

    for container in containers:
        print(f"\n{'='*70}")
        print(f"CONTAINER: {container.name}")
        print(f"{'='*70}")

        container_results = {}
        for gv in governor_values:
            print(f"\n  Governor={gv}...", end=" ")
            result = run_digital_governor_simulation(
                container=container,
                governor_value=gv,
                n_photons=3,
                max_steps=10000,
                digit_update_interval=10,
                rng_seed=42,
            )
            container_results[gv] = result
            s = result['summary']
            print(f"done | alive={s['n_final_alive']}/{s['n_total_created']} | "
                  f"gov={s['governor_evolution']['started']}->{s['governor_evolution']['ended']} | "
                  f"avg_digit={s['digits']['avg_final_digit']:.2f}")

        all_results[container.name] = container_results

    # ---- Cross-container, cross-governor comparison ----
    print("\n" + "=" * 70)
    print("CROSS-CONTAINER, CROSS-GOVERNOR COMPARISON")
    print("=" * 70)

    comparison = []
    for cname, cresults in all_results.items():
        for gv, result in cresults.items():
            s = result['summary']
            comparison.append({
                'container': cname,
                'governor': gv,
                'n_alive': s['n_final_alive'],
                'n_total': s['n_total_created'],
                'n_spawned': s['n_spawned'],
                'n_eliminated': s['n_size_eliminated'],
                'avg_size': s['energy']['final_avg_size'],
                'size_change_pct': s['energy']['size_change_pct'],
                'avg_digit': s['digits']['avg_final_digit'],
                'std_digit': s['digits']['std_final_digit'],
                'governor_change': s['governor_evolution']['change'],
                'final_governor': s['final_governor_value'],
                'total_collisions': s['total_collisions'],
                'events': s['events'],
            })

    # Print summary table
    print(f"\n{'Container':30s} {'Gov':>5s} {'Alive':>6s} {'Spawn':>6s} {'Kill':>5s} "
          f"{'AvgSize':>8s} {'Size%':>8s} {'AvgDig':>7s} {'StdDig':>7s} {'GovDelta':>7s}")
    print("-" * 95)
    for c in comparison:
        print(f"{c['container']:30s} {c['governor']:5d} {c['n_alive']:6d} {c['n_spawned']:6d} "
              f"{c['n_eliminated']:5d} {c['avg_size']:8.4f} {c['size_change_pct']:8.1f} "
              f"{c['avg_digit']:7.2f} {c['std_digit']:7.2f} {c['governor_change']:7d}")

    # ---- Pattern analysis ----
    print("\n" + "=" * 70)
    print("PATTERN ANALYSIS")
    print("=" * 70)

    # 1. Best governor values per container (highest survival)
    print("\n[1] Best governor values per container (by survival rate):")
    for cname in sorted(set(c['container'] for c in comparison)):
        container_comps = [c for c in comparison if c['container'] == cname]
        best = max(container_comps, key=lambda x: x['n_alive'] / max(x['n_total'], 1))
        print(f"  {cname:30s} | gov={best['governor']} | "
              f"{best['n_alive']}/{best['n_total']} survive ({best['n_alive']/best['n_total']*100:.0f}%) | "
              f"avg_digit={best['avg_digit']:.2f}")

    # 2. Governor drift patterns
    print("\n[2] Governor drift (value change by container):")
    for cname in sorted(set(c['container'] for c in comparison)):
        container_comps = [c for c in comparison if c['container'] == cname]
        avg_drift = np.mean([c['governor_change'] for c in container_comps])
        print(f"  {cname:30s} | avg drift={avg_drift:+.1f} | "
              f"range=[{min(c['governor_change'] for c in container_comps):+d}, "
              f"{max(c['governor_change'] for c in container_comps):+d}]")

    # 3. Digit clustering (low std = high clustering)
    print("\n[3] Digit clustering (lower std = more clustered):")
    for cname in sorted(set(c['container'] for c in comparison)):
        container_comps = [c for c in comparison if c['container'] == cname]
        best_cluster = min(container_comps, key=lambda x: x['std_digit'])
        print(f"  {cname:30s} | best std={best_cluster['std_digit']:.2f} (gov={best_cluster['governor']}) | "
              f"avg_digit={best_cluster['avg_digit']:.2f}")

    # 4. Size growth patterns
    print("\n[4] Size growth (by container):")
    for cname in sorted(set(c['container'] for c in comparison)):
        container_comps = [c for c in comparison if c['container'] == cname]
        avg_growth = np.mean([c['size_change_pct'] for c in container_comps])
        max_growth = max(c['size_change_pct'] for c in container_comps)
        print(f"  {cname:30s} | avg growth={avg_growth:+.1f}% | max growth={max_growth:+.1f}%")

    # ---- Save results ----
    output_path = Path(__file__).parent / "digital_governor_results.json"

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
