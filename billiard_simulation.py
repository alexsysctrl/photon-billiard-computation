#!/usr/bin/env python3
"""
PHOTON BILLIARD COMPUTATION HYPOTHESIS
======================================

Core idea: A photon (point particle) confined to a closed 3D container,
bouncing off walls via specular reflection. The collision pattern encodes
mathematical information based on:
  1. Container geometry
  2. Initial position + velocity
  3. Number and sequence of wall interactions

We simulate multiple container shapes and extract mathematical signatures
from the collision trajectories.

This connects to:
  - Billiard dynamics (dynamical systems)
  - Spectral geometry ("Can you hear the shape of a drum?")
  - Chaotic billiards (Sinai, stadium)
  - Quantum chaos and random matrix theory
  - Periodic orbit theory
"""

import numpy as np
import json
import math
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional, Any
from collections import Counter
import hashlib


# ============================================================
# VECTOR MATH HELPERS
# ============================================================

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector."""
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        return v
    return v / norm


def dot(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b)


def reflect(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Specular reflection: r = d - 2(d.n)n"""
    return direction - 2.0 * dot(direction, normal) * normal


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.linalg.norm(p2 - p1)


# ============================================================
# COLLISION DETECTION
# ============================================================

def closest_wall_hit(
    pos: np.ndarray,
    direction: np.ndarray,
    container,
    max_dist: float = 1000.0,
    epsilon: float = 1e-10
) -> Tuple[float, np.ndarray, np.ndarray, str]:
    """
    Find the closest wall intersection along the direction vector.
    Returns (t, hit_point, normal, wall_label) or None.
    """
    best_t = max_dist
    best_point = None
    best_normal = None
    best_wall = None

    hits = container.get_wall_intersections(pos, direction, max_dist)

    for t, point, normal, wall in hits:
        if t > epsilon and t < best_t:
            best_t = t
            best_point = point.copy()
            best_normal = normal.copy()
            best_wall = wall

    if best_point is not None:
        return (best_t, best_point, best_normal, best_wall)
    return None


# ============================================================
# CONTAINER SHAPES
# ============================================================

class Sphere:
    """Sphere container. Integrable — trajectories are predictable."""

    def __init__(self, radius=1.0):
        self.radius = radius
        self.name = f"sphere(r={radius})"

    def get_wall_intersections(self, pos, direction, max_dist):
        # Ray-sphere intersection
        op = pos  # vector from sphere center to photon
        a = dot(direction, direction)
        b = 2.0 * dot(op, direction)
        c = dot(op, op) - self.radius ** 2

        discriminant = b * b - 4 * a * c
        if discriminant < 0:
            return []

        sqrt_d = math.sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2 * a)
        t2 = (-b + sqrt_d) / (2 * a)

        results = []
        for t in [t1, t2]:
            if t > 1e-10:
                hit_point = pos + t * direction
                normal = normalize(hit_point)  # outward normal at sphere surface
                results.append((t, hit_point, normal, "sphere"))
        return results

    def get_normal_at(self, point):
        return normalize(point)

    def is_inside(self, point):
        return np.linalg.norm(point) < self.radius - 1e-10

    def volume(self):
        return 4.0 / 3.0 * math.pi * self.radius ** 3


class Cube:
    """Cube container. Integrable in 2D, chaotic in 3D with off-axis entry."""

    def __init__(self, size=2.0):
        self.size = size
        self.half = size / 2
        self.name = f"cube(s={size})"

    def get_wall_intersections(self, pos, direction, max_dist):
        results = []
        # Each face: defined by axis and sign (+/-)
        faces = [
            (0, 1, self.half, "x+"),
            (0, -1, -self.half, "x-"),
            (1, 1, self.half, "y+"),
            (1, -1, -self.half, "y-"),
            (2, 1, self.half, "z+"),
            (2, -1, -self.half, "z-"),
        ]

        for axis, sign, coord, label in faces:
            if abs(direction[axis]) < 1e-15:
                continue
            t = (coord - pos[axis]) / direction[axis]
            if t > 1e-10 and t < max_dist:
                hit_point = pos + t * direction
                # Check if hit point is on the face (within bounds)
                other_axes = [i for i in range(3) if i != axis]
                if all(-self.half <= hit_point[i] <= self.half for i in other_axes):
                    normal = np.array([0.0, 0.0, 0.0])
                    normal[axis] = sign
                    results.append((t, hit_point, normal, label))

        return results

    def is_inside(self, point):
        return all(-self.half < p < self.half for p in point)

    def volume(self):
        return self.size ** 3


class Cylinder:
    """Cylinder container. Mix of integrable and chaotic behavior."""

    def __init__(self, radius=1.0, height=2.0):
        self.radius = radius
        self.height = height
        self.half_h = height / 2
        self.name = f"cylinder(r={radius},h={height})"

    def get_wall_intersections(self, pos, direction, max_dist):
        results = []

        # Side wall (cylindrical surface)
        # x^2 + y^2 = r^2
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
                        # Normal is radial in xy plane
                        normal = np.array([hit_point[0], hit_point[1], 0.0])
                        nlen = np.linalg.norm(normal)
                        if nlen > 1e-15:
                            normal /= nlen
                        else:
                            normal = np.array([hit_point[0], hit_point[1], 0.0])
                        results.append((t, hit_point, normal, "cyl-side"))

        # Top cap
        if direction[2] > 1e-15:
            t = (self.half_h - pos[2]) / direction[2]
            if t > 1e-10 and t < max_dist:
                hit = pos + t * direction
                if np.hypot(hit[0], hit[1]) <= self.radius + 1e-10:
                    results.append((t, hit, np.array([0, 0, 1]), "cyl-top"))

        # Bottom cap
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


class Torus:
    """Torus container (inner tube). Chaotic — multiply connected domain."""

    def __init__(self, R=1.5, r=0.6):
        self.R = R  # major radius
        self.r = r  # minor radius
        self.name = f"torus(R={R},r={r})"

    def get_wall_intersections(self, pos, direction, max_dist):
        # Numerical ray-torus intersection
        results = []
        n_steps = 200
        dt = max_dist / n_steps

        prev_dist = self._dist_from_surface(pos)
        for i in range(1, n_steps + 1):
            t = i * dt
            pt = pos + t * direction
            curr_dist = self._dist_from_surface(pt)

            # Sign change means we crossed the surface
            if prev_dist * curr_dist < 0:
                # Bisection to find exact crossing
                t_lo = (i - 1) * dt
                t_hi = t
                for _ in range(20):
                    t_mid = (t_lo + t_hi) / 2
                    pt_mid = pos + t_mid * direction
                    d_mid = self._dist_from_surface(pt_mid)
                    if d_mid * prev_dist < 0:
                        t_hi = t_mid
                    else:
                        t_lo = t_mid
                        prev_dist = d_mid

                hit_point = pos + t_hi * direction
                normal = self._surface_normal(hit_point)
                results.append((t_hi, hit_point, normal, "torus-surface"))
                prev_dist = self._dist_from_surface(pos + t_hi * direction + direction * 0.001)
                continue

            prev_dist = curr_dist

        return results

    def _dist_from_surface(self, point):
        """Signed distance from torus surface. Positive = inside."""
        q = np.hypot(point[0], point[2]) - self.R
        return self.r - np.hypot(q, point[1])

    def _surface_normal(self, point):
        """Outward normal at torus surface point."""
        q = np.hypot(point[0], point[2])
        if q < 1e-15:
            return np.array([0, 1, 0])
        nx = point[0] / q
        nz = point[2] / q
        n = np.array([nx * self.R, point[1], nz * self.R])
        nlen = np.linalg.norm(n)
        if nlen > 1e-15:
            n /= nlen
        return -n  # inward (we're inside)

    def is_inside(self, point):
        return self._dist_from_surface(point) > 1e-10

    def volume(self):
        return math.pi * self.r ** 2 * 2 * math.pi * self.R


class Ellipsoid:
    """Ellipsoid container. Generally chaotic unless spheroidal."""

    def __init__(self, a=1.5, b=1.0, c=0.8):
        self.a = a
        self.b = b
        self.c = c
        self.name = f"ellipsoid(a={a},b={b},c={c})"

    def get_wall_intersections(self, pos, direction, max_dist):
        # Ray-ellipsoid intersection: (x/a)^2 + (y/b)^2 + (z/c)^2 = 1
        ax = direction[0] / self.a
        ay = direction[1] / self.b
        az = direction[2] / self.c
        ox = pos[0] / self.a
        oy = pos[1] / self.b
        oz = pos[2] / self.c

        a_q = ax * ax + ay * ay + az * az
        b_q = 2 * (ox * ax + oy * ay + oz * az)
        c_q = ox * ox + oy * oy + oz * oz - 1

        disc = b_q * b_q - 4 * a_q * c_q
        if disc < 0:
            return []

        sqrt_d = math.sqrt(disc)
        results = []
        for t in [(-b_q - sqrt_d) / (2 * a_q), (-b_q + sqrt_d) / (2 * a_q)]:
            if t > 1e-10 and t < max_dist:
                hit = pos + t * direction
                # Normal: gradient of ellipsoid equation
                normal = np.array([
                    hit[0] / self.a ** 2,
                    hit[1] / self.b ** 2,
                    hit[2] / self.c ** 2
                ])
                nlen = np.linalg.norm(normal)
                if nlen > 1e-15:
                    normal /= nlen
                else:
                    normal = normalize(hit)
                results.append((t, hit, normal, "ellipsoid"))
        return results

    def is_inside(self, point):
        return (point[0] / self.a) ** 2 + (point[1] / self.b) ** 2 + (point[2] / self.c) ** 2 < 1.0 - 1e-10

    def volume(self):
        return 4.0 / 3.0 * math.pi * self.a * self.b * self.c


class SinaiBilliard:
    """Sinai billiard: square with circular obstacle in center. Chaotic."""

    def __init__(self, size=2.0, obstacle_radius=0.4):
        self.size = size
        self.half = size / 2
        self.obstacle_radius = obstacle_radius
        self.name = f"sinai(s={size},obs={obstacle_radius})"

    def get_wall_intersections(self, pos, direction, max_dist):
        results = []

        # Wall intersections (cube faces)
        faces = [
            (0, 1, self.half, "x+"),
            (0, -1, -self.half, "x-"),
            (1, 1, self.half, "y+"),
            (1, -1, -self.half, "y-"),
            (2, 1, self.half, "z+"),
            (2, -1, -self.half, "z-"),
        ]

        for axis, sign, coord, label in faces:
            if abs(direction[axis]) < 1e-15:
                continue
            t = (coord - pos[axis]) / direction[axis]
            if t > 1e-10 and t < max_dist:
                hit = pos + t * direction
                other = [i for i in range(3) if i != axis]
                if all(-self.half <= hit[i] <= self.half for i in other):
                    # Check if blocked by obstacle
                    obs_dist = np.hypot(hit[0], hit[1])
                    if obs_dist > self.obstacle_radius + 1e-10:
                        normal = np.array([0, 0, 0])
                        normal[axis] = sign
                        results.append((t, hit, normal, label))

        # Obstacle intersection (cylinder along z)
        r_op = np.array([pos[0], pos[1], 0])
        r_dir = np.array([direction[0], direction[1], 0])
        a = dot(r_dir, r_dir)
        b = 2 * dot(r_op, r_dir)
        c = dot(r_op, r_op) - self.obstacle_radius ** 2
        disc = b * b - 4 * a * c

        if disc >= 0 and a > 1e-15:
            sqrt_d = math.sqrt(disc)
            for t in [(-b - sqrt_d) / (2 * a), (-b + sqrt_d) / (2 * a)]:
                if t > 1e-10 and t < max_dist:
                    hit = pos + t * direction
                    if abs(hit[2]) < self.half - 1e-10:
                        normal = np.array([hit[0], hit[1], 0])
                        nlen = np.linalg.norm(normal)
                        if nlen > 1e-15:
                            normal /= nlen
                        results.append((t, hit, normal, "sinai-obs"))

        return results

    def is_inside(self, point):
        if not all(-self.half < p < self.half for p in point):
            return False
        return np.hypot(point[0], point[1]) > self.obstacle_radius + 1e-10

    def volume(self):
        return self.size ** 2 * self.size - math.pi * self.obstacle_radius ** 2 * self.size


class StadiumBilliard:
    """Stadium billiard: 2D approximation in x-y plane. Chaotic."""

    def __init__(self, length=1.5, radius=0.75):
        self.length = length
        self.radius = radius
        self.name = f"stadium(L={length},r={radius})"

    def get_wall_intersections(self, pos, direction, max_dist):
        results = []
        # Flat walls at z = +/- half_h (we work in 2D x-y, z is constant)
        half_h = self.length / 2

        # Check z boundaries (top/bottom in our 2D view)
        if abs(direction[2]) > 1e-15:
            for coord, label in [(half_h, "stadium-z+"), (-half_h, "stadium-z-")]:
                t = (coord - pos[2]) / direction[2]
                if t > 1e-10 and t < max_dist:
                    hit = pos + t * direction
                    # Check if in flat region or semicircle
                    if abs(hit[0]) <= self.length - self.radius:
                        normal = np.array([0, 0, 1 if label.endswith('+') else -1])
                        results.append((t, hit, normal, label))

        # Semicircular ends (centered at +/- (length-radius) in x, y=0)
        for cx_sign in [-1, 1]:
            cx = cx_sign * (self.length - self.radius)
            cy = 0.0
            # Circle: (x-cx)^2 + y^2 = r^2
            op = np.array([pos[0] - cx, pos[1] - cy, 0])
            dir_xy = np.array([direction[0], direction[1], 0])
            a = dot(dir_xy, dir_xy)
            b = 2 * dot(op, dir_xy)
            c = dot(op, op) - self.radius ** 2
            disc = b * b - 4 * a * c
            if disc >= 0 and a > 1e-15:
                sqrt_d = math.sqrt(disc)
                for t in [(-b - sqrt_d) / (2 * a), (-b + sqrt_d) / (2 * a)]:
                    if t > 1e-10 and t < max_dist:
                        hit = pos + t * direction
                        # Verify on correct semicircle
                        if (cx_sign == -1 and hit[0] < cx) or (cx_sign == 1 and hit[0] > cx):
                            normal = normalize(np.array([hit[0] - cx, hit[1] - cy, 0]))
                            results.append((t, hit, normal, f"stadium-semi-{cx_sign}"))

        return results

    def is_inside(self, point):
        if abs(point[2]) > self.length / 2 - 1e-10:
            return False
        for cx_sign in [-1, 1]:
            cx = cx_sign * (self.length - self.radius)
            d = np.hypot(point[0] - cx, point[1])
            if d < self.radius - 1e-10:
                return True
        return False

    def volume(self):
        # Area of stadium * unit depth
        return (2 * self.length * self.radius + math.pi * self.radius ** 2) * 1.0


# ============================================================
# PHOTON TRAJECTORY SIMULATION
# ============================================================

@dataclass
class CollisionEvent:
    collision_number: int
    position: List[float]
    direction_before: List[float]
    direction_after: List[float]
    normal: List[float]
    wall: str
    time_of_flight: float
    energy: float  # always 1.0 for photon (no speed change)


@dataclass
class Trajectory:
    container_name: str
    initial_pos: List[float]
    initial_dir: List[float]
    collisions: List[CollisionEvent]
    total_bounces: int
    total_path_length: float
    max_time_of_flight: float
    min_time_of_flight: float
    mean_time_of_flight: float
    unique_walls_hit: int
    wall_distribution: Dict[str, int]


def simulate_photon(
    container,
    initial_pos: np.ndarray,
    initial_dir: np.ndarray,
    max_collisions: int = 5000,
    max_step: float = 100.0,
    epsilon: float = 1e-10
) -> Trajectory:
    """
    Simulate a photon bouncing inside a container.

    Args:
        container: Container shape object
        initial_pos: Starting position (must be inside container)
        initial_dir: Starting direction (will be normalized)
        max_collisions: Maximum number of wall bounces
        max_step: Maximum distance per ray cast
        epsilon: Collision detection tolerance

    Returns:
        Trajectory with all collision events and statistics
    """
    pos = initial_pos.copy()
    direction = normalize(initial_dir).copy()
    collisions = []

    for i in range(max_collisions):
        hit = closest_wall_hit(pos, direction, container, max_step, epsilon)
        if hit is None:
            break

        t, hit_point, normal, wall_label = hit

        # Record collision
        dir_before = direction.copy()
        direction = reflect(direction, normal)
        direction = normalize(direction)

        # Small push to prevent sticking
        pos = hit_point + direction * 1e-8

        collisions.append(CollisionEvent(
            collision_number=i,
            position=[round(p, 8) for p in hit_point],
            direction_before=[round(d, 8) for d in dir_before],
            direction_after=[round(d, 8) for d in direction],
            normal=[round(n, 8) for n in normal],
            wall=wall_label,
            time_of_flight=round(t, 8),
            energy=1.0
        ))

    # Compute statistics
    times_of_flight = [c.time_of_flight for c in collisions]
    wall_counter = Counter(c.wall for c in collisions)

    return Trajectory(
        container_name=container.name,
        initial_pos=[round(p, 8) for p in initial_pos],
        initial_dir=[round(d, 8) for d in normalize(initial_dir)],
        collisions=collisions,
        total_bounces=len(collisions),
        total_path_length=sum(times_of_flight),
        max_time_of_flight=max(times_of_flight) if times_of_flight else 0,
        min_time_of_flight=min(times_of_flight) if times_of_flight else 0,
        mean_time_of_flight=np.mean(times_of_flight) if times_of_flight else 0,
        unique_walls_hit=len(wall_counter),
        wall_distribution=dict(wall_counter)
    )


# ============================================================
# MATHEMATICAL SIGNATURE EXTRACTION
# ============================================================

def extract_signatures(traj: Trajectory) -> Dict[str, Any]:
    """
    Extract mathematical signatures from a photon trajectory.

    These signatures are what we use as the "computation result."
    """
    if not traj.collisions:
        return {"error": "no collisions"}

    collisions = traj.collisions
    n = len(collisions)

    # 1. Wall distribution entropy (Shannon entropy of wall hits)
    wall_counts = np.array(list(traj.wall_distribution.values()), dtype=float)
    wall_probs = wall_counts / wall_counts.sum()
    wall_probs = wall_probs[wall_probs > 0]
    entropy = -np.sum(wall_probs * np.log2(wall_probs + 1e-30))

    # 2. Time-of-flight statistics
    tofs = np.array([c.time_of_flight for c in collisions])

    # 3. Time-of-flight distribution shape
    if len(tofs) > 10:
        tof_mean = np.mean(tofs)
        tof_std = np.std(tofs)
        tof_skew = float(np.mean(((tofs - tof_mean) / tof_std) ** 3)) if tof_std > 1e-15 else 0
        tof_kurt = float(np.mean(((tofs - tof_mean) / tof_std) ** 4)) - 3.0 if tof_std > 1e-15 else 0
    else:
        tof_mean = tof_std = tof_skew = tof_kurt = 0

    # 4. Collision sequence hash (wall labels as a string)
    wall_seq = "".join(c.wall[0] for c in collisions)  # first char of each wall label
    seq_hash = hashlib.md5(wall_seq.encode()).hexdigest()[:16]

    # 5. Direction change angles (how much the photon turns each bounce)
    angles = []
    for i in range(1, min(n, 500)):  # sample first 500
        d1 = np.array(collisions[i-1].direction_after)
        d2 = np.array(collisions[i].direction_after)
        cos_angle = np.clip(np.dot(d1, d2), -1, 1)
        angles.append(math.degrees(math.acos(cos_angle)))

    if angles:
        mean_angle_change = np.mean(angles)
        std_angle_change = np.std(angles)
    else:
        mean_angle_change = std_angle_change = 0

    # 6. Recurrence statistics (how often does the photon return near previous positions)
    positions = np.array([c.position for c in collisions[:min(n, 1000)]])
    if len(positions) > 10:
        # Compute pairwise distances (sampled)
        sampled_idx = np.linspace(0, len(positions) - 1, min(200, len(positions)), dtype=int)
        sampled = positions[sampled_idx]
        dists = []
        for i in range(len(sampled)):
            for j in range(i + 1, len(sampled)):
                dists.append(np.linalg.norm(sampled[i] - sampled[j]))
        if dists:
            recurrence_mean = np.mean(dists)
            recurrence_std = np.std(dists)
        else:
            recurrence_mean = recurrence_std = 0
    else:
        recurrence_mean = recurrence_std = 0

    # 7. Poincaré section signature (direction components at collision)
    # Sample direction components across collisions
    if len(collisions) > 10:
        sample_size = min(200, n)
        step = max(1, n // sample_size)
        sampled_dirs = [collisions[i].direction_after for i in range(0, n, step)]
        dx_components = [d[0] for d in sampled_dirs]
        dy_components = [d[1] for d in sampled_dirs]
        dz_components = [d[2] for d in sampled_dirs]

        poincare_features = {
            "dx_mean": float(np.mean(dx_components)),
            "dx_std": float(np.std(dx_components)),
            "dy_mean": float(np.mean(dy_components)),
            "dy_std": float(np.std(dy_components)),
            "dz_mean": float(np.mean(dz_components)),
            "dz_std": float(np.std(dz_components)),
        }
    else:
        poincare_features = {}

    # 8. Path length per collision (growth rate)
    if n > 1:
        cumulative_lengths = np.cumsum(tofs)
        path_growth = cumulative_lengths[-1] / n  # avg path per collision
    else:
        path_growth = 0

    # 9. Wall hit Markov chain transition matrix (simplified)
    unique_walls = sorted(set(traj.wall_distribution.keys()))
    wall_to_idx = {w: i for i, w in enumerate(unique_walls)}
    n_walls = len(unique_walls)

    if n_walls > 0 and n > 1:
        transitions = np.zeros((n_walls, n_walls))
        for i in range(1, min(n, 1000)):
            w1 = wall_to_idx.get(collisions[i-1].wall)
            w2 = wall_to_idx.get(collisions[i].wall)
            if w1 is not None and w2 is not None:
                transitions[w1][w2] += 1

        # Normalize to get transition probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_probs = transitions / row_sums

        # Entropy of transition matrix (averaged over rows)
        trans_entropy = 0
        for row in transition_probs:
            rp = row[row > 0]
            trans_entropy -= np.sum(rp * np.log2(rp + 1e-30))
        trans_entropy /= max(n_walls, 1)
    else:
        transitions = np.array([])
        trans_entropy = 0

    # 10. Lyapunov-like sensitivity indicator
    # Simulate with perturbed initial condition and compare divergence
    # (computed separately in compare_trajectories)

    return {
        "container": traj.container_name,
        "initial_pos": traj.initial_pos,
        "initial_dir": traj.initial_dir,
        "total_bounces": n,
        "total_path_length": round(float(traj.total_path_length), 6),
        "wall_entropy": round(float(entropy), 6),
        "tof_mean": round(float(tof_mean), 6),
        "tof_std": round(float(tof_std), 6),
        "tof_skewness": round(tof_skew, 6),
        "tof_kurtosis": round(tof_kurt, 6),
        "mean_angle_change": round(float(mean_angle_change), 6),
        "std_angle_change": round(float(std_angle_change), 6),
        "recurrence_mean": round(float(recurrence_mean), 6),
        "recurrence_std": round(float(recurrence_std), 6),
        "path_per_collision": round(float(path_growth), 6),
        "transition_entropy": round(float(trans_entropy), 6),
        "sequence_hash": seq_hash,
        "unique_walls": n_walls,
        "wall_distribution": traj.wall_distribution,
        "poincare": poincare_features,
        "transition_matrix_shape": list(transitions.shape) if len(transitions.shape) > 0 else [],
    }


def compare_trajectories(
    traj1: Trajectory,
    traj2: Trajectory,
    max_compare: int = 1000
) -> Dict[str, Any]:
    """
    Compare two trajectories for sensitivity to initial conditions
    (Lyapunov exponent estimation).
    """
    if len(traj1.collisions) < 3 or len(traj2.collisions) < 3:
        return {"error": "not enough collisions", "traj1_bounces": len(traj1.collisions), "traj2_bounces": len(traj2.collisions)}

    collisions1 = traj1.collisions[:max_compare]
    collisions2 = traj2.collisions[:max_compare]

    # Track position divergence over collision number
    divergences = []
    for i in range(min(len(collisions1), len(collisions2))):
        d = np.linalg.norm(
            np.array(collisions1[i].position) -
            np.array(collisions2[i].position)
        )
        divergences.append(d)

    divergences = np.array(divergences)

    # Estimate Lyapunov exponent: avg log divergence rate
    if len(divergences) > 1:
        log_divs = np.log(divergences + 1e-30)
        lyapunov = float(np.mean(np.diff(log_divs)))
    else:
        lyapunov = 0

    # Final divergence
    final_div = divergences[-1] if len(divergences) > 0 else 0

    # Kullback-Leibler divergence of wall distributions
    dist1 = np.array(list(traj1.wall_distribution.values()), dtype=float)
    dist2 = np.array(list(traj2.wall_distribution.values()), dtype=float)
    dist1 = dist1 / (dist1.sum() + 1e-30)
    dist2 = dist2 / (dist2.sum() + 1e-30)
    kl_div = float(np.sum(dist1 * np.log((dist1 + 1e-30) / (dist2 + 1e-30))))

    return {
        "final_divergence": round(float(final_div), 8),
        "lyapunov_exponent": round(lyapunov, 8),
        "kl_divergence": round(kl_div, 8),
        "mean_divergence": round(float(np.mean(divergences)), 8),
        "max_divergence": round(float(np.max(divergences)), 8),
    }


# ============================================================
# EXPERIMENT RUNNER
# ============================================================

def run_experiments():
    """
    Run a comprehensive set of photon billiard experiments.

    Tests:
    1. Different container shapes with same initial conditions
    2. Same container with perturbed initial conditions (sensitivity)
    3. Multiple random initial conditions per container (statistical signatures)
    4. Periodic orbit detection
    """

    containers = [
        Sphere(radius=1.0),
        Cube(size=2.0),
        Cylinder(radius=1.0, height=2.0),
        Ellipsoid(a=1.5, b=1.0, c=0.8),
        SinaiBilliard(size=2.0, obstacle_radius=0.3),
    ]

    # Standard initial conditions
    standard_pos = np.array([0.0, 0.0, 0.0])
    standard_dir = np.array([1.0, 0.3, 0.0])  # Keep in xy plane for 2D-compatible shapes

    # Perturbed initial conditions (for sensitivity test)
    perturbations = [
        np.array([1e-6, 0, 0]),
        np.array([0, 1e-6, 0]),
        np.array([0, 0, 1e-6]),
        np.array([1e-6, 1e-6, 1e-6]),
    ]

    results = {
        "experiments": [],
        "shape_comparison": [],
        "sensitivity_analysis": [],
        "statistical_signatures": [],
        "signature_clusters": [],
    }

    print("=" * 70)
    print("PHOTON BILLIARD COMPUTATION HYPOTHESIS — EXPERIMENTS")
    print("=" * 70)

    # ---- EXPERIMENT 1: Shape comparison with same initial conditions ----
    print("\n[1] Shape comparison — same initial conditions, different containers")
    print("-" * 70)

    shape_results = []
    for container in containers:
        test_pos = standard_pos.copy()
        if not container.is_inside(test_pos):
            test_pos = np.array([0.2, 0.0, 0.0])
        if not container.is_inside(test_pos):
            test_pos = np.array([0.5, 0.0, 0.0])
        if not container.is_inside(test_pos):
            test_pos = np.array([0.0, 0.5, 0.0])
        if not container.is_inside(test_pos):
            test_pos = np.array([0.3, 0.3, 0.3])
        standard_pos = test_pos

        traj = simulate_photon(container, standard_pos, standard_dir, max_collisions=2000)
        sig = extract_signatures(traj)
        sig["initial_pos"] = [round(p, 8) for p in standard_pos]
        sig["initial_dir"] = [round(d, 8) for d in standard_dir]
        shape_results.append(sig)
        print(f"  {container.name:30s} | bounces={traj.total_bounces:5d} | "
              f"entropy={sig['wall_entropy']:.4f} | path={sig['total_path_length']:.2f}")

    results["shape_comparison"] = shape_results

    # ---- EXPERIMENT 2: Sensitivity analysis (Lyapunov exponent) ----
    print("\n[2] Sensitivity analysis — perturbed initial conditions")
    print("-" * 70)

    sensitivity_results = []
    for container in containers:
        test_pos = standard_pos.copy()
        if not container.is_inside(test_pos):
            test_pos = np.array([0.2, 0.0, 0.0])
        if not container.is_inside(test_pos):
            test_pos = np.array([0.5, 0.0, 0.0])
        if not container.is_inside(test_pos):
            test_pos = np.array([0.0, 0.5, 0.0])
        standard_pos = test_pos

        traj_main = simulate_photon(container, standard_pos, standard_dir, max_collisions=2000)

        for pert in perturbations:
            perturbed_pos = standard_pos + pert
            if not container.is_inside(perturbed_pos):
                perturbed_pos = standard_pos + np.array([1e-8, 0, 0])

            traj_pert = simulate_photon(container, perturbed_pos, standard_dir, max_collisions=2000)
            comparison = compare_trajectories(traj_main, traj_pert)
            comparison["container"] = container.name
            comparison["perturbation"] = [round(p, 10) for p in pert]
            sensitivity_results.append(comparison)
            if "lyapunov_exponent" in comparison:
                print(f"  {container.name:30s} | pert={pert} | "
                      f"Lyapunov={comparison['lyapunov_exponent']:+.6f} | "
                      f"final_div={comparison['final_divergence']:.4f}")
            else:
                print(f"  {container.name:30s} | pert={pert} | INSUFFICIENT COLLISIONS")

    results["sensitivity_analysis"] = sensitivity_results

    # ---- EXPERIMENT 3: Statistical signatures (many random ICs) ----
    print("\n[3] Statistical signatures — 50 random initial conditions per shape")
    print("-" * 70)

    statistical_results = []
    rng = np.random.RandomState(42)

    for container in containers:
        signatures = []
        for i in range(50):
            # Random position inside container (approximate)
            while True:
                rand_pos = rng.uniform(-0.8, 0.8, 3)
                if container.is_inside(rand_pos):
                    break

            # Random direction
            rand_dir = rng.uniform(-1, 1, 3)
            rand_dir = normalize(rand_dir)

            traj = simulate_photon(container, rand_pos, rand_dir, max_collisions=1000)
            if traj.total_bounces > 50:  # only keep meaningful trajectories
                sig = extract_signatures(traj)
                signatures.append(sig)

        if signatures:
            # Aggregate statistics
            entropies = [s["wall_entropy"] for s in signatures]
            tof_means = [s["tof_mean"] for s in signatures]
            lyapunovs = []

            # Compute average Lyapunov for this container
            traj_ref = simulate_photon(container, np.array([0.0, 0.0, 0.0]),
                                       np.array([1, 0.3, 0.5]), max_collisions=500)
            for sig in signatures[:10]:
                traj_test = simulate_photon(container,
                    np.array(sig["initial_pos"]) + np.array([1e-6, 0, 0]),
                    np.array(sig["initial_dir"]), max_collisions=500)
                comp = compare_trajectories(traj_ref, traj_test)
                if "lyapunov_exponent" in comp:
                    lyapunovs.append(comp["lyapunov_exponent"])

            cluster = {
                "container": container.name,
                "n_trajectories": len(signatures),
                "entropy_mean": round(float(np.mean(entropies)), 6),
                "entropy_std": round(float(np.std(entropies)), 6),
                "tof_mean_mean": round(float(np.mean(tof_means)), 6),
                "tof_mean_std": round(float(np.std(tof_means)), 6),
                "lyapunov_mean": round(float(np.mean(lyapunovs)), 6) if lyapunovs else 0,
                "signature_vectors": [
                    {
                        "entropy": s["wall_entropy"],
                        "tof_skew": s["tof_skewness"],
                        "tof_kurt": s["tof_kurtosis"],
                        "transition_entropy": s["transition_entropy"],
                        "mean_angle": s["mean_angle_change"],
                    }
                    for s in signatures[:20]  # sample
                ],
            }
            statistical_results.append(cluster)
            print(f"  {container.name:30s} | n={len(signatures):3d} | "
                  f"H_mean={cluster['entropy_mean']:.4f} | "
                  f"Lyap={cluster['lyapunov_mean']:+.6f}")

    results["statistical_signatures"] = statistical_results

    # ---- EXPERIMENT 4: Signature space clustering ----
    print("\n[4] Signature space — embedding all trajectories")
    print("-" * 70)

    # Build signature vectors for all trajectories
    all_sig_vectors = []
    all_labels = []

    for cluster in statistical_results:
        for i, sig in enumerate(cluster["signature_vectors"]):
            vec = [
                sig["entropy"],
                sig["tof_skew"],
                sig["tof_kurt"],
                sig["transition_entropy"],
                sig["mean_angle"],
            ]
            all_sig_vectors.append(vec)
            all_labels.append(cluster["container"])

    if len(all_sig_vectors) > 1:
        sig_matrix = np.array(all_sig_vectors)

        # Normalize
        sig_mean = sig_matrix.mean(axis=0)
        sig_std = sig_matrix.std(axis=0) + 1e-10
        sig_normalized = (sig_matrix - sig_mean) / sig_std

        # PCA (simple: just compute covariance and eigenvalues)
        cov = np.cov(sig_normalized.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Project to 2D
        proj_2d = sig_normalized @ eigenvectors[:, :2]

        # Cluster by container (compute centroids)
        container_names = sorted(set(all_labels))
        centroids = {}
        for name in container_names:
            mask = [i for i, l in enumerate(all_labels) if l == name]
            centroid = np.mean(proj_2d[mask], axis=0)
            centroids[name] = {
                "x": round(float(centroid[0]), 4),
                "y": round(float(centroid[1]), 4),
                "spread": round(float(np.std(proj_2d[mask], axis=0).mean()), 4),
                "count": len(mask),
            }

        cluster_result = {
            "total_trajectories": len(all_sig_vectors),
            "signature_dimensions": 5,
            "pca_variance_ratio": [round(float(v / cov.trace()), 4) for v in eigenvalues[:5]],
            "eigenvalues": [round(float(v), 6) for v in eigenvalues[:5]],
            "container_centroids": centroids,
        }
        results["signature_clusters"] = cluster_result

        print(f"  Total trajectory signatures: {len(all_sig_vectors)}")
        print(f"  PCA variance explained: {[round(float(v/cov.trace()), 4) for v in eigenvalues[:3]]}")
        print(f"  Container centroids in signature space:")
        for name, c in centroids.items():
            print(f"    {name:30s} -> ({c['x']:+.4f}, {c['y']:+.4f}) spread={c['spread']:.4f} n={c['count']}")

    # ---- SUMMARY ----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Find most distinctive containers
    print("\nContainer distinction (by entropy separation):")
    for cluster in statistical_results:
        print(f"  {cluster['container']:30s} | H={cluster['entropy_mean']:.4f}+/-{cluster['entropy_std']:.4f} | "
              f"Lyap={cluster['lyapunov_mean']:+.6f}")

    # Save results
    output_path = Path(__file__).parent / "billiard_results.json"
    # Convert numpy types for JSON serialization
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

    json_results = json.loads(json.dumps(results, default=convert))
    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=convert)

    print(f"\nResults saved to: {output_path}")
    return json_results


if __name__ == "__main__":
    results = run_experiments()
