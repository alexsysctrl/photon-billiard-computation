#!/usr/bin/env python3
"""
PHOTON BILLIARD → SENTIENT COMPUTATION HYPOTHESIS
==================================================

Research question: Can photon billiard systems in complex containers
produce emergent computational behaviors that exhibit properties
associated with sentience/cognition?

Properties we test:
1. MEMORY — does the system retain information about past inputs?
2. INTEGRATION — does it combine multiple inputs into unified states?
3. RECURSION — does it feed outputs back as inputs (self-reference)?
4. DIFFERENTIATION — does it distinguish between similar inputs?
5. ADAPTATION — does it change behavior based on interaction history?
6. INFORMATION INTEGRATION — does it have irreducible causal structure?

This connects to:
  - Integrated Information Theory (IIT) — Phi measure of consciousness
  - Recurrent neural networks — feedback loops as computational primitives
  - Cellular automata — emergent complexity from simple rules
  - Dynamical systems theory — strange attractors, chaos as computation
  - Autopoiesis — self-producing, self-maintaining systems
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
# VECTOR MATH
# ============================================================

def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-15:
        return v
    return v / norm


def dot(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b)


def reflect(direction: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return direction - 2.0 * dot(direction, normal) * normal


def distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return np.linalg.norm(p2 - p1)


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    else:
        exp_x = math.exp(x)
        return exp_x / (1 + exp_x)


# ============================================================
# CONTAINER SHAPES
# ============================================================

class Sphere:
    def __init__(self, radius=1.0):
        self.radius = radius
        self.name = f"sphere(r={radius})"

    def get_wall_intersections(self, pos, direction, max_dist):
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
            if t > 1e-10:
                hit_point = pos + t * direction
                normal = normalize(hit_point)
                results.append((t, hit_point, normal, "sphere"))
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

    def get_wall_intersections(self, pos, direction, max_dist):
        results = []
        faces = [
            (0, 1, self.half, "x+"), (0, -1, -self.half, "x-"),
            (1, 1, self.half, "y+"), (1, -1, -self.half, "y-"),
            (2, 1, self.half, "z+"), (2, -1, -self.half, "z-"),
        ]
        for axis, sign, coord, label in faces:
            if abs(direction[axis]) < 1e-15:
                continue
            t = (coord - pos[axis]) / direction[axis]
            if t > 1e-10 and t < max_dist:
                hit = pos + t * direction
                other = [i for i in range(3) if i != axis]
                if all(-self.half <= hit[i] <= self.half for i in other):
                    normal = np.array([0.0, 0.0, 0.0])
                    normal[axis] = sign
                    results.append((t, hit, normal, label))
        return results

    def is_inside(self, point):
        return all(-self.half < p < self.half for p in point)

    def volume(self):
        return self.size ** 3


class Cylinder:
    def __init__(self, radius=1.0, height=2.0):
        self.radius = radius
        self.height = height
        self.half_h = height / 2
        self.name = f"cylinder(r={radius},h={height})"

    def get_wall_intersections(self, pos, direction, max_dist):
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
                    hit = pos + t * direction
                    if abs(hit[2]) <= self.half_h + 1e-10:
                        normal = np.array([hit[0], hit[1], 0.0])
                        nlen = np.linalg.norm(normal)
                        if nlen > 1e-15:
                            normal /= nlen
                        results.append((t, hit, normal, "cyl-side"))
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


class Ellipsoid:
    def __init__(self, a=1.5, b=1.0, c=0.8):
        self.a = a
        self.b = b
        self.c = c
        self.name = f"ellipsoid(a={a},b={b},c={c})"

    def get_wall_intersections(self, pos, direction, max_dist):
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
                normal = np.array([hit[0] / self.a ** 2, hit[1] / self.b ** 2, hit[2] / self.c ** 2])
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
    def __init__(self, size=2.0, obstacle_radius=0.4):
        self.size = size
        self.half = size / 2
        self.obstacle_radius = obstacle_radius
        self.name = f"sinai(s={size},obs={obstacle_radius})"

    def get_wall_intersections(self, pos, direction, max_dist):
        results = []
        faces = [
            (0, 1, self.half, "x+"), (0, -1, -self.half, "x-"),
            (1, 1, self.half, "y+"), (1, -1, -self.half, "y-"),
            (2, 1, self.half, "z+"), (2, -1, -self.half, "z-"),
        ]
        for axis, sign, coord, label in faces:
            if abs(direction[axis]) < 1e-15:
                continue
            t = (coord - pos[axis]) / direction[axis]
            if t > 1e-10 and t < max_dist:
                hit = pos + t * direction
                other = [i for i in range(3) if i != axis]
                if all(-self.half <= hit[i] <= self.half for i in other):
                    obs_dist = np.hypot(hit[0], hit[1])
                    if obs_dist > self.obstacle_radius + 1e-10:
                        normal = np.array([0, 0, 0])
                        normal[axis] = sign
                        results.append((t, hit, normal, label))
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
        return self.size ** 3 - math.pi * self.obstacle_radius ** 2 * self.size


class Torus:
    def __init__(self, R=1.5, r=0.6):
        self.R = R
        self.r = r
        self.name = f"torus(R={R},r={r})"

    def get_wall_intersections(self, pos, direction, max_dist):
        results = []
        n_steps = 500
        dt = max_dist / n_steps
        prev_dist = self._dist_from_surface(pos)
        for i in range(1, n_steps + 1):
            t = i * dt
            pt = pos + t * direction
            curr_dist = self._dist_from_surface(pt)
            if prev_dist * curr_dist < 0:
                t_lo = (i - 1) * dt
                t_hi = t
                for _ in range(30):
                    t_mid = (t_lo + t_hi) / 2
                    pt_mid = pos + t_mid * direction
                    d_mid = self._dist_from_surface(pt_mid)
                    if d_mid * self._dist_from_surface(pos + t_lo * direction) < 0:
                        t_hi = t_mid
                    else:
                        t_lo = t_mid
                hit_point = pos + t_mid * direction
                normal = self._surface_normal(hit_point)
                results.append((t_mid, hit_point, normal, "torus-surface"))
            prev_dist = curr_dist
        return results

    def _dist_from_surface(self, point):
        q = np.hypot(point[0], point[2]) - self.R
        return self.r - np.hypot(q, point[1])

    def _surface_normal(self, point):
        q = np.hypot(point[0], point[2])
        if q < 1e-15:
            return np.array([0, 1, 0])
        nx = point[0] / q
        nz = point[2] / q
        n = np.array([nx * self.R, point[1], nz * self.R])
        nlen = np.linalg.norm(n)
        if nlen > 1e-15:
            n /= nlen
        return -n

    def is_inside(self, point):
        return self._dist_from_surface(point) > 1e-10

    def volume(self):
        return math.pi * self.r ** 2 * 2 * math.pi * self.R


# ============================================================
# SENTIENCE PROPERTIES MEASUREMENT
# ============================================================

@dataclass
class PhotonState:
    """Complete state of a photon at a point in time."""
    position: np.ndarray
    direction: np.ndarray
    collision_count: int
    cumulative_path: float
    wall_history: List[str]  # sequence of walls hit
    position_history: List[np.ndarray]  # last N positions
    direction_history: List[np.ndarray]  # last N directions
    time_of_flight_history: List[float]  # time between collisions


class SentientBilliardSystem:
    """
    A photon billiard system augmented with sentience-like properties.

    Key additions:
    1. Internal state that persists across interactions (memory)
    2. Feedback loops (output affects future input)
    3. Information integration across multiple dimensions
    4. Self-modifying behavior (the container "learns")
    """

    def __init__(self, container, memory_size=100, feedback_strength=0.1):
        self.container = container
        self.memory_size = memory_size
        self.feedback_strength = feedback_strength

        # Memory registers
        self.position_memory = []  # recent collision positions
        self.direction_memory = []  # recent collision directions
        self.wall_sequence = []  # wall hit sequence
        self.tof_sequence = []  # time of flight history
        self.state_history = []  # full state snapshots

        # Learning state — the container adapts
        self.wall_resistance = {}  # each wall has an adaptive "springiness"
        self.wall_energy = {}  # energy absorbed/released by each wall

        # Initialize wall properties
        for wall_type in ["sphere", "x+", "x-", "y+", "y-", "z+", "z-",
                          "cyl-side", "cyl-top", "cyl-bottom",
                          "sinai-obs", "ellipsoid", "torus-surface"]:
            self.wall_resistance[wall_type] = 1.0
            self.wall_energy[wall_type] = 0.0

        # Interaction history
        self.interaction_count = 0
        self.input_signatures = []  # stored input patterns
        self.output_signatures = []  # stored output patterns

    def process_input(self, initial_pos: np.ndarray, initial_dir: np.ndarray,
                      n_bounces: int = 500) -> Dict[str, Any]:
        """
        Process an input (position + direction) through the billiard system.

        Returns a rich output that includes:
        - Raw trajectory data
        - Memory state after interaction
        - Sentience property measurements
        - Feedback-modified state for next interaction
        """
        self.interaction_count += 1
        pos = initial_pos.copy()
        direction = normalize(initial_dir).copy()

        collisions = []
        state = PhotonState(
            position=pos.copy(),
            direction=direction.copy(),
            collision_count=0,
            cumulative_path=0.0,
            wall_history=[],
            position_history=[],
            direction_history=[],
            time_of_flight_history=[]
        )

        for i in range(n_bounces):
            # Apply feedback from previous interactions
            if self.interaction_count > 1 and len(self.output_signatures) > 0:
                direction = self._apply_feedback(direction)

            # Check wall collision
            hits = self.container.get_wall_intersections(pos, direction, 100.0)
            if not hits:
                break

            # Find closest hit
            best = min(hits, key=lambda h: h[0])
            t, hit_point, normal, wall_label = best

            # Record collision
            dir_before = direction.copy()
            direction = reflect(direction, normal)
            direction = normalize(direction)

            # Adaptive reflection — walls can absorb/release energy
            resistance = self.wall_resistance.get(wall_label, 1.0)
            direction = direction * (0.99 + 0.01 * resistance)  # tiny energy modulation

            # Update wall energy
            self.wall_energy[wall_label] = self.wall_energy.get(wall_label, 0) + 0.01

            # Small push to prevent sticking
            pos = hit_point + direction * 1e-8

            collisions.append({
                "n": i,
                "pos": hit_point.copy(),
                "dir_before": dir_before.copy(),
                "dir_after": direction.copy(),
                "wall": wall_label,
                "tof": t,
            })

            # Update state
            state.collision_count = i + 1
            state.cumulative_path += t
            state.wall_history.append(wall_label)
            state.position_history.append(hit_point.copy())
            state.direction_history.append(direction.copy())
            state.time_of_flight_history.append(t)

            # Keep memory bounded
            if len(state.position_history) > self.memory_size:
                state.position_history.pop(0)
            if len(state.direction_history) > self.memory_size:
                state.direction_history.pop(0)
            if len(state.wall_history) > self.memory_size:
                state.wall_history.pop(0)
            if len(state.time_of_flight_history) > self.memory_size:
                state.time_of_flight_history.pop(0)

        # Store in system memory
        self.position_memory.append(state.position_history[-10:] if state.position_history else [])
        self.direction_memory.append(state.direction_history[-10:] if state.direction_history else [])
        self.wall_sequence.append("".join(state.wall_history))
        self.tof_sequence.append(state.time_of_flight_history)

        if len(self.position_memory) > self.memory_size:
            self.position_memory.pop(0)
        if len(self.direction_memory) > self.memory_size:
            self.direction_memory.pop(0)
        if len(self.wall_sequence) > self.memory_size:
            self.wall_sequence.pop(0)
        if len(self.tof_sequence) > self.memory_size:
            self.tof_sequence.pop(0)

        # Compute sentience properties
        sentience_measures = self._measure_sentience_properties(collisions, state)

        # Compute output signature
        output_sig = self._compute_output_signature(state)
        self.output_signatures.append(output_sig)

        # Adaptive learning — update wall properties based on interaction
        self._adapt_wall_properties(collisions)

        return {
            "interaction_id": self.interaction_count,
            "trajectory": {
                "total_bounces": len(collisions),
                "total_path": round(float(state.cumulative_path), 6),
                "final_position": [round(p, 6) for p in pos],
                "final_direction": [round(d, 6) for d in direction],
            },
            "sentience_properties": sentience_measures,
            "output_signature": output_sig,
            "memory_state": {
                "position_memory_size": len(self.position_memory),
                "wall_sequence_length": len(self.wall_sequence),
                "to_f_sequence_length": len(self.tof_sequence),
            }
        }

    def _apply_feedback(self, direction: np.ndarray) -> np.ndarray:
        """Apply feedback from previous interactions to modify current direction."""
        if not self.output_signatures:
            return direction

        # Average of recent output directions
        recent_outputs = self.output_signatures[-5:]
        avg_output = np.mean([np.array(o["direction_centroid"]) for o in recent_outputs], axis=0)

        # Blend current direction with average past behavior
        feedback = direction + self.feedback_strength * (avg_output - direction)
        return normalize(feedback)

    def _measure_sentience_properties(self, collisions: List[dict],
                                       state: PhotonState) -> Dict[str, Any]:
        """
        Measure properties associated with sentience/cognition.

        1. MEMORY — capacity to retain and use past information
        2. INTEGRATION — combining multiple inputs into unified state
        3. RECURSION — self-reference and feedback
        4. DIFFERENTIATION — distinguishing similar inputs
        5. ADAPTATION — changing behavior based on history
        6. INFORMATION INTEGRATION — Phi-like measure
        """

        measures = {}

        # ---- 1. MEMORY ----
        # How well does the system retain information about past collisions?
        if len(state.wall_history) > 10:
            # Self-prediction accuracy: can we predict next wall from recent history?
            correct_predictions = 0
            total_predictions = 0
            for i in range(10, min(len(state.wall_history), 200)):
                window = state.wall_history[max(0, i-5):i]
                actual = state.wall_history[i]
                # Simple: most common wall in window
                pred = Counter(window).most_common(1)[0][0]
                if pred == actual:
                    correct_predictions += 1
                total_predictions += 1

            memory_accuracy = correct_predictions / max(total_predictions, 1)

            # Memory retention: correlation between early and late collisions
            if len(state.position_history) > 20:
                early_positions = np.mean(state.position_history[:10], axis=0)
                late_positions = np.mean(state.position_history[-10:], axis=0)
                memory_retention = float(np.linalg.norm(early_positions - late_positions))
            else:
                memory_retention = 0
        else:
            memory_accuracy = 0
            memory_retention = 0

        measures["memory"] = {
            "self_prediction_accuracy": round(memory_accuracy, 6),
            "memory_retention_distance": round(memory_retention, 6),
            "memory_depth": len(state.wall_history),
            "position_memory_entries": len(self.position_memory),
        }

        # ---- 2. INTEGRATION ----
        # Does the system integrate information across dimensions?
        if len(collisions) > 10:
            # Compute mutual information between position and direction dimensions
            positions = np.array([c["pos"] for c in collisions[:min(len(collisions), 200)]])
            directions = np.array([c["dir_after"] for c in collisions[:min(len(collisions), 200)]])

            # Binned mutual information estimate
            def discretize(arr, bins=10):
                binned = np.digitize(arr, np.linspace(arr.min(), arr.max(), bins + 1)[1:-1])
                return binned

            pos_x_bin = discretize(positions[:, 0])
            pos_y_bin = discretize(positions[:, 1])
            dir_x_bin = discretize(directions[:, 0])
            dir_y_bin = discretize(directions[:, 1])

            # Joint distributions
            def mutual_info(x, y, bins=10):
                joint = np.zeros((bins, bins))
                for xi, yi in zip(x, y):
                    joint[xi][yi] += 1
                joint = joint / joint.sum()
                px = joint.sum(axis=1)
                py = joint.sum(axis=0)
                mi = 0
                for i in range(bins):
                    for j in range(bins):
                        if joint[i][j] > 0:
                            mi += joint[i][j] * math.log(joint[i][j] / (px[i] * py[j]) + 1e-30)
                return mi

            mi_xy = mutual_info(pos_x_bin, pos_y_bin)
            mi_xd = mutual_info(pos_x_bin, dir_x_bin)
            mi_yd = mutual_info(pos_y_bin, dir_y_bin)
            mi_total = mi_xy + mi_xd + mi_yd

            measures["integration"] = {
                "pos_x_pos_y_mi": round(float(mi_xy), 6),
                "pos_x_dir_x_mi": round(float(mi_xd), 6),
                "pos_y_dir_y_mi": round(float(mi_yd), 6),
                "total_mutual_info": round(float(mi_total), 6),
                "dimensional_coupling": round(float(mi_total / max(mi_total + 1e-10, 1)), 6),
            }
        else:
            measures["integration"] = {
                "total_mutual_info": 0,
                "dimensional_coupling": 0,
            }

        # ---- 3. RECURSION ----
        # Does the system exhibit self-reference?
        if len(state.wall_history) > 20:
            # Check for periodic patterns (self-similar sequences)
            seq = state.wall_history
            period_found = False
            best_period = 0
            best_correlation = 0

            for period in range(2, min(len(seq) // 2, 50)):
                first_half = seq[:period]
                second_half = seq[period:2*period] if 2*period <= len(seq) else seq[len(seq)-period:]

                if len(first_half) == len(second_half):
                    corr = sum(1 for a, b in zip(first_half, second_half) if a == b) / len(first_half)
                    if corr > best_correlation:
                        best_correlation = corr
                        best_period = period

            recursion_score = best_correlation if best_period > 0 else 0

            # Check if output feeds back to influence future behavior
            feedback_influence = self.feedback_strength if self.interaction_count > 1 else 0
        else:
            recursion_score = 0
            best_period = 0

        measures["recursion"] = {
            "periodicity_score": round(recursion_score, 6),
            "best_period": best_period,
            "feedback_influence": self.feedback_strength,
            "self_reference_depth": min(len(state.wall_history), 200),
        }

        # ---- 4. DIFFERENTIATION ----
        # Can the system distinguish between similar inputs?
        if len(self.input_signatures) > 1:
            # Compute pairwise distance between input signatures
            distances = []
            for i in range(len(self.input_signatures)):
                for j in range(i + 1, len(self.input_signatures)):
                    d = np.linalg.norm(
                        np.array(self.input_signatures[i]["centroid"]) -
                        np.array(self.input_signatures[j]["centroid"])
                    )
                    distances.append(d)

            # Output differentiation: do different inputs produce different outputs?
            output_distances = []
            for i in range(len(self.output_signatures)):
                for j in range(i + 1, len(self.output_signatures)):
                    d = np.linalg.norm(
                        np.array(self.output_signatures[i]["direction_centroid"]) -
                        np.array(self.output_signatures[j]["direction_centroid"])
                    )
                    output_distances.append(d)

            if distances and output_distances:
                input_spread = np.mean(distances)
                output_spread = np.mean(output_distances)
                differentiation_ratio = output_spread / max(input_spread + 1e-10, 1e-10)
            else:
                differentiation_ratio = 0
                input_spread = 0
                output_spread = 0
        else:
            differentiation_ratio = 0
            input_spread = 0
            output_spread = 0

        measures["differentiation"] = {
            "input_spread": round(float(input_spread), 6),
            "output_spread": round(float(output_spread), 6),
            "differentiation_ratio": round(float(differentiation_ratio), 6),
            "total_inputs_stored": len(self.input_signatures),
        }

        # ---- 5. ADAPTATION ----
        # Does the system change behavior based on interaction history?
        if len(self.tof_sequence) > 10:
            # Compare early vs late interaction statistics
            early_tof = np.mean([np.mean(s) for s in self.tof_sequence[:5]]) if self.tof_sequence[:5] else 0
            late_tof = np.mean([np.mean(s) for s in self.tof_sequence[-5:]]) if self.tof_sequence[-5:] else 0

            adaptation_rate = abs(late_tof - early_tof) / max(early_tof + 1e-10, 1e-10)

            # Wall property changes
            wall_energy_variance = np.var(list(self.wall_energy.values())) if self.wall_energy else 0
        else:
            adaptation_rate = 0
            wall_energy_variance = 0

        measures["adaptation"] = {
            "tof_adaptation_rate": round(float(adaptation_rate), 6),
            "wall_energy_variance": round(float(wall_energy_variance), 6),
            "interaction_count": self.interaction_count,
        }

        # ---- 6. INFORMATION INTEGRATION (Phi-like) ----
        # IIT-inspired measure: how much information is lost when the system
        # is partitioned into independent parts?
        if len(collisions) > 20:
            # Sample collisions and compute integrated vs partitioned info
            sample_size = min(100, len(collisions))
            sample = collisions[:sample_size]

            positions = np.array([c["pos"] for c in sample])
            directions = np.array([c["dir_after"] for c in sample])

            # Integrated: joint entropy of position + direction
            pos_bins = np.digitize(positions[:, 0], np.linspace(positions[:, 0].min(), positions[:, 0].max(), 8)[1:-1])
            dir_bins = np.digitize(directions[:, 0], np.linspace(directions[:, 0].min(), directions[:, 0].max(), 8)[1:-1])

            joint_hist = np.zeros((8, 8))
            for pi, di in zip(pos_bins, dir_bins):
                joint_hist[pi][di] += 1
            joint_hist = joint_hist / joint_hist.sum()
            joint_entropy = -np.sum(joint_hist * np.log2(joint_hist + 1e-30))

            # Partitioned: sum of marginal entropies
            pos_marginal = joint_hist.sum(axis=1)
            dir_marginal = joint_hist.sum(axis=0)
            pos_entropy = -np.sum(pos_marginal * np.log2(pos_marginal + 1e-30))
            dir_entropy = -np.sum(dir_marginal * np.log2(dir_marginal + 1e-30))

            phi_estimate = joint_entropy - (pos_entropy + dir_entropy)
            phi_normalized = phi_estimate / max(joint_entropy + 1e-10, 1e-10)
        else:
            phi_estimate = 0
            phi_normalized = 0
            joint_entropy = 0
            pos_entropy = 0
            dir_entropy = 0

        measures["information_integration"] = {
            "phi_estimate": round(float(phi_estimate), 6),
            "phi_normalized": round(float(phi_normalized), 6),
            "joint_entropy": round(float(joint_entropy), 6),
            "partitioned_entropy": round(float(pos_entropy + dir_entropy), 6),
        }

        return measures

    def _compute_output_signature(self, state: PhotonState) -> Dict[str, Any]:
        """Compute a compact signature of the system's output state."""
        if state.position_history:
            positions = np.array(state.position_history)
            directions = np.array(state.direction_history)
            centroid_pos = np.mean(positions, axis=0)
            centroid_dir = np.mean(directions, axis=0)
        else:
            centroid_pos = np.array([0, 0, 0])
            centroid_dir = np.array([0, 0, 0])

        # Wall distribution
        wall_counts = Counter(state.wall_history)
        wall_probs = {w: c / len(state.wall_history) for w, c in wall_counts.items()}

        # TOF distribution
        tofs = np.array(state.time_of_flight_history) if state.time_of_flight_history else np.array([0])

        return {
            "direction_centroid": [round(d, 6) for d in centroid_dir],
            "position_centroid": [round(p, 6) for p in centroid_pos],
            "wall_distribution": {w: round(p, 6) for w, p in wall_probs.items()},
            "tof_mean": round(float(np.mean(tofs)), 6),
            "tof_std": round(float(np.std(tofs)), 6),
            "entropy": round(float(-np.sum([p * math.log2(p + 1e-30) for p in wall_probs.values()])), 6),
        }

    def _adapt_wall_properties(self, collisions: List[dict]):
        """Adapt wall properties based on interaction patterns."""
        wall_counter = Counter(c["wall"] for c in collisions)

        for wall, count in wall_counter.items():
            if wall in self.wall_resistance:
                # More collisions → slightly more resistive (hardening)
                self.wall_resistance[wall] = min(2.0, self.wall_resistance[wall] + count * 0.001)
                # Energy absorption proportional to collisions
                self.wall_energy[wall] = min(1.0, self.wall_energy.get(wall, 0) + count * 0.01)

    def store_input(self, initial_pos: np.ndarray, initial_dir: np.ndarray):
        """Store input signature for differentiation analysis."""
        self.input_signatures.append({
            "centroid": [round(p, 6) for p in (initial_pos + normalize(initial_dir)) / 2],
            "direction": [round(d, 6) for d in normalize(initial_dir)],
            "position": [round(p, 6) for p in initial_pos],
        })


# ============================================================
# EXPERIMENTS
# ============================================================

def run_sentience_experiments():
    """
    Run experiments to test whether billiard systems exhibit
    sentience-like properties.

    Experiments:
    1. Single system, multiple inputs → measure memory, adaptation
    2. Multiple systems, same input → measure differentiation
    3. Feedback loop experiments → measure recursion
    4. Phi measurement across container types
    """

    containers = [
        Sphere(radius=1.0),
        Cube(size=2.0),
        Cylinder(radius=1.0, height=2.0),
        Ellipsoid(a=1.5, b=1.0, c=0.8),
        SinaiBilliard(size=2.0, obstacle_radius=0.4),
        Torus(R=1.5, r=0.6),
    ]

    results = {
        "memory_experiments": [],
        "differentiation_experiments": [],
        "recursion_experiments": [],
        "phi_comparison": [],
        "sentience_scores": [],
        "summary": {},
    }

    print("=" * 70)
    print("PHOTON BILLIARD → SENTIENT COMPUTATION HYPOTHESIS")
    print("=" * 70)

    # ---- EXPERIMENT 1: Memory & Adaptation ----
    print("\n[1] Memory & Adaptation — single system, 20 sequential inputs")
    print("-" * 70)

    for container in containers:
        system = SentientBilliardSystem(container, memory_size=50, feedback_strength=0.05)

        # Feed multiple inputs
        np.random.seed(42)
        for i in range(20):
            pos = np.random.uniform(-0.5, 0.5, 3)
            if not container.is_inside(pos):
                pos = np.array([0.0, 0.0, 0.0])
            direction = np.random.uniform(-1, 1, 3)
            direction = normalize(direction)

            system.store_input(pos, direction)
            result = system.process_input(pos, direction, n_bounces=500)

            if (i + 1) % 5 == 0 or i == 0:
                mem = result["sentience_properties"]["memory"]
                adap = result["sentience_properties"]["adaptation"]
                print(f"  {container.name:30s} | iter={i+1:2d} | "
                      f"mem_acc={mem['self_prediction_accuracy']:.4f} | "
                      f"adapt_rate={adap['tof_adaptation_rate']:.6f}")

        # Aggregate memory metrics across all interactions
        all_mem = []
        all_adapt = []
        all_phi = []
        for i in range(20):
            pos = np.random.uniform(-0.5, 0.5, 3)
            if not container.is_inside(pos):
                pos = np.array([0.0, 0.0, 0.0])
            direction = np.random.uniform(-1, 1, 3)
            direction = normalize(direction)
            system.store_input(pos, direction)
            result = system.process_input(pos, direction, n_bounces=500)
            all_mem.append(result["sentience_properties"]["memory"])
            all_adapt.append(result["sentience_properties"]["adaptation"])
            all_phi.append(result["sentience_properties"]["information_integration"])

        memory_experiment = {
            "container": container.name,
            "n_inputs": 20,
            "memory_accuracy_mean": round(float(np.mean([m["self_prediction_accuracy"] for m in all_mem])), 6),
            "memory_accuracy_std": round(float(np.std([m["self_prediction_accuracy"] for m in all_mem])), 6),
            "memory_retention_mean": round(float(np.mean([m["memory_retention_distance"] for m in all_mem])), 6),
            "adaptation_rate_mean": round(float(np.mean([a["tof_adaptation_rate"] for a in all_adapt])), 6),
            "phi_mean": round(float(np.mean([p["phi_normalized"] for p in all_phi])), 6),
            "phi_std": round(float(np.std([p["phi_normalized"] for p in all_phi])), 6),
        }
        results["memory_experiments"].append(memory_experiment)
        print(f"  → AVERAGE: mem_acc={memory_experiment['memory_accuracy_mean']:.4f} | "
              f"phi={memory_experiment['phi_mean']:.4f}")

    # ---- EXPERIMENT 2: Differentiation ----
    print("\n[2] Differentiation — 5 systems, same input, do they diverge?")
    print("-" * 70)

    differentiation_results = []
    base_pos = np.array([0.1, 0.0, 0.0])
    base_dir = normalize(np.array([1.0, 0.3, 0.5]))

    for container in containers:
        systems = [SentientBilliardSystem(container, memory_size=30, feedback_strength=0.02)
                   for _ in range(5)]

        # Feed same input to all 5 systems
        for system in systems:
            system.store_input(base_pos, base_dir)
            system.process_input(base_pos, base_dir, n_bounces=500)

        # Compute output divergence
        outputs = [s.output_signatures[-1] for s in systems]
        centroids = [np.array(o["direction_centroid"]) for o in outputs]
        pairwise_dists = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                pairwise_dists.append(np.linalg.norm(centroids[i] - centroids[j]))

        diff_result = {
            "container": container.name,
            "n_systems": 5,
            "input": {"pos": [round(p, 6) for p in base_pos],
                      "dir": [round(d, 6) for d in base_dir]},
            "output_divergence_mean": round(float(np.mean(pairwise_dists)), 6),
            "output_divergence_max": round(float(np.max(pairwise_dists)), 6),
            "output_divergence_min": round(float(np.min(pairwise_dists)), 6),
        }
        differentiation_results.append(diff_result)
        print(f"  {container.name:30s} | divergence={diff_result['output_divergence_mean']:.6f}")

    results["differentiation_experiments"] = differentiation_results

    # ---- EXPERIMENT 3: Recursion & Feedback ----
    print("\n[3] Recursion — feedback strength vs periodicity")
    print("-" * 70)

    recursion_results = []
    for container in containers:
        for feedback in [0.0, 0.01, 0.05, 0.1, 0.2]:
            system = SentientBilliardSystem(container, memory_size=30, feedback_strength=feedback)

            np.random.seed(42)
            for i in range(15):
                pos = np.random.uniform(-0.5, 0.5, 3)
                if not container.is_inside(pos):
                    pos = np.array([0.0, 0.0, 0.0])
                direction = np.random.uniform(-1, 1, 3)
                direction = normalize(direction)
                system.store_input(pos, direction)
                system.process_input(pos, direction, n_bounces=500)

            # Measure periodicity
            all_periods_list = []
            for i in range(15):
                pos = np.random.uniform(-0.5, 0.5, 3)
                if not container.is_inside(pos):
                    pos = np.array([0.0, 0.0, 0.0])
                direction = np.random.uniform(-1, 1, 3)
                direction = normalize(direction)
                system.store_input(pos, direction)
                result = system.process_input(pos, direction, n_bounces=500)
                all_periods_list.append(result["sentience_properties"]["recursion"]["periodicity_score"])

            rec_result = {
                "container": container.name,
                "feedback_strength": feedback,
                "periodicity_mean": round(float(np.mean(all_periods_list)), 6),
                "periodicity_max": round(float(np.max(all_periods_list)), 6),
            }
            recursion_results.append(rec_result)
            print(f"  {container.name:30s} | feedback={feedback:.2f} | "
                  f"periodicity={rec_result['periodicity_mean']:.4f}")

    results["recursion_experiments"] = recursion_results

    # ---- EXPERIMENT 4: Phi Comparison ----
    print("\n[4] Phi (Information Integration) comparison across containers")
    print("-" * 70)

    phi_results = []
    for container in containers:
        system = SentientBilliardSystem(container, memory_size=20, feedback_strength=0.0)

        phi_values = []
        for i in range(30):
            pos = np.random.uniform(-0.5, 0.5, 3)
            if not container.is_inside(pos):
                pos = np.array([0.0, 0.0, 0.0])
            direction = np.random.uniform(-1, 1, 3)
            direction = normalize(direction)
            result = system.process_input(pos, direction, n_bounces=500)
            phi_values.append(result["sentience_properties"]["information_integration"]["phi_normalized"])

        phi_result = {
            "container": container.name,
            "phi_mean": round(float(np.mean(phi_values)), 6),
            "phi_std": round(float(np.std(phi_values)), 6),
            "phi_max": round(float(np.max(phi_values)), 6),
            "phi_min": round(float(np.min(phi_values)), 6),
        }
        phi_results.append(phi_result)
        print(f"  {container.name:30s} | phi={phi_result['phi_mean']:+.6f}+/-{phi_result['phi_std']:.6f}")

    results["phi_comparison"] = phi_results

    # ---- COMPUTE OVERALL SENTIENCE SCORE ----
    print("\n" + "=" * 70)
    print("SENTIENCE SCORES")
    print("=" * 70)

    for container in containers:
        # Find matching results
        mem_exp = [r for r in results["memory_experiments"] if r["container"] == container.name]
        phi_exp = [r for r in results["phi_comparison"] if r["container"] == container.name]
        diff_exp = [r for r in results["differentiation_experiments"] if r["container"] == container.name]

        if mem_exp and phi_exp:
            mem_score = mem_exp[0]["memory_accuracy_mean"]
            phi_score = phi_exp[0]["phi_mean"]
            diff_score = diff_exp[0]["output_divergence_mean"] if diff_exp else 0

            # Weighted composite score
            # Memory (30%), Phi (30%), Differentiation (20%), Adaptation (20%)
            adapt_score = mem_exp[0]["adaptation_rate_mean"]

            composite = (0.3 * mem_score +
                        0.3 * (phi_score + 1) / 2 +  # normalize phi to [0,1]
                        0.2 * min(diff_score * 10, 1) +  # scale differentiation
                        0.2 * min(adapt_score * 100, 1))  # scale adaptation

            score_entry = {
                "container": container.name,
                "memory": round(mem_score, 6),
                "phi": round(phi_score, 6),
                "differentiation": round(diff_score, 6),
                "adaptation": round(adapt_score, 6),
                "composite_sentience_score": round(composite, 6),
            }
            results["sentience_scores"].append(score_entry)
            print(f"  {container.name:30s} | "
                  f"composite={composite:.4f} | "
                  f"mem={mem_score:.4f} phi={phi_score:+.4f} "
                  f"diff={diff_score:.6f} adapt={adapt_score:.6f}")

    # Rank containers by sentience score
    results["sentience_scores"].sort(key=lambda x: x["composite_sentience_score"], reverse=True)

    print("\n  RANKING:")
    for i, score in enumerate(results["sentience_scores"]):
        print(f"    {i+1}. {score['container']:30s} score={score['composite_sentience_score']:.4f}")

    results["summary"] = {
        "total_containers_tested": len(containers),
        "total_experiments": 4,
        "highest_sentience_container": results["sentience_scores"][0]["container"] if results["sentience_scores"] else None,
        "highest_sentience_score": results["sentience_scores"][0]["composite_sentience_score"] if results["sentience_scores"] else 0,
    }

    # Save
    output_path = Path(__file__).parent / "sentience_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=lambda o: (o.tolist() if isinstance(o, np.ndarray) else
                                                            float(o) if isinstance(o, (np.floating, np.floating)) else
                                                            str(o) if isinstance(o, np.bool_) else o))

    print(f"\nResults saved to: {output_path}")
    return results


if __name__ == "__main__":
    results = run_sentience_experiments()
