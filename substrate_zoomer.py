#!/usr/bin/env python3
"""
UMST Substrate Zoomer - Visualization and Discovery Harness
Built on top of the Unified Mathematical Substrate Theory Engine

This is the first visualization tool that demonstrates how local chaos
transforms into global order when viewed through UMST dynamics.

The Zoomer shows:
- How micro-loops aggregate into resonance motifs
- How motifs stabilize into harmonic lattices  
- How cross-scale patterns emerge as substrate fingerprints

Author: UMST Research Team
Version: 1.0.0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.colors import hsv_to_rgb
import json
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Loop:
    """A fundamental loop in the UMST substrate."""
    id: int
    phase: float  # Current phase [0, 2π)
    frequency: float  # Base oscillation frequency
    amplitude: float  # Oscillation amplitude
    position: np.ndarray  # 2D position in substrate
    coupling_strength: float = 1.0  # How strongly this couples to others
    bound_loops: List[int] = field(default_factory=list)  # IDs of bound loops
    binding_weights: List[float] = field(default_factory=list)  # Weights of bindings
    last_resonance: float = 0.0  # Last measured resonance
    friction: float = 0.1  # Internal friction parameter
    
    def __post_init__(self):
        if isinstance(self.position, list):
            self.position = np.array(self.position)


@dataclass
class Motif:
    """A resonance motif - collection of coupled loops."""
    id: int
    loop_ids: List[int]
    centroid: np.ndarray
    harmony: float  # Harmonic measure
    stability: float  # Temporal stability
    scale: int  # Which coarse-graining scale this exists at
    formation_time: float = 0.0
    
    def __post_init__(self):
        if isinstance(self.centroid, list):
            self.centroid = np.array(self.centroid)


@dataclass 
class LatticeNode:
    """A node in the emergent harmonic lattice."""
    id: int
    position: np.ndarray
    motifs: List[int]  # Motif IDs at this node
    order_parameter: float  # Local order measure
    scale: int  # Scale level
    connections: List[int] = field(default_factory=list)  # Connected node IDs
    
    def __post_init__(self):
        if isinstance(self.position, list):
            self.position = np.array(self.position)


class TemporalSmoothingLayer:
    """TSL: Temporal Smoothing Layer for stable present state."""
    
    def __init__(self, window_size: int = 10, alpha: float = 0.3):
        self.window_size = window_size
        self.alpha = alpha  # EWMA coefficient
        self.history: List[float] = []
        self.smoothed_value: float = 0.0
    
    def update(self, value: float) -> float:
        """Update with new value and return smoothed result."""
        self.history.append(value)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        # Exponentially weighted moving average
        if len(self.history) == 1:
            self.smoothed_value = value
        else:
            self.smoothed_value = self.alpha * value + (1 - self.alpha) * self.smoothed_value
        
        return self.smoothed_value


class HASRReinforcement:
    """Hebbian-Analog Symbolic Reinforcement for loop coupling."""
    
    def __init__(self, alpha: float = 0.1, beta: float = 0.05, decay: float = 0.99):
        self.alpha = alpha  # Learning rate for reinforcement
        self.beta = beta   # Learning rate for decay
        self.decay = decay  # Weight decay factor
        self.min_weight = 0.01
        self.max_weight = 2.0
    
    def update_binding(self, weight: float, resonance: float, dissonance: float = 0.0) -> float:
        """Update binding weight based on resonance and dissonance."""
        # Hebbian-like update
        delta = self.alpha * resonance - self.beta * dissonance
        new_weight = weight * self.decay + delta
        
        # Clip to valid range
        return np.clip(new_weight, self.min_weight, self.max_weight)


class SubstrateZoomer:
    """
    The main UMST Substrate Zoomer engine.
    Demonstrates emergence of order from chaos through UMST dynamics.
    """
    
    def __init__(self, 
                 substrate_size: Tuple[int, int] = (100, 100),
                 num_loops: int = 1000,
                 seed: Optional[int] = 42):
        """Initialize the substrate zoomer."""
        self.substrate_size = substrate_size
        self.num_loops = num_loops
        self.seed = seed
        
        if seed is not None:
            np.random.seed(seed)
            
        # Core components
        self.loops: List[Loop] = []
        self.motifs: List[Motif] = []
        self.lattice_nodes: List[LatticeNode] = []
        self.hasr = HASRReinforcement()
        self.tsl = TemporalSmoothingLayer()
        
        # Simulation state
        self.time_step = 0
        self.dt = 0.01
        self.entropy = 0.0
        self.global_order = 0.0
        self.global_friction = 0.1
        
        # Metrics tracking
        self.metrics_history = {
            'time': [],
            'order_parameter': [],
            'harmony': [],
            'bound_fraction': [],
            'entropy': [],
            'friction': [],
            'num_motifs': [],
            'lattice_size': []
        }
        
        # Visualization state
        self.fig = None
        self.axes = None
        
        logger.info(f"Initialized UMST Substrate Zoomer with {num_loops} loops on {substrate_size} substrate")
    
    def seed_substrate(self, planted_motifs: int = 3, chaos_level: float = 0.8):
        """Seed the substrate with loops - some random, some in planted motifs."""
        self.loops = []
        
        # Plant some organized motifs
        for motif_idx in range(planted_motifs):
            center = np.random.uniform(10, min(self.substrate_size) - 10, 2)
            motif_size = np.random.randint(8, 20)
            
            for i in range(motif_size):
                angle = 2 * np.pi * i / motif_size
                radius = np.random.uniform(3, 8)
                position = center + radius * np.array([np.cos(angle), np.sin(angle)])
                
                # Ensure position is within bounds
                position = np.clip(position, [2, 2], 
                                 [self.substrate_size[0] - 2, self.substrate_size[1] - 2])
                
                frequency = np.random.uniform(0.5, 1.5)  # Similar frequencies for motifs
                loop = Loop(
                    id=len(self.loops),
                    phase=np.random.uniform(0, 2*np.pi),
                    frequency=frequency,
                    amplitude=np.random.uniform(0.5, 1.0),
                    position=position,
                    coupling_strength=np.random.uniform(0.8, 1.2)
                )
                self.loops.append(loop)
        
        # Fill rest with chaotic loops
        remaining_loops = self.num_loops - len(self.loops)
        for i in range(remaining_loops):
            position = np.random.uniform(0, self.substrate_size, 2)
            frequency = np.random.uniform(0.1, 3.0)  # Wide frequency range for chaos
            
            loop = Loop(
                id=len(self.loops),
                phase=np.random.uniform(0, 2*np.pi),
                frequency=frequency,
                amplitude=np.random.uniform(0.1, 1.5),
                position=position,
                coupling_strength=np.random.uniform(0.2, 1.5)
            )
            self.loops.append(loop)
        
        logger.info(f"Seeded substrate with {planted_motifs} planted motifs and {remaining_loops} chaotic loops")
    
    def detect_resonance(self, tile_size: int = 10) -> Tuple[float, float]:
        """Detect local resonance patterns across substrate tiles."""
        if not self.loops:
            return 0.0, 0.0
            
        # Create grid of tiles
        x_tiles = self.substrate_size[0] // tile_size
        y_tiles = self.substrate_size[1] // tile_size
        
        harmony_sum = 0.0
        order_sum = 0.0
        tile_count = 0
        
        for i in range(x_tiles):
            for j in range(y_tiles):
                # Get loops in this tile
                x_min, x_max = i * tile_size, (i + 1) * tile_size
                y_min, y_max = j * tile_size, (j + 1) * tile_size
                
                tile_loops = [
                    loop for loop in self.loops
                    if x_min <= loop.position[0] < x_max and y_min <= loop.position[1] < y_max
                ]
                
                if len(tile_loops) < 2:
                    continue
                
                # Measure local harmony (phase coherence)
                phases = np.array([loop.phase for loop in tile_loops])
                frequencies = np.array([loop.frequency for loop in tile_loops])
                
                # Kuramoto order parameter for local harmony
                complex_phases = np.exp(1j * phases)
                local_order = np.abs(np.mean(complex_phases))
                
                # Frequency coherence as harmony measure  
                freq_std = np.std(frequencies)
                local_harmony = 1.0 / (1.0 + freq_std)
                
                harmony_sum += local_harmony
                order_sum += local_order
                tile_count += 1
        
        global_harmony = harmony_sum / tile_count if tile_count > 0 else 0.0
        global_order = order_sum / tile_count if tile_count > 0 else 0.0
        
        return global_harmony, global_order
    
    def hasr_reinforcement(self):
        """Apply HASR reinforcement to strengthen resonant couplings."""
        if len(self.loops) < 2:
            return
        
        # For each loop, find nearby loops and measure resonance
        for loop in self.loops:
            nearby_loops = [
                other for other in self.loops 
                if other.id != loop.id and 
                np.linalg.norm(other.position - loop.position) < 15.0
            ]
            
            for nearby in nearby_loops:
                # Measure phase and frequency resonance
                phase_diff = abs(loop.phase - nearby.phase)
                phase_resonance = 1.0 / (1.0 + phase_diff)
                
                freq_diff = abs(loop.frequency - nearby.frequency)  
                freq_resonance = 1.0 / (1.0 + freq_diff)
                
                total_resonance = (phase_resonance + freq_resonance) / 2.0
                
                # Check if binding already exists
                if nearby.id not in loop.bound_loops:
                    # Create new binding
                    loop.bound_loops.append(nearby.id)
                    loop.binding_weights.append(0.5)
                
                # Update binding weight via HASR
                binding_idx = loop.bound_loops.index(nearby.id)
                old_weight = loop.binding_weights[binding_idx]
                new_weight = self.hasr.update_binding(old_weight, total_resonance)
                loop.binding_weights[binding_idx] = new_weight
                
                loop.last_resonance = total_resonance
    
    def apply_friction_tsl(self):
        """Apply friction and temporal smoothing."""
        # Estimate entropy from phase distribution  
        phases = [loop.phase for loop in self.loops]
        phase_hist, _ = np.histogram(phases, bins=20, range=(0, 2*np.pi))
        phase_probs = phase_hist / len(phases)
        phase_probs = phase_probs[phase_probs > 0]  # Remove zeros
        self.entropy = -np.sum(phase_probs * np.log(phase_probs))
        
        # Adaptive friction based on entropy
        target_entropy = 2.0  # Moderate entropy target
        entropy_error = self.entropy - target_entropy
        self.global_friction = np.clip(0.05 + 0.1 * entropy_error, 0.01, 0.5)
        
        # Apply friction to loop dynamics
        for loop in self.loops:
            loop.friction = self.global_friction + np.random.normal(0, 0.02)
            loop.friction = np.clip(loop.friction, 0.01, 0.5)
    
    def coarse_grain_step(self, scale: int):
        """Perform coarse-graining to reveal larger-scale structure."""
        # Clear existing motifs and lattice at this scale
        self.motifs = [m for m in self.motifs if m.scale != scale]
        self.lattice_nodes = [n for n in self.lattice_nodes if n.scale != scale]
        
        # Group loops into motifs based on binding strength
        unassigned_loops = set(range(len(self.loops)))
        motif_id = 0
        
        while unassigned_loops:
            # Start new motif with highest coupling loop
            seed_loop_id = max(unassigned_loops, 
                             key=lambda i: self.loops[i].coupling_strength)
            
            motif_loops = {seed_loop_id}
            unassigned_loops.remove(seed_loop_id)
            
            # Grow motif by adding strongly bound loops
            while True:
                added_any = False
                for loop_id in list(motif_loops):
                    loop = self.loops[loop_id]
                    
                    for bound_id, weight in zip(loop.bound_loops, loop.binding_weights):
                        if bound_id in unassigned_loops and weight > 0.8:  # Strong binding threshold
                            motif_loops.add(bound_id)
                            unassigned_loops.remove(bound_id)
                            added_any = True
                
                if not added_any:
                    break
            
            # Create motif if it has enough loops
            if len(motif_loops) >= 3:
                loop_positions = [self.loops[i].position for i in motif_loops]
                centroid = np.mean(loop_positions, axis=0)
                
                # Calculate motif harmony
                motif_phases = [self.loops[i].phase for i in motif_loops]
                complex_phases = np.exp(1j * np.array(motif_phases))
                harmony = float(np.abs(np.mean(complex_phases)))
                
                motif = Motif(
                    id=motif_id,
                    loop_ids=list(motif_loops),
                    centroid=centroid,
                    harmony=harmony,
                    stability=harmony,  # Simplified stability measure
                    scale=scale,
                    formation_time=self.time_step * self.dt
                )
                self.motifs.append(motif)
                motif_id += 1
        
        # Create lattice nodes from stable motifs
        stable_motifs = [m for m in self.motifs if m.scale == scale and m.stability > 0.6]
        
        for motif in stable_motifs:
            # Calculate local order parameter
            nearby_loops = [self.loops[i] for i in motif.loop_ids]
            phases = [loop.phase for loop in nearby_loops]
            complex_phases = np.exp(1j * np.array(phases))
            order_param = float(np.abs(np.mean(complex_phases)))
            
            lattice_node = LatticeNode(
                id=len(self.lattice_nodes),
                position=motif.centroid.copy(),
                motifs=[motif.id],
                order_parameter=order_param,
                scale=scale
            )
            self.lattice_nodes.append(lattice_node)
    
    def evolve_substrate(self, steps: int = 1):
        """Evolve the substrate through UMST dynamics."""
        for _ in range(steps):
            # Update loop phases
            for loop in self.loops:
                # Base phase evolution
                phase_velocity = loop.frequency * (1 - loop.friction)
                
                # Coupling effects from bound loops
                coupling_force = 0.0
                for bound_id, weight in zip(loop.bound_loops, loop.binding_weights):
                    if bound_id < len(self.loops):
                        bound_loop = self.loops[bound_id]
                        phase_diff = bound_loop.phase - loop.phase
                        coupling_force += weight * np.sin(phase_diff) * 0.1
                
                # Update phase
                loop.phase += (phase_velocity + coupling_force) * self.dt
                loop.phase = loop.phase % (2 * np.pi)
            
            self.time_step += 1
    
    def run_zooming_simulation(self, 
                              total_steps: int = 1000,
                              coarse_grain_interval: int = 100,
                              max_scales: int = 3) -> Dict[str, Any]:
        """Run the complete zooming simulation."""
        logger.info(f"Starting zooming simulation: {total_steps} steps, {max_scales} scales")
        
        results = {
            'metrics': [],
            'motifs_by_scale': {},
            'lattice_by_scale': {},
            'final_state': None
        }
        
        for step in range(total_steps):
            # Core evolution
            self.evolve_substrate(1)
            
            # Detect resonance
            harmony, order = self.detect_resonance()
            
            # Apply HASR reinforcement
            self.hasr_reinforcement()
            
            # Apply friction and TSL
            self.apply_friction_tsl()
            
            # Smooth order parameter
            smoothed_order = self.tsl.update(order)
            
            # Coarse-graining at intervals
            if step % coarse_grain_interval == 0:
                current_scale = (step // coarse_grain_interval) % max_scales + 1
                self.coarse_grain_step(current_scale)
            
            # Record metrics
            bound_count = sum(len(loop.bound_loops) for loop in self.loops)
            bound_fraction = bound_count / (len(self.loops) * len(self.loops)) if self.loops else 0
            
            metrics = {
                'time': self.time_step * self.dt,
                'order_parameter': smoothed_order,
                'harmony': harmony,
                'bound_fraction': bound_fraction,
                'entropy': self.entropy,
                'friction': self.global_friction,
                'num_motifs': len(self.motifs),
                'lattice_size': len(self.lattice_nodes)
            }
            
            for key, value in metrics.items():
                self.metrics_history[key].append(value)
            
            if step % 100 == 0:
                logger.info(f"Step {step}: Order={smoothed_order:.3f}, Harmony={harmony:.3f}, "
                          f"Motifs={len(self.motifs)}, Lattice={len(self.lattice_nodes)}")
        
        # Organize results by scale
        for scale in range(1, max_scales + 1):
            results['motifs_by_scale'][scale] = [
                m for m in self.motifs if m.scale == scale
            ]
            results['lattice_by_scale'][scale] = [
                n for n in self.lattice_nodes if n.scale == scale  
            ]
        
        results['metrics'] = self.metrics_history
        results['final_state'] = {
            'loops': len(self.loops),
            'total_motifs': len(self.motifs),
            'total_lattice_nodes': len(self.lattice_nodes),
            'final_order': smoothed_order,
            'final_harmony': harmony
        }
        
        logger.info("Simulation complete!")
        return results
    
    def create_visualizations(self, results: Dict[str, Any], output_dir: str = "output"):
        """Create comprehensive visualizations of the results."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. Metrics over time
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("UMST Substrate Zoomer - Emergence Metrics", fontsize=16)
        
        # Order parameter evolution
        axes[0,0].plot(self.metrics_history['time'], self.metrics_history['order_parameter'], 'b-', linewidth=2)
        axes[0,0].set_title('Order Parameter (R_k) vs Time')
        axes[0,0].set_xlabel('Time')
        axes[0,0].set_ylabel('Order Parameter')
        axes[0,0].grid(True, alpha=0.3)
        
        # Harmony evolution
        axes[0,1].plot(self.metrics_history['time'], self.metrics_history['harmony'], 'g-', linewidth=2)
        axes[0,1].set_title('Harmony (H_k) vs Time')
        axes[0,1].set_xlabel('Time')
        axes[0,1].set_ylabel('Harmony')
        axes[0,1].grid(True, alpha=0.3)
        
        # Motif and lattice growth
        axes[1,0].plot(self.metrics_history['time'], self.metrics_history['num_motifs'], 'r-', label='Motifs', linewidth=2)
        axes[1,0].plot(self.metrics_history['time'], self.metrics_history['lattice_size'], 'm-', label='Lattice Nodes', linewidth=2)
        axes[1,0].set_title('Structure Formation')
        axes[1,0].set_xlabel('Time')
        axes[1,0].set_ylabel('Count')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Entropy and friction
        axes[1,1].plot(self.metrics_history['time'], self.metrics_history['entropy'], 'c-', label='Entropy', linewidth=2)
        axes[1,1].plot(self.metrics_history['time'], self.metrics_history['friction'], 'y-', label='Friction', linewidth=2)
        axes[1,1].set_title('System Regulation')
        axes[1,1].set_xlabel('Time')
        axes[1,1].set_ylabel('Value')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "metrics_evolution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Multi-scale substrate visualization
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle("Multi-Scale Substrate Structure", fontsize=16)
        
        scales = [1, 2, 3]
        for i, scale in enumerate(scales):
            ax = axes[i]
            
            # Plot all loops as background
            for loop in self.loops:
                color_hue = loop.phase / (2 * np.pi)
                color = hsv_to_rgb([color_hue, 0.3, 0.8])
                ax.scatter(loop.position[0], loop.position[1], 
                          c=[color], s=10, alpha=0.4)
            
            # Plot motifs at this scale
            scale_motifs = results['motifs_by_scale'].get(scale, [])
            for motif in scale_motifs:
                # Draw motif as circle
                circle = patches.Circle(motif.centroid, radius=8, 
                                      fill=False, edgecolor='red', linewidth=2, alpha=0.8)
                ax.add_patch(circle)
                
                # Draw connections between motif loops
                loop_positions = [self.loops[i].position for i in motif.loop_ids]
                for j, pos1 in enumerate(loop_positions):
                    for pos2 in loop_positions[j+1:]:
                        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]], 
                               'r-', alpha=0.3, linewidth=1)
            
            # Plot lattice nodes at this scale
            scale_lattice = results['lattice_by_scale'].get(scale, [])
            for node in scale_lattice:
                # Size by order parameter
                size = 50 + 100 * node.order_parameter
                ax.scatter(node.position[0], node.position[1], 
                          c='blue', s=size, alpha=0.8, marker='s')
            
            ax.set_title(f'Scale {scale}: {len(scale_motifs)} Motifs, {len(scale_lattice)} Lattice Nodes')
            ax.set_xlim(0, self.substrate_size[0])
            ax.set_ylim(0, self.substrate_size[1])
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "multiscale_structure.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_path}")
    
    def export_results(self, results: Dict[str, Any], output_dir: str = "output"):
        """Export results in various formats."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. Metrics JSON
        metrics_data = {
            'metadata': {
                'substrate_size': self.substrate_size,
                'num_loops': len(self.loops),
                'simulation_time': self.time_step * self.dt,
                'seed': self.seed
            },
            'metrics': self.metrics_history,
            'summary': results['final_state']
        }
        
        with open(output_path / "metrics.json", 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # 2. Lattice structure (GLB-ready format)
        lattice_data = {
            'nodes': [],
            'edges': [],
            'scales': list(results['lattice_by_scale'].keys())
        }
        
        # Export lattice nodes
        for scale, nodes in results['lattice_by_scale'].items():
            for node in nodes:
                lattice_data['nodes'].append({
                    'id': node.id,
                    'position': node.position.tolist(),
                    'scale': node.scale,
                    'order_parameter': node.order_parameter,
                    'motifs': node.motifs
                })
        
        # Export lattice connections (simplified)
        for scale, nodes in results['lattice_by_scale'].items():
            for i, node1 in enumerate(nodes):
                for j, node2 in enumerate(nodes[i+1:], i+1):
                    distance = np.linalg.norm(node1.position - node2.position)
                    if distance < 25:  # Connection threshold
                        lattice_data['edges'].append({
                            'source': node1.id,
                            'target': node2.id,
                            'weight': 1.0 / (1.0 + distance),
                            'scale': scale
                        })
        
        with open(output_path / "lattice.json", 'w') as f:
            json.dump(lattice_data, f, indent=2)
        
        # 3. Loop state export
        loop_data = {
            'loops': [],
            'bindings': [],
            'timestamp': self.time_step * self.dt
        }
        
        for loop in self.loops:
            loop_data['loops'].append({
                'id': loop.id,
                'position': loop.position.tolist(),
                'phase': float(loop.phase),
                'frequency': float(loop.frequency),
                'amplitude': float(loop.amplitude),
                'coupling_strength': float(loop.coupling_strength),
                'friction': float(loop.friction),
                'last_resonance': float(loop.last_resonance)
            })
            
            # Export bindings
            for bound_id, weight in zip(loop.bound_loops, loop.binding_weights):
                loop_data['bindings'].append({
                    'source': loop.id,
                    'target': bound_id,
                    'weight': float(weight),
                    'type': 'resonant_coupling'
                })
        
        with open(output_path / "loops.json", 'w') as f:
            json.dump(loop_data, f, indent=2)
        
        logger.info(f"Results exported to {output_path}")
        
        return {
            'metrics_file': str(output_path / "metrics.json"),
            'lattice_file': str(output_path / "lattice.json"),
            'loops_file': str(output_path / "loops.json"),
            'visualizations': [
                str(output_path / "metrics_evolution.png"),
                str(output_path / "multiscale_structure.png")
            ]
        }


def run_falsification_tests(base_results: Dict[str, Any]) -> Dict[str, Any]:
    """Run falsification tests to validate UMST predictions."""
    logger.info("Running falsification tests...")
    
    tests = {}
    
    # Test 1: Null seed (pure chaos)
    logger.info("Test 1: Null seed control")
    zoomer_null = SubstrateZoomer(num_loops=1000, seed=123)
    zoomer_null.seed_substrate(planted_motifs=0, chaos_level=1.0)  # Pure chaos
    results_null = zoomer_null.run_zooming_simulation(total_steps=500)
    tests['null_seed'] = {
        'final_order': results_null['final_state']['final_order'],
        'final_motifs': results_null['final_state']['total_motifs'],
        'max_order': max(results_null['metrics']['order_parameter'])
    }
    
    # Test 2: No HASR (disable reinforcement)
    logger.info("Test 2: No HASR control")
    zoomer_no_hasr = SubstrateZoomer(num_loops=1000, seed=456)
    zoomer_no_hasr.seed_substrate(planted_motifs=3, chaos_level=0.8)
    zoomer_no_hasr.hasr.alpha = 0.0  # Disable reinforcement
    results_no_hasr = zoomer_no_hasr.run_zooming_simulation(total_steps=500)
    tests['no_hasr'] = {
        'final_order': results_no_hasr['final_state']['final_order'],
        'final_motifs': results_no_hasr['final_state']['total_motifs'],
        'max_order': max(results_no_hasr['metrics']['order_parameter'])
    }
    
    # Test 3: High friction (suppressed dynamics)
    logger.info("Test 3: High friction control")
    zoomer_high_friction = SubstrateZoomer(num_loops=1000, seed=789)
    zoomer_high_friction.seed_substrate(planted_motifs=3, chaos_level=0.8)
    zoomer_high_friction.global_friction = 0.8  # High friction
    results_high_friction = zoomer_high_friction.run_zooming_simulation(total_steps=500)
    tests['high_friction'] = {
        'final_order': results_high_friction['final_state']['final_order'],
        'final_motifs': results_high_friction['final_state']['total_motifs'],
        'max_order': max(results_high_friction['metrics']['order_parameter'])
    }
    
    # Compare with base results
    base_order = base_results['final_state']['final_order']
    base_motifs = base_results['final_state']['total_motifs']
    
    tests['comparison'] = {
        'base_order': base_order,
        'base_motifs': base_motifs,
        'null_degradation': base_order - tests['null_seed']['final_order'],
        'hasr_importance': base_order - tests['no_hasr']['final_order'],
        'friction_effect': base_order - tests['high_friction']['final_order']
    }
    
    logger.info("Falsification tests complete!")
    return tests


def main():
    """Main function to run the UMST Substrate Zoomer."""
    parser = argparse.ArgumentParser(description='UMST Substrate Zoomer - Visualizing Emergence of Order')
    parser.add_argument('--loops', type=int, default=1000, help='Number of loops to simulate')
    parser.add_argument('--steps', type=int, default=1000, help='Simulation steps')
    parser.add_argument('--size', type=int, nargs=2, default=[100, 100], help='Substrate size [width height]')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='output', help='Output directory')
    parser.add_argument('--falsify', action='store_true', help='Run falsification tests')
    
    args = parser.parse_args()
    
    print("█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + " UMST SUBSTRATE ZOOMER ".center(78) + "█")
    print("█" + " Visualizing the Emergence of Order from Chaos ".center(78) + "█")
    print("█" + " " * 78 + "█") 
    print("█" * 80)
    print()
    
    # Initialize and run simulation
    zoomer = SubstrateZoomer(
        substrate_size=tuple(args.size),
        num_loops=args.loops,
        seed=args.seed
    )
    
    # Seed substrate with planted motifs
    zoomer.seed_substrate(planted_motifs=3, chaos_level=0.8)
    
    # Run simulation
    print("🔬 Running zooming simulation...")
    results = zoomer.run_zooming_simulation(total_steps=args.steps)
    
    # Create visualizations
    print("📊 Creating visualizations...")
    zoomer.create_visualizations(results, args.output)
    
    # Export results
    print("💾 Exporting results...")
    export_info = zoomer.export_results(results, args.output)
    
    # Print summary
    print("\n" + "=" * 80)
    print("SIMULATION SUMMARY")
    print("=" * 80)
    print(f"Final Order Parameter: {results['final_state']['final_order']:.3f}")
    print(f"Final Harmony: {results['final_state']['final_harmony']:.3f}")
    print(f"Total Motifs Formed: {results['final_state']['total_motifs']}")
    print(f"Lattice Nodes: {results['final_state']['total_lattice_nodes']}")
    print(f"Peak Order: {max(results['metrics']['order_parameter']):.3f}")
    print(f"Peak Harmony: {max(results['metrics']['harmony']):.3f}")
    
    print(f"\n📁 Outputs saved to: {args.output}/")
    for file_path in export_info['visualizations']:
        print(f"  📊 {file_path}")
    for key, file_path in export_info.items():
        if key != 'visualizations':
            print(f"  📄 {file_path}")
    
    # Run falsification tests if requested
    if args.falsify:
        print("\n🧪 Running falsification tests...")
        falsification_results = run_falsification_tests(results)
        
        print("\n" + "=" * 80) 
        print("FALSIFICATION TEST RESULTS")
        print("=" * 80)
        print(f"Base case order: {falsification_results['comparison']['base_order']:.3f}")
        print(f"Null seed degradation: {falsification_results['comparison']['null_degradation']:.3f}")
        print(f"No HASR impact: {falsification_results['comparison']['hasr_importance']:.3f}")
        print(f"High friction impact: {falsification_results['comparison']['friction_effect']:.3f}")
        
        # Save falsification results
        falsification_file = Path(args.output) / "falsification_tests.json"
        with open(falsification_file, 'w') as f:
            json.dump(falsification_results, f, indent=2)
        print(f"\n🧪 Falsification results saved to: {falsification_file}")
    
    print("\n✅ UMST Substrate Zoomer simulation complete!")
    print("🔭 The substrate's order has been revealed through zooming out from chaos to cosmos.")


if __name__ == "__main__":
    main()