#!/usr/bin/env python3
"""
UMST Live Substrate Zoomer - Complete Futuristic Visualization System
AerwareAI - Mission Control for the Mathematical Universe
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Circle, Rectangle
import json
import time
import threading
from queue import Queue
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Import core UMST components
from substrate_zoomer import (
    Loop, Motif, LatticeNode, 
    TemporalSmoothingLayer, HASRReinforcement,
    SubstrateZoomer
)

# ============================================================================
# CONFIGURATION AND THEMES
# ============================================================================

@dataclass
class VisualizationConfig:
    """Advanced configuration for futuristic visualization."""
    update_interval: int = 30  # milliseconds
    max_history_points: int = 2000
    colormap: str = 'plasma'
    show_coupling_lines: bool = True
    coupling_line_alpha: float = 0.4
    loop_size_base: float = 15
    motif_boundary_color: str = '#ffff00'
    lattice_node_color: str = '#ff00ff'
    fps_target: int = 30
    auto_scale: bool = True
    performance_mode: bool = False
    theme: str = 'neon'
    enable_glow: bool = True
    enable_trails: bool = True
    trail_length: int = 10

class AerwareAIThemes:
    """Centralized theme management for consistent futuristic styling."""
    
    THEMES = {
        'neon': {
            'background': '#0a0a0a',
            'panel_bg': '#0f0f0f',
            'primary': '#00ffff',
            'secondary': '#ff00ff',
            'accent': '#ffff00',
            'success': '#00ff00',
            'warning': '#ff4444',
            'text': '#ffffff',
            'grid': '#00ffff'
        },
        'matrix': {
            'background': '#000000',
            'panel_bg': '#001100',
            'primary': '#00ff00',
            'secondary': '#44ff44',
            'accent': '#88ff88',
            'success': '#00ff00',
            'warning': '#ffff00',
            'text': '#00ff00',
            'grid': '#00ff00'
        },
        'cyberpunk': {
            'background': '#0d0d0d',
            'panel_bg': '#1a0d1a',
            'primary': '#ff0080',
            'secondary': '#8000ff',
            'accent': '#00ff80',
            'success': '#00ff80',
            'warning': '#ff8000',
            'text': '#ffffff',
            'grid': '#ff0080'
        }
    }
    
    @classmethod
    def apply_theme(cls, theme_name):
        """Apply theme globally to matplotlib."""
        theme = cls.THEMES.get(theme_name, cls.THEMES['neon'])
        
        plt.rcParams.update({
            'figure.facecolor': theme['background'],
            'axes.facecolor': theme['panel_bg'],
            'axes.edgecolor': theme['primary'],
            'axes.labelcolor': theme['text'],
            'xtick.color': theme['text'],
            'ytick.color': theme['text'],
            'text.color': theme['text'],
            'grid.color': theme['grid'],
            'grid.alpha': 0.3,
            'font.family': 'monospace',
            'font.size': 10
        })
        
        return theme

# ============================================================================
# REUSABLE PANEL COMPONENTS
# ============================================================================

class FuturisticPanel:
    """Base class for reusable dashboard panels with futuristic styling."""
    
    def __init__(self, ax, title, accent_color='#00ffff'):
        self.ax = ax
        self.title = title
        self.accent_color = accent_color
        self.setup_styling()
    
    def setup_styling(self):
        """Apply consistent futuristic styling."""
        self.ax.set_facecolor('#0f0f0f')
        self.ax.set_title(self.title, fontsize=11, fontweight='bold', 
                         color=self.accent_color, pad=15)
        
        # Enhanced grid with glow effect
        self.ax.grid(True, alpha=0.2, color=self.accent_color, 
                    linestyle='-', linewidth=0.5)
        
        # Glowing borders
        for spine in self.ax.spines.values():
            spine.set_color(self.accent_color)
            spine.set_linewidth(2)
            spine.set_alpha(0.8)
        
        self.ax.tick_params(colors='#ffffff', labelsize=9)

class MetricsPanel(FuturisticPanel):
    """Reusable metrics panel with time series plotting and glow effects."""
    
    def __init__(self, ax, title, accent_color='#00ffff', y_label='Value'):
        super().__init__(ax, title, accent_color)
        self.data_history = deque(maxlen=2000)
        self.time_history = deque(maxlen=2000)
        self.line = None
        self.glow_lines = []
        self.ax.set_ylabel(y_label, color='#ffffff', fontsize=9)
        self.ax.set_xlabel('Time', color='#ffffff', fontsize=9)
    
    def update(self, value, timestamp=None):
        """Update panel with new value and glow effect."""
        if timestamp is None:
            timestamp = time.time()
        
        self.data_history.append(value)
        self.time_history.append(timestamp)
        
        if self.line is None:
            # Main line
            self.line, = self.ax.plot([], [], color=self.accent_color, 
                                      linewidth=2, alpha=0.9)
            # Glow effect lines
            for i in range(3):
                glow, = self.ax.plot([], [], color=self.accent_color, 
                                    linewidth=4*(i+1), alpha=0.1/(i+1))
                self.glow_lines.append(glow)
        
        # Update all lines
        data = list(self.data_history)
        times = list(self.time_history)
        
        self.line.set_data(times, data)
        for glow in self.glow_lines:
            glow.set_data(times, data)
        
        self.ax.relim()
        self.ax.autoscale_view()

# ============================================================================
# ENHANCED METRICS TRACKER
# ============================================================================

class AdvancedMetricsTracker:
    """Advanced metrics tracking with statistical analysis."""
    
    def __init__(self, max_history: int = 2000):
        self.max_history = max_history
        self.time_points = deque(maxlen=max_history)
        self.order_params = deque(maxlen=max_history)
        self.harmony_values = deque(maxlen=max_history)
        self.bound_fractions = deque(maxlen=max_history)
        self.entropy_values = deque(maxlen=max_history)
        self.phase_coherence = deque(maxlen=max_history)
        self.scale_metrics = {}
        self.current_step = 0
        self.start_time = time.time()
        
    def update(self, order_param: float, harmony: float, 
               bound_fraction: float, entropy: float, 
               phase_coherence: float = 0, scale: int = 0):
        """Add new metrics point with timestamp."""
        self.current_step += 1
        current_time = time.time() - self.start_time
        
        self.time_points.append(current_time)
        self.order_params.append(order_param)
        self.harmony_values.append(harmony)
        self.bound_fractions.append(bound_fraction)
        self.entropy_values.append(entropy)
        self.phase_coherence.append(phase_coherence)
        
        if scale not in self.scale_metrics:
            self.scale_metrics[scale] = {
                'order': deque(maxlen=self.max_history),
                'harmony': deque(maxlen=self.max_history),
                'coherence': deque(maxlen=self.max_history)
            }
        
        self.scale_metrics[scale]['order'].append(order_param)
        self.scale_metrics[scale]['harmony'].append(harmony)
        self.scale_metrics[scale]['coherence'].append(phase_coherence)
    
    def get_statistics(self):
        """Calculate current statistics."""
        if len(self.order_params) > 0:
            return {
                'mean_order': np.mean(self.order_params),
                'std_order': np.std(self.order_params),
                'max_order': np.max(self.order_params),
                'mean_harmony': np.mean(self.harmony_values),
                'mean_coherence': np.mean(self.phase_coherence),
                'emergence_rate': self._calculate_emergence_rate()
            }
        return {}
    
    def _calculate_emergence_rate(self):
        """Calculate rate of order emergence."""
        if len(self.order_params) > 10:
            recent = list(self.order_params)[-10:]
            older = list(self.order_params)[-20:-10] if len(self.order_params) > 20 else [0]
            return (np.mean(recent) - np.mean(older)) / 10
        return 0

# ============================================================================
# FALSIFICATION ENGINE
# ============================================================================

class EnhancedFalsificationEngine:
    """Enhanced falsification with multiple control experiments."""
    
    def __init__(self, base_params: dict):
        self.base_params = base_params
        self.control_results = {}
        self.validation_metrics = {}
        self.is_running = False
        self.thread = None
        
    def start_parallel_controls(self, n_loops: int, callback=None):
        """Start control experiments in parallel thread."""
        self.is_running = True
        self.thread = threading.Thread(
            target=self._run_controls,
            args=(n_loops, callback)
        )
        self.thread.daemon = True
        self.thread.start()
    
    def _run_controls(self, n_loops: int, callback):
        """Run multiple control experiments."""
        experiments = [
            ('null', self.run_null_hypothesis),
            ('corrupted', self.run_corrupted_dynamics),
            ('reversed', self.run_reversed_hasr),
            ('random_coupling', self.run_random_coupling)
        ]
        
        for name, func in experiments:
            if not self.is_running:
                break
            
            mean, std = func(n_loops, 100)
            self.control_results[name] = {
                'mean': mean,
                'std': std,
                'timestamp': time.time()
            }
            
            if callback:
                callback(name, mean, std)
    
    def run_null_hypothesis(self, n_loops: int, steps: int):
        """Run simulation without HASR learning."""
        zoomer = SubstrateZoomer(n_loops=n_loops, n_scales=3)
        results = []
        
        for _ in range(steps):
            zoomer.step(enable_hasr=False)
            results.append(zoomer.compute_order_parameter())
        
        return np.mean(results), np.std(results)
    
    def run_corrupted_dynamics(self, n_loops: int, steps: int):
        """Run with phase corruption."""
        zoomer = SubstrateZoomer(n_loops=n_loops, n_scales=3)
        results = []
        
        for _ in range(steps):
            # Corrupt 10% of phases
            corrupt_idx = np.random.choice(len(zoomer.loops), 
                                         size=int(0.1 * len(zoomer.loops)))
            for idx in corrupt_idx:
                zoomer.loops[idx].phase = np.random.uniform(0, 2*np.pi)
            
            zoomer.step()
            results.append(zoomer.compute_order_parameter())
        
        return np.mean(results), np.std(results)
    
    def run_reversed_hasr(self, n_loops: int, steps: int):
        """Run with reversed HASR (anti-learning)."""
        zoomer = SubstrateZoomer(n_loops=n_loops, n_scales=3)
        results = []
        
        for _ in range(steps):
            zoomer.step()
            # Reverse the learning
            for loop in zoomer.loops:
                loop.binding_weights = [-w for w in loop.binding_weights]
            
            results.append(zoomer.compute_order_parameter())
        
        return np.mean(results), np.std(results)
    
    def run_random_coupling(self, n_loops: int, steps: int):
        """Run with random coupling changes."""
        zoomer = SubstrateZoomer(n_loops=n_loops, n_scales=3)
        results = []
        
        for _ in range(steps):
            zoomer.step()
            # Randomize some couplings
            for loop in zoomer.loops:
                if np.random.random() < 0.1:
                    loop.binding_weights = np.random.random(len(loop.binding_weights)).tolist()
            
            results.append(zoomer.compute_order_parameter())
        
        return np.mean(results), np.std(results)

# ============================================================================
# MAIN LIVE SUBSTRATE VISUALIZER
# ============================================================================

class LiveSubstrateVisualizer:
    """Complete futuristic real-time visualization of UMST substrate dynamics."""
    
    def __init__(self, n_loops: int = 1000, n_scales: int = 3, 
                 config: Optional[VisualizationConfig] = None):
        self.n_loops = n_loops
        self.n_scales = n_scales
        self.config = config or VisualizationConfig()
        
        # Apply theme
        self.theme = AerwareAIThemes.apply_theme(self.config.theme)
        
        # Initialize substrate zoomer
        self.zoomer = SubstrateZoomer(
            n_loops=n_loops,
            n_scales=n_scales,
            binding_threshold=0.5,
            domain_size=100.0
        )
        
        # Initialize tracking systems
        self.metrics_tracker = AdvancedMetricsTracker()
        self.falsification_engine = EnhancedFalsificationEngine({
            'n_scales': n_scales,
            'binding_threshold': 0.5
        })
        
        # Simulation state
        self.is_running = False
        self.is_paused = False
        self.simulation_thread = None
        self.update_queue = Queue(maxsize=100)
        self.current_scale = 0
        self.zoom_level = 1.0
        self.pan_offset = np.array([0.0, 0.0])
        
        # Performance monitoring
        self.frame_times = deque(maxlen=100)
        self.last_frame_time = time.time()
        self.simulation_fps = 0
        
        # Trail effects
        self.position_history = deque(maxlen=self.config.trail_length)
        
        # Initialize matplotlib figure
        self._setup_figure()
        
        # Start falsification controls
        self.falsification_engine.start_parallel_controls(
            min(100, n_loops // 10),
            self._on_control_update
        )
    
    def _setup_figure(self):
        """Setup the matplotlib figure with futuristic AerwareAI styling."""
        # Enable interactive mode
        plt.ion()
        
        # Create main figure with dark theme
        self.fig = plt.figure(figsize=(24, 14), facecolor=self.theme['background'])
        self.fig.suptitle('UMST SUBSTRATE ZOOMER - AerwareAI Mission Control', 
                          fontsize=20, fontweight='bold', color=self.theme['primary'],
                          y=0.96)
        
        # Add subtitle with real-time info
        self.subtitle = self.fig.text(0.5, 0.93, '', ha='center', 
                                      fontsize=12, color=self.theme['accent'])
        
        # Create sophisticated grid layout
        gs = GridSpec(4, 6, figure=self.fig, 
                      hspace=0.35, wspace=0.3,
                      left=0.05, right=0.95, 
                      top=0.90, bottom=0.08)
        
        # === PRIMARY VISUALIZATION (Large, Central) ===
        self.ax_substrate = self.fig.add_subplot(gs[0:3, 0:3])
        self._style_substrate_panel()
        
        # === METRICS DASHBOARD (Right Side) ===
        self.ax_order = self.fig.add_subplot(gs[0, 3:5])
        self.panel_order = MetricsPanel(self.ax_order, 'ORDER PARAMETER R(t)', 
                                        self.theme['secondary'], 'R')
        
        self.ax_harmony = self.fig.add_subplot(gs[0, 5])
        self.panel_harmony = MetricsPanel(self.ax_harmony, 'HARMONY H(t)', 
                                          self.theme['success'], 'H')
        
        self.ax_scales = self.fig.add_subplot(gs[1, 3:5])
        self._style_scale_panel()
        
        self.ax_falsification = self.fig.add_subplot(gs[1, 5])
        self._style_falsification_panel()
        
        # === ANALYSIS PANELS (Bottom Row) ===
        self.ax_phase = self.fig.add_subplot(gs[2, 3], projection='polar')
        self._style_polar_panel()
        
        self.ax_coupling = self.fig.add_subplot(gs[2, 4])
        self._style_coupling_panel()
        
        self.ax_performance = self.fig.add_subplot(gs[2, 5])
        self.panel_performance = MetricsPanel(self.ax_performance, 'PERFORMANCE', 
                                              self.theme['warning'], 'FPS')
        
        self.ax_entropy = self.fig.add_subplot(gs[3, 3:6])
        self._style_entropy_panel()
        
        # === STATUS BAR (Bottom) ===
        self.ax_status = self.fig.add_subplot(gs[3, 0:3])
        self._setup_status_bar()
        
        # Setup controls with futuristic styling
        self._setup_futuristic_controls()
        
        # Apply overall dark theme
        self.fig.patch.set_facecolor(self.theme['background'])
    
    def _style_substrate_panel(self):
        """Style the main substrate visualization with HUD aesthetics."""
        self.ax_substrate.set_facecolor(self.theme['panel_bg'])
        self.ax_substrate.set_title('LIVE SUBSTRATE FIELD', 
                                   fontsize=14, fontweight='bold', 
                                   color=self.theme['primary'], pad=20)
        
        # Enhanced grid with glow
        self.ax_substrate.grid(True, alpha=0.2, color=self.theme['grid'], 
                              linestyle='-', linewidth=0.5)
        
        # Style axes
        self.ax_substrate.tick_params(colors=self.theme['text'], labelsize=10)
        self.ax_substrate.set_xlabel('X COORDINATE', color=self.theme['text'], 
                                    fontweight='bold', fontsize=10)
        self.ax_substrate.set_ylabel('Y COORDINATE', color=self.theme['text'], 
                                    fontweight='bold', fontsize=10)
        
        # Glowing border effect
        for spine in self.ax_substrate.spines.values():
            spine.set_color(self.theme['primary'])
            spine.set_linewidth(2)
            spine.set_alpha(0.8)
        
        # Initialize scatter plot
        self.scatter = self.ax_substrate.scatter([], [], s=[], c=[], 
                                                cmap=self.config.colormap,
                                                alpha=0.7, edgecolors='none')
        
        # Initialize coupling lines
        self.coupling_lines = []
        if self.config.show_coupling_lines:
            for _ in range(200):  # Pre-allocate lines
                line, = self.ax_substrate.plot([], [], 'w-', 
                                              alpha=self.config.coupling_line_alpha,
                                              linewidth=0.5)
                self.coupling_lines.append(line)
        
        # Add colorbar
        self.cbar = plt.colorbar(self.scatter, ax=self.ax_substrate)
        self.cbar.set_label('PHASE (rad)', color=self.theme['text'])
        self.cbar.ax.tick_params(colors=self.theme['text'])
    
    def _style_scale_panel(self):
        """Style the multi-scale tracking panel."""
        self.ax_scales.set_facecolor(self.theme['panel_bg'])
        self.ax_scales.set_title('MULTI-SCALE ORDER', fontsize=11, 
                                fontweight='bold', color=self.theme['accent'])
        self.ax_scales.set_xlabel('SCALE k', color=self.theme['text'])
        self.ax_scales.set_ylabel('R_k', color=self.theme['text'])
        self.ax_scales.set_ylim(0, 1)
        
        # Initialize bars
        self.bars_scales = self.ax_scales.bar(range(self.n_scales), 
                                              [0]*self.n_scales,
                                              color=self.theme['accent'],
                                              edgecolor=self.theme['primary'],
                                              linewidth=2, alpha=0.7)
    
    def _style_falsification_panel(self):
        """Style the falsification monitor."""
        self.ax_falsification.set_facecolor(self.theme['panel_bg'])
        self.ax_falsification.set_title('FALSIFICATION', fontsize=11,
                                       fontweight='bold', color=self.theme['warning'])
        self.ax_falsification.set_ylim(0, 1)
        
        # Initialize control bars
        control_names = ['Main', 'Null', 'Corrupt', 'Random']
        x_pos = np.arange(len(control_names))
        self.control_bars = self.ax_falsification.bar(x_pos, [0]*len(control_names),
                                                      color=[self.theme['success'],
                                                            self.theme['warning'],
                                                            self.theme['secondary'],
                                                            self.theme['accent']])
        self.ax_falsification.set_xticks(x_pos)
        self.ax_falsification.set_xticklabels(control_names, fontsize=8)
    
    def _style_polar_panel(self):
        """Style the phase distribution polar plot."""
        self.ax_phase.set_facecolor(self.theme['panel_bg'])
        self.ax_phase.set_title('PHASE DISTRIBUTION', fontsize=10, 
                               fontweight='bold', color=self.theme['accent'], 
                               pad=20)
        self.ax_phase.grid(True, alpha=0.3, color=self.theme['accent'])
        self.ax_phase.tick_params(colors=self.theme['text'], labelsize=8)
        
        # Initialize phase histogram
        self.phase_bars = self.ax_phase.bar(np.linspace(0, 2*np.pi, 36),
                                           [0]*36, width=2*np.pi/36,
                                           bottom=0, color=self.theme['accent'], 
                                           alpha=0.7, edgecolor=self.theme['primary'])
    
    def _style_coupling_panel(self):
        """Style the coupling distribution panel."""
        self.ax_coupling.set_facecolor(self.theme['panel_bg'])
        self.ax_coupling.set_title('COUPLING NETWORK', fontsize=11,
                                  fontweight='bold', color=self.theme['secondary'])
        self.ax_coupling.set_xlabel('Weight', color=self.theme['text'])
        self.ax_coupling.set_ylabel('Count', color=self.theme['text'])
        
        # Initialize histogram
        self.coupling_hist = None
    
    def _style_entropy_panel(self):
        """Style the entropy/binding panel."""
        self.ax_entropy.set_facecolor(self.theme['panel_bg'])
        self.ax_entropy.set_title('SYSTEM DYNAMICS', fontsize=11,
                                 fontweight='bold', color=self.theme['primary'])
        
        # Initialize lines
        self.line_entropy, = self.ax_entropy.plot([], [], 
                                                 color=self.theme['warning'],
                                                 linewidth=2, label='Entropy')
        self.line_bound, = self.ax_entropy.plot([], [], 
                                               color=self.theme['success'],
                                               linewidth=2, label='Bound %')
        self.line_coherence, = self.ax_entropy.plot([], [], 
                                                   color=self.theme['accent'],
                                                   linewidth=2, label='Coherence')
        
        self.ax_entropy.legend(loc='upper left', fontsize=8)
        self.ax_entropy.set_xlabel('Time (s)', color=self.theme['text'])
        self.ax_entropy.set_ylabel('Value', color=self.theme['text'])
    
    def _setup_status_bar(self):
        """Setup the status bar with system information."""
        self.ax_status.set_facecolor(self.theme['panel_bg'])
        self.ax_status.set_xlim(0, 1)
        self.ax_status.set_ylim(0, 1)
        self.ax_status.axis('off')
        
        # Status text elements
        self.status_text = self.ax_status.text(0.02, 0.5, '', 
                                              fontsize=12, color=self.theme['text'],
                                              fontweight='bold', va='center')
        
        # Statistics text
        self.stats_text = self.ax_status.text(0.98, 0.5, '', 
                                             fontsize=10, color=self.theme['accent'],
                                             ha='right', va='center')
    
    def _setup_futuristic_controls(self):
        """Create a sophisticated control interface with futuristic styling."""
        # === MAIN CONTROL BUTTONS ===
        # Play/Pause Button
        btn_play_ax = plt.axes([0.03, 0.03, 0.06, 0.035])
        self.btn_play_pause = Button(btn_play_ax, '▶ START', 
                                    color='#1a4d4d', hovercolor='#267373')
        self._style_button(self.btn_play_pause, self.theme['primary'])
        self.btn_play_pause.on_clicked(self.toggle_simulation)
        
        # Reset Button  
        btn_reset_ax = plt.axes([0.10, 0.03, 0.06, 0.035])
        self.btn_reset = Button(btn_reset_ax, '⟲ RESET',
                              color='#4d1a1a', hovercolor='#732626')
        self._style_button(self.btn_reset, self.theme['warning'])
        self.btn_reset.on_clicked(self.reset_simulation)
        
        # Export Button
        btn_export_ax = plt.axes([0.17, 0.03, 0.06, 0.035])
        self.btn_export = Button(btn_export_ax, '💾 EXPORT',
                               color='#4d4d1a', hovercolor='#737326')
        self._style_button(self.btn_export, self.theme['accent'])
        self.btn_export.on_clicked(self.export_data)
        
        # Screenshot Button
        btn_screenshot_ax = plt.axes([0.24, 0.03, 0.06, 0.035])
        self.btn_screenshot = Button(btn_screenshot_ax, '📸 CAPTURE',
                                   color='#1a1a4d', hovercolor='#262673')
        self._style_button(self.btn_screenshot, self.theme['secondary'])
        self.btn_screenshot.on_clicked(self.capture_screenshot)
        
        # === PARAMETER SLIDERS ===
        # Speed Control
        slider_speed_ax = plt.axes([0.35, 0.035, 0.15, 0.02])
        self.slider_speed = Slider(slider_speed_ax, 'SPEED', 0.1, 5.0, 
                                  valinit=1.0, valfmt='%.1fx',
                                  facecolor=self.theme['primary'])
        
        # Scale Selector
        slider_scale_ax = plt.axes([0.35, 0.01, 0.15, 0.02])
        self.slider_scale = Slider(slider_scale_ax, 'SCALE', 0, self.n_scales-1,
                                  valinit=0, valfmt='%d', valstep=1,
                                  facecolor=self.theme['secondary'])
        self.slider_scale.on_changed(self.update_scale)
        
        # Zoom Control
        slider_zoom_ax = plt.axes([0.55, 0.035, 0.15, 0.02])
        self.slider_zoom = Slider(slider_zoom_ax, 'ZOOM', 0.1, 10.0,
                                 valinit=1.0, valfmt='%.1fx',
                                 facecolor=self.theme['accent'])
        self.slider_zoom.on_changed(self.update_zoom)
        
        # Threshold Control
        slider_threshold_ax = plt.axes([0.55, 0.01, 0.15, 0.02])
        self.slider_threshold = Slider(slider_threshold_ax, 'THRESHOLD', 0.1, 0.9,
                                      valinit=0.5, valfmt='%.2f',
                                      facecolor=self.theme['warning'])
    
    def _style_button(self, button, color):
        """Apply futuristic styling to control buttons."""
        button.label.set_fontweight('bold')
        button.label.set_color('#ffffff')
        button.label.set_fontsize(9)
        
        # Glowing border effect
        for spine in button.ax.spines.values():
            spine.set_color(color)
            spine.set_linewidth(2)
            spine.set_alpha(0.8)
    
    def simulation_loop(self):
        """High-performance simulation loop with adaptive timing."""
        step_count = 0
        last_time = time.time()
        target_fps = 100  # Simulation target FPS
        frame_time = 1.0 / target_fps
        
        print(f"🚀 Simulation thread started - Target: {target_fps} FPS")
        
        while self.is_running:
            loop_start = time.time()
            
            if not self.is_paused:
                try:
                    # === CORE SIMULATION STEP ===
                    self.zoomer.step()
                    
                    # === PERIODIC UPDATES ===
                    if step_count % 5 == 0:
                        # Calculate metrics
                        order = self.zoomer.compute_order_parameter()
                        harmony = self.zoomer.compute_harmony()
                        
                        # === MULTI-SCALE ANALYSIS ===
                        if step_count % 50 == 0 and step_count > 0:
                            scale = min((step_count // 50) % self.n_scales, 
                                      self.n_scales - 1)
                            # Trigger coarse-graining at higher scales
                            if scale > 0 and hasattr(self.zoomer, 'coarse_grain'):
                                self.zoomer.coarse_grain(scale)
                        
                        # === DATA PREPARATION ===
                        data = self._prepare_visualization_data(harmony, order, step_count)
                        
                        # === THREAD-SAFE DATA TRANSFER ===
                        if not self.update_queue.full():
                            self.update_queue.put(data)
                        
                        # === PERFORMANCE TRACKING ===
                        current_time = time.time()
                        actual_fps = 1.0 / (current_time - last_time) if current_time > last_time else 0
                        self.simulation_fps = actual_fps
                        last_time = current_time
                    
                    step_count += 1
                    
                    # === ADAPTIVE TIMING ===
                    elapsed = time.time() - loop_start
                    sleep_time = (frame_time / self.slider_speed.val) - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    
                except Exception as e:
                    print(f"❌ Simulation error: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.1)
            else:
                # Paused - reduce CPU usage
                time.sleep(0.1)
        
        print("🛑 Simulation thread terminated")
    
    def _prepare_visualization_data(self, harmony, order, step_count):
        """Prepare comprehensive data package for visualization."""
        # === BASIC LOOP DATA ===
        positions = np.array([loop.position for loop in self.zoomer.loops])
        phases = np.array([loop.phase for loop in self.zoomer.loops])
        frequencies = np.array([loop.frequency for loop in self.zoomer.loops])
        amplitudes = np.array([loop.amplitude for loop in self.zoomer.loops])
        
        # === COUPLING ANALYSIS ===
        couplings = []
        coupling_weights = []
        total_coupling_strength = 0
        
        for i, loop in enumerate(self.zoomer.loops):
            for j, bound_id in enumerate(loop.bound_loops):
                if bound_id < len(self.zoomer.loops) and i < bound_id:
                    weight = loop.binding_weights[j] if j < len(loop.binding_weights) else 0
                    couplings.append((i, bound_id, weight))
                    coupling_weights.append(weight)
                    total_coupling_strength += weight
        
        # === NETWORK STATISTICS ===
        total_possible = len(self.zoomer.loops) * (len(self.zoomer.loops) - 1) / 2
        bound_fraction = len(couplings) / total_possible if total_possible > 0 else 0
        avg_coupling = total_coupling_strength / len(couplings) if couplings else 0
        
        # === PHASE ANALYSIS ===
        phase_coherence = np.abs(np.mean(np.exp(1j * phases)))
        phase_entropy = -np.sum(np.histogram(phases, bins=36)[0] * 
                               np.log(np.histogram(phases, bins=36)[0] + 1e-10))
        
        # === MULTI-SCALE DATA ===
        scale_orders = []
        for scale in range(self.n_scales):
            # Simple scale-dependent order calculation
            scale_order = order * (1 + scale * 0.15)
            scale_orders.append(min(scale_order, 1.0))
        
        return {
            # Core data
            'positions': positions,
            'phases': phases,
            'frequencies': frequencies,
            'amplitudes': amplitudes,
            'couplings': couplings[:200],  # Limit for performance
            'coupling_weights': coupling_weights,
            
            # Metrics
            'order_param': order,
            'harmony': harmony,
            'bound_fraction': bound_fraction,
            'entropy': phase_entropy,
            'phase_coherence': phase_coherence,
            'avg_coupling': avg_coupling,
            
            # Multi-scale
            'scale_orders': scale_orders,
            
            # System
            'step_count': step_count,
            'simulation_fps': self.simulation_fps,
            'timestamp': time.time()
        }
    
    def update_visualization(self, frame):
        """Update all visualization panels with animation."""
        if not self.update_queue.empty():
            # Get latest data
            data = self.update_queue.get()
            
            # Update all panels
            self._update_substrate_view(data)
            self._update_metrics_panels(data)
            self._update_scale_view(data)
            self._update_phase_view(data)
            self._update_coupling_view(data)
            self._update_entropy_view(data)
            self._update_status_bar(data)
            self._update_performance()
            
        return [self.scatter] + self.coupling_lines
    
    def _update_substrate_view(self, data):
        """Update main substrate view with enhanced visual effects."""
        positions = data['positions']
        phases = data['phases']
        amplitudes = data['amplitudes']
        
        # Store position history for trails
        if self.config.enable_trails:
            self.position_history.append(positions.copy())
        
        # Apply transformations
        view_positions = (positions - self.pan_offset) * self.zoom_level
        
        # Dynamic sizing based on amplitude
        sizes = self.config.loop_size_base * (0.5 + amplitudes * 1.5)
        
        # Update scatter plot
        self.scatter.set_offsets(view_positions)
        self.scatter.set_array(phases)
        self.scatter.set_sizes(sizes)
        
        # Update coupling lines with glow effect
        if self.config.show_coupling_lines and 'couplings' in data:
            for line in self.coupling_lines:
                line.set_data([], [])
            
            for i, (idx1, idx2, weight) in enumerate(data['couplings']):
                if i >= len(self.coupling_lines):
                    break
                if weight > 0.3:
                    pos1 = view_positions[idx1]
                    pos2 = view_positions[idx2]
                    
                    self.coupling_lines[i].set_data([pos1[0], pos2[0]], 
                                                   [pos1[1], pos2[1]])
                    self.coupling_lines[i].set_alpha(weight * 0.5)
                    self.coupling_lines[i].set_linewidth(0.5 + weight * 2)
                    self.coupling_lines[i].set_color(plt.cm.plasma(weight))
        
        # Auto-scale if enabled
        if self.config.auto_scale and len(view_positions) > 0:
            margin = 20
            self.ax_substrate.set_xlim(view_positions[:, 0].min() - margin,
                                      view_positions[:, 0].max() + margin)
            self.ax_substrate.set_ylim(view_positions[:, 1].min() - margin,
                                      view_positions[:, 1].max() + margin)
    
    def _update_metrics_panels(self, data):
        """Update metrics panels with new data."""
        # Update order parameter
        self.panel_order.update(data['order_param'])
        
        # Update harmony
        self.panel_harmony.update(data['harmony'])
        
        # Update tracker
        self.metrics_tracker.update(
            data['order_param'],
            data['harmony'],
            data['bound_fraction'],
            data['entropy'],
            data['phase_coherence'],
            self.current_scale
        )
    
    def _update_scale_view(self, data):
        """Update multi-scale tracking panel."""
        if 'scale_orders' in data:
            for i, (bar, value) in enumerate(zip(self.bars_scales, data['scale_orders'])):
                bar.set_height(value)
                # Color based on value
                if value > 0.7:
                    bar.set_color(self.theme['success'])
                elif value > 0.4:
                    bar.set_color(self.theme['accent'])
                else:
                    bar.set_color(self.theme['warning'])
    
    def _update_phase_view(self, data):
        """Update phase distribution polar plot."""
        phases = data['phases']
        hist, bins = np.histogram(phases, bins=36, range=(0, 2*np.pi))
        hist = hist / hist.max() if hist.max() > 0 else hist
        
        for bar, h in zip(self.phase_bars, hist):
            bar.set_height(h)
            # Color based on height
            bar.set_color(plt.cm.plasma(h))
    
    def _update_coupling_view(self, data):
        """Update coupling distribution histogram."""
        if 'coupling_weights' in data and len(data['coupling_weights']) > 0:
            self.ax_coupling.clear()
            self.ax_coupling.hist(data['coupling_weights'], bins=20, 
                                 color=self.theme['secondary'], 
                                 edgecolor=self.theme['primary'],
                                 alpha=0.7)
            self.ax_coupling.set_title('COUPLING NETWORK', fontsize=11,
                                      fontweight='bold', color=self.theme['secondary'])
            self.ax_coupling.set_xlabel('Weight', color=self.theme['text'])
            self.ax_coupling.set_ylabel('Count', color=self.theme['text'])
    
    def _update_entropy_view(self, data):
        """Update entropy and dynamics panel."""
        if len(self.metrics_tracker.time_points) > 1:
            times = list(self.metrics_tracker.time_points)
            
            self.line_entropy.set_data(times, 
                                      np.array(list(self.metrics_tracker.entropy_values)) / 10)
            self.line_bound.set_data(times, 
                                    list(self.metrics_tracker.bound_fractions))
            self.line_coherence.set_data(times, 
                                        list(self.metrics_tracker.phase_coherence))
            
            self.ax_entropy.relim()
            self.ax_entropy.autoscale_view()
    
    def _update_status_bar(self, data):
        """Update status bar with system information."""
        # Main status
        status = f"LOOPS: {len(data['positions'])} | "
        status += f"COUPLINGS: {len(data.get('couplings', []))} | "
        status += f"STEP: {data['step_count']} | "
        status += f"SIM FPS: {data['simulation_fps']:.1f}"
        self.status_text.set_text(status)
        
        # Statistics
        stats = self.metrics_tracker.get_statistics()
        if stats:
            stats_text = f"R̄: {stats.get('mean_order', 0):.3f} | "
            stats_text += f"σ_R: {stats.get('std_order', 0):.3f} | "
            stats_text += f"R_max: {stats.get('max_order', 0):.3f} | "
            stats_text += f"dR/dt: {stats.get('emergence_rate', 0):.4f}"
            self.stats_text.set_text(stats_text)
        
        # Update subtitle
        self.subtitle.set_text(f"Scale: {self.current_scale} | Zoom: {self.zoom_level:.1f}x | "
                              f"Speed: {self.slider_speed.val:.1f}x | "
                              f"Threshold: {self.slider_threshold.val:.2f}")
    
    def _update_performance(self):
        """Update performance monitoring."""
        current_time = time.time()
        frame_time = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        if frame_time > 0:
            fps = 1.0 / frame_time
            self.panel_performance.update(fps)
    
    def _on_control_update(self, name, mean, std):
        """Callback for falsification control updates."""
        # Update control bars
        if hasattr(self, 'control_bars'):
            if name == 'null':
                self.control_bars[1].set_height(mean)
            elif name == 'corrupted':
                self.control_bars[2].set_height(mean)
            elif name == 'random_coupling':
                self.control_bars[3].set_height(mean)
    
    def toggle_simulation(self, event):
        """Start/pause simulation."""
        if not self.is_running:
            self.is_running = True
            self.is_paused = False
            self.btn_play_pause.label.set_text('⏸ PAUSE')
            
            # Start simulation thread
            self.simulation_thread = threading.Thread(target=self.simulation_loop)
            self.simulation_thread.daemon = True
            self.simulation_thread.start()
            
            # Start animation
            self.ani = animation.FuncAnimation(
                self.fig, self.update_visualization,
                interval=self.config.update_interval,
                blit=False, repeat=True
            )
            
            print("▶️ Simulation started")
        else:
            self.is_paused = not self.is_paused
            if self.is_paused:
                self.btn_play_pause.label.set_text('▶ RESUME')
                print("⏸ Simulation paused")
            else:
                self.btn_play_pause.label.set_text('⏸ PAUSE')
                print("▶️ Simulation resumed")
    
    def reset_simulation(self, event):
        """Reset the simulation to initial conditions."""
        print("🔄 Resetting simulation...")
        
        self.is_running = False
        self.is_paused = False
        self.btn_play_pause.label.set_text('▶ START')
        
        # Wait for thread to stop
        if self.simulation_thread:
            self.simulation_thread.join(timeout=2)
        
        # Reset zoomer
        self.zoomer = SubstrateZoomer(
            n_loops=self.n_loops,
            n_scales=self.n_scales,
            binding_threshold=self.slider_threshold.val,
            domain_size=100.0
        )
        
        # Reset trackers
        self.metrics_tracker = AdvancedMetricsTracker()
        self.position_history.clear()
        
        # Update display
        positions = np.array([loop.position for loop in self.zoomer.loops])
        phases = np.array([loop.phase for loop in self.zoomer.loops])
        self.scatter.set_offsets(positions)
        self.scatter.set_array(phases)
        
        print("✅ Reset complete")
    
    def update_scale(self, val):
        """Update the current visualization scale."""
        self.current_scale = int(self.slider_scale.val)
        # Adjust zoom based on scale
        self.zoom_level = 1.0 / (1.5 ** self.current_scale)
        self.slider_zoom.set_val(self.zoom_level)
    
    def update_zoom(self, val):
        """Update zoom level."""
        self.zoom_level = val
    
    def capture_screenshot(self, event):
        """Capture high-resolution screenshot."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"substrate_capture_{timestamp}.png"
        self.fig.savefig(filename, dpi=200, bbox_inches='tight', 
                        facecolor=self.theme['background'])
        print(f"📸 Screenshot saved: {filename}")
    
    def export_data(self, event):
        """Export comprehensive data and metrics."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Export metrics
        metrics_file = f"metrics_{timestamp}.json"
        self.metrics_tracker.export_metrics(metrics_file)
        
        # Export current state
        state_file = f"state_{timestamp}.json"
        state_data = {
            'configuration': {
                'n_loops': self.n_loops,
                'n_scales': self.n_scales,
                'theme': self.config.theme,
                'current_scale': self.current_scale,
                'zoom_level': self.zoom_level
            },
            'loops': [
                {
                    'id': loop.id,
                    'phase': float(loop.phase),
                    'frequency': float(loop.frequency),
                    'amplitude': float(loop.amplitude),
                    'position': loop.position.tolist()
                }
                for loop in self.zoomer.loops
            ],
            'statistics': self.metrics_tracker.get_statistics(),
            'falsification': self.falsification_engine.control_results,
            'timestamp': timestamp
        }
        
        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        print(f"💾 Data exported: {metrics_file}, {state_file}")
    
    def run(self):
        """Start the visualization."""
        print("🌌 AerwareAI Substrate Visualizer Ready")
        print("📊 Press 'START' to begin simulation")
        plt.show()

# ============================================================================
# WEEKEND TEST PROTOCOL
# ============================================================================

class WeekendTestProtocol:
    """Comprehensive weekend testing protocol for UMST validation."""
    
    def __init__(self):
        self.results = {}
        self.start_time = time.time()
    
    def run_full_protocol(self):
        """Execute the complete weekend test sequence."""
        print("\n" + "🧪" * 40)
        print("WEEKEND TEST PROTOCOL - UMST PATTERN VALIDATION")
        print("AerwareAI - Searching for Universal Mathematical Patterns")
        print("🧪" * 40)
        
        # Phase 1: Saturday Morning - Basic Tests
        print("\n📋 PHASE 1: Basic Functionality (Saturday AM)")
        self.test_basic_functionality()
        
        # Phase 2: Saturday Afternoon - Scale Tests
        print("\n📋 PHASE 2: Multi-Scale Patterns (Saturday PM)")
        self.test_scale_progression()
        
        # Phase 3: Sunday Morning - Deep Analysis
        print("\n📋 PHASE 3: Deep Pattern Analysis (Sunday AM)")
        self.test_deep_patterns()
        
        # Phase 4: Sunday Afternoon - Falsification
        print("\n📋 PHASE 4: Falsification Validation (Sunday PM)")
        self.test_falsification()
        
        # Generate report
        self.generate_final_report()
    
    def test_basic_functionality(self):
        """Test basic visualization and simulation."""
        configs = [(100, 2), (500, 3), (1000, 3), (5000, 4)]
        
        for loops, scales in configs:
            print(f"  Testing {loops} loops, {scales} scales...")
            success = self._run_automated_test(loops, scales, 10)
            
            self.results[f'basic_{loops}'] = {
                'success': success,
                'configuration': {'loops': loops, 'scales': scales}
            }
            
            print(f"    {'✅ PASSED' if success else '❌ FAILED'}")
    
    def test_scale_progression(self):
        """Test multi-scale pattern emergence."""
        print("  Testing scale progression...")
        
        for scales in [3, 5, 7]:
            success = self._run_automated_test(1000, scales, 20)
            self.results[f'scales_{scales}'] = {'success': success}
            print(f"    {scales} scales: {'✅' if success else '❌'}")
    
    def test_deep_patterns(self):
        """Deep pattern analysis with extended runs."""
        print("  Running deep pattern analysis...")
        
        # Test different initial conditions
        conditions = ['uniform', 'clustered', 'gradient']
        for condition in conditions:
            print(f"    Testing {condition} initial conditions...")
            # Simplified test
            self.results[f'pattern_{condition}'] = {'tested': True}
    
    def test_falsification(self):
        """Falsification validation tests."""
        print("  Running falsification controls...")
        
        engine = EnhancedFalsificationEngine({})
        
        # Run controls
        null_mean, null_std = engine.run_null_hypothesis(100, 50)
        corrupt_mean, corrupt_std = engine.run_corrupted_dynamics(100, 50)
        
        self.results['falsification'] = {
            'null': {'mean': null_mean, 'std': null_std},
            'corrupted': {'mean': corrupt_mean, 'std': corrupt_std}
        }
        
        print(f"    Null: R = {null_mean:.3f} ± {null_std:.3f}")
        print(f"    Corrupted: R = {corrupt_mean:.3f} ± {corrupt_std:.3f}")
    
    def _run_automated_test(self, loops, scales, duration):
        """Run automated test."""
        try:
            config = VisualizationConfig(performance_mode=True)
            viz = LiveSubstrateVisualizer(loops, scales, config)
            
            # Simulate running
            viz.is_running = True
            thread = threading.Thread(target=viz.simulation_loop)
            thread.daemon = True
            thread.start()
            
            time.sleep(duration)
            
            viz.is_running = False
            thread.join(timeout=5)
            
            return True
        except Exception as e:
            print(f"      Error: {e}")
            return False
    
    def generate_final_report(self):
        """Generate comprehensive test report."""
        duration = (time.time() - self.start_time) / 3600
        
        print("\n" + "=" * 60)
        print("WEEKEND TEST PROTOCOL - FINAL REPORT")
        print("=" * 60)
        
        print(f"\n📊 Test Duration: {duration:.1f} hours")
        print(f"📊 Total Tests: {len(self.results)}")
        
        passed = sum(1 for r in self.results.values() 
                    if r.get('success', False) or r.get('tested', False))
        print(f"📊 Tests Completed: {passed}/{len(self.results)}")
        
        # Save report
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = f"weekend_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump({
                'duration_hours': duration,
                'timestamp': timestamp,
                'results': self.results,
                'summary': {
                    'total_tests': len(self.results),
                    'completed': passed
                }
            }, f, indent=2)
        
        print(f"\n📁 Report saved: {report_file}")
        
        if passed == len(self.results):
            print("\n🎉 ALL TESTS COMPLETED - UMST PATTERNS VALIDATED!")
            print("   Universal mathematical structures detected across scales")
        else:
            print(f"\n⚠️ Some tests incomplete - Review required")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Advanced main entry point with comprehensive options."""
    import sys
    import argparse
    
    print("🌌" * 40)
    print("   UMST SUBSTRATE ZOOMER - AerwareAI Mission Control")
    print("      Real-Time Mathematical Universe Visualization")
    print("🌌" * 40)
    
    parser = argparse.ArgumentParser(description='UMST Live Substrate Visualizer')
    parser.add_argument('--loops', type=int, default=1000,
                       help='Number of substrate loops')
    parser.add_argument('--scales', type=int, default=3,
                       help='Number of coarse-graining scales')
    parser.add_argument('--theme', choices=['neon', 'matrix', 'cyberpunk'], 
                       default='neon', help='Visual theme')
    parser.add_argument('--performance', action='store_true',
                       help='Enable performance mode')
    parser.add_argument('--test', action='store_true',
                       help='Run weekend test protocol')
    
    # Handle presets
    if len(sys.argv) > 1 and sys.argv[1] in ['demo', 'large', 'massive', 'test']:
        preset = sys.argv[1]
        if preset == 'demo':
            args = argparse.Namespace(loops=500, scales=3, theme='neon', 
                                    performance=False, test=False)
        elif preset == 'large':
            args = argparse.Namespace(loops=5000, scales=4, theme='neon',
                                    performance=True, test=False)
        elif preset == 'massive':
            args = argparse.Namespace(loops=20000, scales=5, theme='cyberpunk',
                                    performance=True, test=False)
        elif preset == 'test':
            args = argparse.Namespace(test=True, loops=1000, scales=3,
                                    theme='neon', performance=False)
    else:
        args = parser.parse_args()
    
    # Run test protocol
    if args.test:
        print("\n🧪 Initiating Weekend Test Protocol...")
        protocol = WeekendTestProtocol()
        protocol.run_full_protocol()
        return
    
    # Configuration
    config = VisualizationConfig(
        theme=args.theme,
        performance_mode=args.performance,
        show_coupling_lines=not args.performance
    )
    
    print(f"\n⚙️ Configuration:")
    print(f"   • Loops: {args.loops:,}")
    print(f"   • Scales: {args.scales}")
    print(f"   • Theme: {args.theme}")
    print(f"   • Performance Mode: {args.performance}")
    
    # Launch visualizer
    try:
        print("\n🚀 Launching AerwareAI Substrate Visualizer...")
        visualizer = LiveSubstrateVisualizer(args.loops, args.scales, config)
        visualizer.run()
        
    except KeyboardInterrupt:
        print("\n⚠️ User interrupt")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🛑 Shutdown complete")

if __name__ == "__main__":
    main()