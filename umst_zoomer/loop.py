from dataclasses import dataclass
from typing import Dict
import numpy as np

@dataclass
class Loop:
    id: int
    phi: float       # phase (radians)
    omega: float     # frequency
    amp: float       # amplitude
    payload: Dict    # symbolic payload
    t0: float
    t1: float
    pos: np.ndarray  # R^d

@dataclass
class Edge:
    i: int
    j: int
    w: float         # coupling weight
