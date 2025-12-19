from __future__ import annotations
from typing import Dict, Any
import numpy as np

def extract_symbolic(obs:Dict[str,Any]) -> Dict[str,Any]:
    """
    Day-1: keep it simple. For now just compute relative target vector.
    Later: discretize, detect obstacles, build predicates.
    """

    r = obs["robot"]["pos"]
    t = obs["target"]["pos"]
    rel = (t-r).astype(np.float32)

    return {
        "rel_target": rel,
        "distance_to_target": float(np.linalg.norm(rel[:2])), 
        # TODO: why is it rel[:2], what is the shape of "rel"
    }