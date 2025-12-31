from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
from sim.env import Action
import numpy as np 



def forward(throttle:float=0.6) -> Action: 
    return Action(throttle=throttle, steer=0.0)

def turn_left(throttle:float=0.25, steer:float=0.9) -> Action:
    return Action(throttle=throttle, steer=+steer)

def turn_right(throttle:float=0.25, steer:float=0.9) -> Action:
    return Action(throttle=throttle, steer=-steer)

def stop() -> Action: 
    return Action(throttle=0.0, steer=0.0)

def forward_steer(throttle: float, steer: float) -> Action:
    """
    Drive forward while steering proportionally.
    steer is expected in [-1, +1].
    """
    return Action(
        throttle=throttle,
        steer=float(np.clip(steer, -1.0, 1.0))
    )

def approach_target(
        state: Dict[str, Any],
        *,
        d_safe: float = 0.50,
        bearing_thresh: float = 0.25
        ) -> Action:

    """
    Macro: reach target using only symbolic state + primitives
    This is Day-2 scripted controller, no LLM involved
    """

    if state["at_target"]:
        return stop()
    
    # Safety first
    if state["nearest_obstacle_dist"] < d_safe:
        # turn away from the closer side
        if state["left_dist"] < state["right_dist"]:
            return turn_right()
        else:
            return turn_left()
        

    # Smooth steering toward target
    b = state["target_bearing"]
    # proportional steering toward target
    steer_gain = 2.0        # tuning knob
    steer_cmd = steer_gain * b

    return forward_steer(throttle=0.6, steer=steer_cmd)
