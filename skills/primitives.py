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


def approach_target(state):
    # hard stop zone
    if state["target_dist"] < 0.10:
        return stop()

    # proportional steering
    b = state["target_bearing"]
    steer = np.clip(1.5 * b, -1.0, 1.0)

    # distance-based throttle
    d = state["target_dist"]

    if d > 1.0:
        throttle = 0.6
    elif d > 0.4:
        throttle = 0.3
    else:
        throttle = 0.15

    return forward_steer(throttle=throttle, steer=steer)