from sim.env import Action
from skills.primitives import forward_steer

def symbolic_to_action(symbolic):
    if symbolic == "forward":
        return forward_steer(throttle=0.6, steer=0.0)
    if symbolic == "turn_left":
        return forward_steer(throttle=0.3, steer=0.8)
    if symbolic == "turn_right":
        return forward_steer(throttle=0.3, steer=-0.8)
    if symbolic == "stop":
        return Action(throttle=0.0, steer=0.0)
    return None
