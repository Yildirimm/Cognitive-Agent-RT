from sim.env import Action
from skills.primitives import forward_steer, approach_target


class FallbackPolicy:
    def __init__(self):
        self.recovery_steps = 0
        self.turn_dir = 1.0

    def select_action(self, state) -> Action:

        # trigger recovery (edge-triggered)
        if self.recovery_steps == 0:
            if state.get("stuck", False) or state.get("contact_with_obstacle", False):
                self.recovery_steps = 50

                # choose turning direction from contact normal
                nx, ny = state.get("contact_normal_xy", [0.0, 0.0])
                self.turn_dir = -1.0 if ny > 0 else +1.0

        # recovery sequence
        if self.recovery_steps > 0:
            self.recovery_steps -= 1

            # reverse phase
            if self.recovery_steps > 30:
                return forward_steer(throttle=-0.6, steer=0.0)

            # turn + creep phase
            return forward_steer(throttle=0.3, steer=0.8 * self.turn_dir)

        # normal behavior
        return approach_target(state)
