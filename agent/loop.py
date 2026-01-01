from sim.state_extractor import extract_symbolic
from collections import deque
import numpy as np
from skills.primitives import stop


class AgentLoop:
    """
    Day3: Observe -> Policy -> Act -> Log
    No YAML, no printing, no “policy logic” inside.
    """

    def __init__(self, env, policy, logger):
        self.env = env 
        self.policy = policy
        self.logger = logger
        self.pos_history = deque(maxlen=25)
        self.stuck_steps = 0
        self.reached_target = False


    def run_episode(self, episode_id: int):
        obs = self.env.reset()

        # reset episode-level memory
        self.pos_history.clear()
        self.stuck_steps = 0
        self.pos_history.clear()
        self.stuck_steps = 0
        self.reached_target = False

        collisions = 0
        last_info = {}
        reason = "max_steps"
        success = False

        for step in range(self.env.max_steps):

            # perception
            state = extract_symbolic(
                obs,
                obstacle_ids=self.env.obstacle_ids,
                robot_id=self.env.robot_id,
            )

            # inject physics-level perception from env
            state["contact"] = bool(last_info.get("contact", False))
            state["contact_with_obstacle"] = bool(
                last_info.get("contact_with_obstacle", False)
            )
            state["contact_normal_xy"] = last_info.get(
                "contact_normal_xy", [0.0, 0.0]
            )

            # executive: stuck detection
            pos = np.array(obs["robot"]["pos"][:2])
            self.pos_history.append(pos)

            is_stuck = False
            if len(self.pos_history) == self.pos_history.maxlen:
                disp = np.linalg.norm(self.pos_history[-1] - self.pos_history[0])

                if disp < 0.03:      # ~3 cm
                    self.stuck_steps += 1
                else:
                    self.stuck_steps = 0

                if self.stuck_steps > 10:
                    is_stuck = True

            state["stuck"] = is_stuck
            
            # terminal latch
            if state["target_dist"] < 0.10:
                self.reached_target = True

            # policy / terminal override
            if self.reached_target:
                print("\nTarget Reached!")
                action = stop()
            else:
                action = self.policy.select_action(state)


            # act 
            obs, reward, done, info = self.env.step(action)
            last_info = info

            # success condition 
            if self.reached_target:
                success = True
                reason = "at_target"
                done = True


            # logging
            self.logger.log_step(
                episode_id=episode_id,
                step=step,
                state=state,
                action=action,
                reward=float(reward) if reward is not None else 0.0,
                done=bool(done),
                info=info,
            )

            if step % 200 == 0:
                print(
                    f"[DBG] step={step} "
                    f"dist={state['target_dist']:.3f} "
                    f"bearing={state['target_bearing']:+.2f} "
                    f"stuck={state['stuck']} "
                    f"at_target={state['at_target']}"
                )

            if done:
                break

        self.logger.log_episode_end(
            episode_id=episode_id,
            success=success,
            steps=step + 1,
            collisions=collisions,
            reason=reason,
            info_last=last_info,
        )

        return {
            "success": success,
            "steps": step + 1,
            "collisions": collisions,
            "reason": reason,
        }
