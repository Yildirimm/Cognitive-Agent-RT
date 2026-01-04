from sim.state_extractor import extract_symbolic
from collections import deque
import numpy as np
from skills.primitives import stop
from planner.llm_planner import choose_action_llm
from planner.action_gate import symbolic_to_action
from planner.dummy_llm import dummy_llm_client

class AgentLoop:
    """
    Day3: Observe -> Policy -> Act -> Log
    No YAML, no printing, no “policy logic” inside.
    """

    def __init__(self, env, policy, logger, llm_client):
        self.env = env 
        self.policy = policy
        self.logger = logger
        self.llm_client = llm_client # dummy_llm_client
        self.use_llm = getattr(env, "use_llm", False)
        self.pos_history = deque(maxlen=25)
        self.stuck_steps = 0
        self.reached_target = False


    def run_episode(self, episode_id: int):
        obs = self.env.reset()

        # reset episode-level memory
        self.pos_history.clear()
        self.stuck_steps = 0
        self.reached_target = False

        collisions = 0
        last_info = {}
        reason = "max_steps"
        success = False

        for step in range(self.env.max_steps):
            symbolic_action = None
            fallback_used = False

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
                if self.use_llm and self.llm_client is not None and step%5==0: # or use 3
                    symbolic_action = choose_action_llm(state, self.llm_client)
                else: 
                    symbolic_action=None

                # compute fallback action 
                action = self.policy.select_action(state)

                # LLM override
                if self.policy.recovery_steps == 0 and symbolic_action is not None:
                    gated = symbolic_to_action(symbolic_action)

                    if gated is not None:
                        action = gated
                    else:
                        fallback_used = True

                else:
                    if self.use_llm:
                        if self.policy.recovery_steps > 0: # fallback due to recovery
                            fallback_used = True
                        elif symbolic_action is None: # fallback due to LLM
                            fallback_used = True


            # act 
            obs, reward, done, info = self.env.step(action)
            last_info = info
            
            # update collision, if any
            if info.get("collision", False):
                collisions += 1

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

            info={
                **info,
                "symbolic_action": symbolic_action,
                "fallback_used": fallback_used,
                },
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
