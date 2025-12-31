from sim.env import BulletEnv, Action
from sim.state_extractor import extract_symbolic
from skills.primitives import approach_target
import numpy as np

def main():
    env = BulletEnv("configs/task.yaml", "configs/timing.yaml")
    obs = env.reset()

    for i in range(env.max_steps):
        s = extract_symbolic( 
            obs, 
            obstacle_ids=env.obstacle_ids,
            robot_id=env.robot_id
        )
        
        if i == 0:
            print("[DEBUG] symbolic keys:", s.keys())
            print("[DEBUG] symbolic state:", s)
            print("[DEBUG] raw robot pos:", obs["robot"]["pos"])
            print("[DEBUG] raw target pos:", obs["target"]["pos"])

        action = approach_target(s)
        obs, _, done, info = env.step(action)

        

        speed = float(np.linalg.norm(obs["robot"]["lin_vel"][:2]))

        if i % 25 == 0:
            print(
                f"step={i:4d} "
                f"dist={s['target_dist']:.2f} "
                f"bear={s['target_bearing']:+.2f} "
                f"near_ob={s['nearest_obstacle_dist']:.2f} "
                f"speed={speed:.2f} "
                f"collision={info.get('collision')}"
            )


        if s["at_target"]:
            print("Reached target")
            break
        if done:
            print("Max steps reached.")
            break

    env.close()

if __name__ == "__main__":
    main()