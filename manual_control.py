from __future__ import annotations
import argparse
import pybullet as p
from sim.env import BulletEnv, Action

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="configs/task.yaml")
    ap.add_argument("--timing", default="configs/timing.yaml")
    args = ap.parse_args()

    env = BulletEnv(task_cfg_path=args.task, timing_cfg_path=args.timing)
    # env.connect()
    env.reset()

    throttle = 0.0 
    steer = 0.0 

    print("Manual control:")
    print("  W/S = throttle +/-")
    print("  A/D = steer left/right")
    print("  Space = brake (throttle=0)")
    print("  R = reset")
    print("  Q = quit")

    while True:
        keys = p.getKeyboardEvents()

        def down(k):
            return (k in keys) and (keys[k] & p.KEY_IS_DOWN)
        
        if down(ord('q')) or down(ord("Q")):
            break 

        if down(ord("r")) or down(ord("R")):
            env.reset()
            throttle, steer = 0.0, 0.0
            continue

        # update controls
        w = down(ord("w")) or down(ord("W"))
        s = down(ord("s")) or down(ord("S"))
        a = down(ord("a")) or down(ord("A"))
        d = down(ord("d")) or down(ord("D"))

        if w:
            throttle = min(1.0, throttle + 0.05)
        elif s:
            throttle = max(-1.0, throttle - 0.05)
        else:
            throttle *= 0.85  # <-- decay to zero when no throttle keys 

        if a:
            steer = max(-1.0, steer - 0.07)
        elif d:
            steer = min(1.0, steer + 0.07)
        else:
            steer *= 0.80     # <-- steering return-to-center

        # hard brake / full stop 
        if down(p.B3G_SPACE):
            throttle = 0.0


        obs, _, done, info = env.step(Action(throttle=throttle, steer=steer))
    
        if env.step_count % 10 == 0:
            v = obs["robot"]["lin_vel"]
            speed = (v[0]**2 + v[1]**2) ** 0.5
            print(
                f"throttle={throttle:+.2f} "
                f"steer={steer:+.2f} "
                f"speed={speed:.2f} m/s"
            )


        # simple on-screen debug
        if info["collision"]:
            print("COLLISION")

        if done:
            print("Episode done → resetting")
            env.reset()
            throttle, steer = 0.0, 0.0

    env.close()


if __name__ == "__main__":
    main()