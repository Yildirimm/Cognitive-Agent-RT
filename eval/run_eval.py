from sim.env import BulletEnv
from planner.fallback_policy import FallbackPolicy
from agent.loop import AgentLoop
from eval.logger import JsonLogger
from eval.metrics import summarize_jsonl

def main():
    env = BulletEnv("configs/task.yaml", "configs/timing.yaml")
    
    policy = FallbackPolicy()
    log_path = "logs/day3_baseline.jsonl"
    logger = JsonLogger(log_path)

    agent = AgentLoop(env, policy, logger)

    num_episodes = 10
    for ep in range(num_episodes):
        result = agent.run_episode(ep)
        print(f"[EP {ep}] success={result['success']} steps={result['steps']} coll={result['collisions']} reason={result['reason']}")

    env.close()

    print("\n=== METRICS ===")
    summarize_jsonl(log_path)


if __name__=="__main__":
    main()