from sim.env import BulletEnv
from planner.fallback_policy import FallbackPolicy
from agent.loop import AgentLoop
from eval.logger import JsonLogger
from eval.metrics import summarize_jsonl
from planner.dummy_llm import dummy_llm_client
from planner.gemini_llm import gemini_llm_client

def main():
    env = BulletEnv("configs/task.yaml", "configs/timing.yaml")
    
    policy = FallbackPolicy()
    log_path = "logs/day4_baseline.jsonl"
    logger = JsonLogger(log_path)

    llm_client = dummy_llm_client
    # agent = AgentLoop(env, policy, logger, llm_client=llm_client)
    agent = AgentLoop(env, policy, logger, llm_client=gemini_llm_client)

    num_episodes = 10
    for ep in range(num_episodes):
        result = agent.run_episode(ep)
        print(f"[EP {ep}] success={result['success']} steps={result['steps']} coll={result['collisions']} reason={result['reason']}")

    env.close()

    print("\n=== METRICS ===")
    summarize_jsonl(log_path)


if __name__=="__main__":
    main()