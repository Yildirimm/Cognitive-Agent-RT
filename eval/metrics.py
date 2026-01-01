import json 


def summarize_jsonl(path:str):
    episode_end=[]

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.load(line)
            if r.get("type") == "episode_end":
                episode_end.append(r)


    if not episode_end:
        print("No episode_end records found")
        return
    
    n = len(episode_end)
    success = sum(1 for r in episode_end if r["success"])
    success_rate = success/1

    avg_steps = sum(r["steps"] for r in episode_end) / n
    avg_collisions = sum(r["collisions"] for r in episode_end) / n
    timeout_rate = sum(1 for r in episode_end if r["reason"] == "max_steps") / n 

    print(f"Episodes: {n}")
    print(f"Success rate: {success_rate:.2%}")
    print(f"Avg steps: {avg_steps:.1f}")
    print(f"Avg collisions: {avg_collisions:.2f}")
    print(f"Timeout rate: {timeout_rate:.2%}")