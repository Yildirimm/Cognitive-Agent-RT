import json
from pathlib import Path
from typing import Any, Dict

class JsonLogger:
    def __init__(self, path: str = "logs/day3_baseline.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)


    def _write(self, record: Dict[str, Any]):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


    def log_step(
            self,
            episode_id: int,
            step: int,
            state: Dict[str, Any],
            action: Any,
            reward: float,
            done: bool,
            info: Dict[str, Any],
        ):


        self._write({
            "type": "step",
            "episode_id": episode_id,
            "step": step,
            "state": state,
            "action": str(action),
            "reward": float(reward) if reward is not None else 0.0,
            "done": bool(done),
            "info": info,
        })

    def log_episode_end(
            self,
            episode_id: int,
            success: bool,
            steps: int,
            collisions: int,
            reason: str,
            info_last: Dict[str, Any],
        ):

        
        self._write({
            "type": "episode_end",
            "episode_id": episode_id,
            "success": bool(success),
            "steps": int(steps),
            "collisions": int(collisions),
            "reason": reason,
            "info_last": info_last,
        })
