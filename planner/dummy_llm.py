import json
from typing import Dict


def dummy_llm_client(prompt: Dict[str, str]) -> str:
    user_text = prompt["user"]

    # dumb but deterministic
    if "at_target': True" in user_text or '"at_target": True' in user_text:
        return json.dumps({"action": "stop"})

    if "obstacle_front': 0.2" in user_text or '"obstacle_front": 0.2' in user_text:
        return json.dumps({"action": "turn_left"})

    return json.dumps({"action": "forward"})
