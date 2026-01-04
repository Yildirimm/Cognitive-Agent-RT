import json
from typing import Dict, Any, Optional, Callable
from planner.prompts import ALLOWED_ACTIONS, build_prompt


SymbolicState = Dict[str, Any]
SymbolicAction = Optional[str]
LLMClient = Callable[[dict], str]


def build_symbolic_state(state: Dict[str, Any]) -> SymbolicState:

    return {
        "target_distance": round(state["target_dist"], 2),
        "target_bearing": round(state["target_bearing"], 2),
        "obstacle_front": round(state["front_dist"], 2),
        "obstacle_left": round(state["left_dist"], 2),
        "obstacle_right": round(state["right_dist"], 2),
        "at_target": bool(state["at_target"]),
        }


def parse_llm_output(text: str) -> SymbolicAction:
    try:
        data = json.loads(text)
        action = data.get("action")

        if action not in ALLOWED_ACTIONS:
            return None

        return action

    except Exception:
        return None


def choose_action_llm(
        state: Dict[str, Any],
        llm_client: LLMClient
        ) -> SymbolicAction:
    
    symbolic = build_symbolic_state(state)
    prompt = build_prompt(symbolic)

    # print("[LLM NAME]", llm_client.__name__)

    try:
        raw = llm_client(prompt)
    except Exception:
        return None

    return parse_llm_output(raw)
