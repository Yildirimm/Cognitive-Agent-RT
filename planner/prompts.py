from typing import Dict, Any, List


ALLOWED_ACTIONS: List[str] = [
    "forward",
    "turn_left",
    "turn_right",
    "stop",
]


SYSTEM_PROMPT: str = (
    "You are a navigation planner for a robot.\n"
    "You MUST output exactly one action.\n"
    "You MUST follow the JSON schema.\n"
    "Do NOT include explanations or extra text."
)


def build_prompt(symbolic_state: Dict[str, Any]) -> Dict[str, str]:
    """
    Returns a prompt dictionary to be consumed by an LLM client.
    The LLM must respond ONLY with valid JSON.
    """

    user_prompt = f"""
                State:
                {symbolic_state}

                Goal:
                Reach the target safely.

                Allowed actions:
                {ALLOWED_ACTIONS}

                Hard rules:
                - If at_target == True -> action MUST be "stop"
                - If obstacle_front < 0.40 -> action MUST NOT be "forward"

                Output format (JSON ONLY):
                {{"action": "<one_of_allowed_actions>"}}
                """.strip()

    return {
        "system": SYSTEM_PROMPT,
        "user": user_prompt,
    }
