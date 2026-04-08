from __future__ import annotations

import json
import os
from typing import Dict, List

from openai import OpenAI

from environment import SupportTicketEnv
from models import Action, ActionType, TicketType


SYSTEM_PROMPT = """You are an AI customer support agent acting in a structured environment.
Return ONLY valid JSON with fields:
- action_type: one of classify, respond, resolve
- classification: one of refund, delay, complaint, query (or null)
- response_text: string (or null)
- resolve: boolean (or null)
- escalate: boolean
- notes: object
"""


def _safe_ticket_type(value: str | None) -> TicketType | None:
    if value is None:
        return None
    try:
        return TicketType(value)
    except ValueError:
        return None


def _build_fallback_action(allowed_actions: List[ActionType]) -> Action:
    if not allowed_actions:
        return Action(action_type=ActionType.RESPOND, response_text="I will review this case now.", escalate=False)

    if allowed_actions[0] == ActionType.CLASSIFY:
        return Action(
            action_type=ActionType.CLASSIFY,
            classification=TicketType.QUERY,
            response_text=None,
            resolve=None,
            escalate=False,
            notes={"fallback": True},
        )
    if allowed_actions[0] == ActionType.RESPOND:
        return Action(
            action_type=ActionType.RESPOND,
            response_text="I understand your issue and will take the next steps to resolve it.",
            resolve=None,
            escalate=False,
            notes={"fallback": True},
        )
    return Action(
        action_type=ActionType.RESOLVE,
        resolve=True,
        escalate=False,
        notes={"fallback": True},
    )


def _action_from_model(client: OpenAI, model_name: str, observation: Dict) -> Action:
    allowed = observation.get("allowed_actions", [])
    user_prompt = (
        "Observation:\n"
        f"{json.dumps(observation, indent=2)}\n\n"
        "Choose the best next action for this step."
    )

    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or "{}"
        payload = json.loads(content)

        action_type = payload.get("action_type")
        if action_type not in {"classify", "respond", "resolve"}:
            return _build_fallback_action([ActionType(a) for a in allowed])

        action = Action(
            action_type=ActionType(action_type),
            classification=_safe_ticket_type(payload.get("classification")),
            response_text=payload.get("response_text"),
            resolve=payload.get("resolve"),
            escalate=bool(payload.get("escalate", False)),
            notes=payload.get("notes", {}) if isinstance(payload.get("notes", {}), dict) else {},
        )
        return action
    except Exception:
        return _build_fallback_action([ActionType(a) for a in allowed])


def run_task(client: OpenAI, model_name: str, difficulty: str) -> float:
    env = SupportTicketEnv(difficulty=difficulty, max_steps=8)
    observation = env.reset().model_dump(mode="json")
    done = False
    total_score = 0.0

    print(f"[START] task={difficulty}")
    while not done:
        action = _action_from_model(client, model_name, observation)
        next_obs, reward, done, info = env.step(action)
        total_score += reward.value

        print(
            "[STEP] "
            f"task={difficulty} stage={info['stage']} status={info['status']} "
            f"reward={reward.value:.3f} cumulative={info['cumulative_reward']:.3f}"
        )
        observation = next_obs.model_dump(mode="json")

    print(f"[END] task={difficulty} total_score={total_score:.3f}")
    return total_score


def main() -> None:
    api_base_url = os.getenv("API_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    hf_token = os.getenv("HF_TOKEN")

    if not api_base_url or not model_name or not hf_token:
        raise RuntimeError("Missing required env vars: API_BASE_URL, MODEL_NAME, HF_TOKEN")

    client = OpenAI(base_url=api_base_url, api_key=hf_token)

    final_scores: Dict[str, float] = {}
    for task in ("easy", "medium", "hard"):
        final_scores[task] = run_task(client, model_name, task)

    print("\nFinal scores:")
    for task in ("easy", "medium", "hard"):
        print(f"- {task}: {final_scores[task]:.3f}")


if __name__ == "__main__":
    main()
