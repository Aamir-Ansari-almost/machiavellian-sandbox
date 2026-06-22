import os
from typing import Literal, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

LLAMA_BASE_URL = os.getenv("LLAMA_BASE_URL", "http://localhost:8080/v1")
MODEL_NAME = os.getenv("MODEL", "qwen2.5-7b-instruct")
MAX_RETRIES = 3

_client = AsyncOpenAI(base_url=LLAMA_BASE_URL, api_key="sk-no-key-required")

_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["cooperate", "defect", "betray", "negotiate", "ignore"],
        },
        "target": {"type": ["string", "null"]},
        "speech": {"type": "string"},
        "reasoning": {"type": "string"},
    },
    "required": ["action", "target", "speech", "reasoning"],
    "additionalProperties": False,
}


class AgentDecision(BaseModel):
    action: Literal["cooperate", "defect", "betray", "negotiate", "ignore"]
    target: Optional[str]
    speech: str
    reasoning: str


async def call_agent(system_prompt: str, user_prompt: str) -> AgentDecision:
    """
    Call the local LLM and return a validated AgentDecision.
    Tries json_schema mode first (strongest guarantee); falls back to
    json_object mode with pydantic retry if the model/server doesn't support it.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = await _client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "agent_decision",
                        "schema": _DECISION_SCHEMA,
                        "strict": True,
                    },
                },
                temperature=0.7,
            )
        except Exception:
            # json_schema not supported by this llama.cpp build — fall back to json_object
            response = await _client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.7,
            )

        content = response.choices[0].message.content.strip()
        try:
            return AgentDecision.model_validate_json(content)
        except ValidationError:
            if attempt == MAX_RETRIES - 1:
                raise ValueError(
                    f"LLM returned invalid JSON after {MAX_RETRIES} attempts.\n"
                    f"Last output: {content}"
                )

    raise RuntimeError("Unreachable")
