"""The request rules of the agent's model calls that the real API enforces and a
stand-in would not.

The scripted model (app/scripted_model.py) calls validate_request on every agent
request and turns a failure into the 400 the API would have sent, and the agent
tests call it on every request their fake receives, so a loop that builds an
invalid conversation fails the same way offline as it would live -- after a
billed call, in the real case.
"""

from itertools import pairwise
from typing import Any

from app.agent.tools import MalformedToolResult, assert_tool_result_content


class InvalidRequestError(ValueError):
    """A request the API would answer with a 400."""


def _field(block: Any, name: str) -> Any:
    """A content block's field, whether the block is a dict (what the loop
    builds) or an SDK object (an assistant turn echoed back as returned)."""
    return block.get(name) if isinstance(block, dict) else getattr(block, name, None)


def validate_request(kwargs: dict[str, Any]) -> None:
    """Raise InvalidRequestError unless the conversation in `kwargs["messages"]`
    is one the API accepts: roles alternate starting with the user, and the user
    message after a tool_use turn opens with exactly one tool_result per tool_use
    id, in order, each with a valid body (assert_tool_result_content)."""
    messages = kwargs["messages"]
    if not messages or messages[0]["role"] != "user":
        raise InvalidRequestError("the first message must be from the user")
    for before, after in pairwise(messages):
        if before["role"] == after["role"]:
            raise InvalidRequestError("roles must alternate")
    for index, message in enumerate(messages):
        content = message["content"]
        if message["role"] == "assistant" and not isinstance(content, str):
            tool_ids = [_field(b, "id") for b in content if _field(b, "type") == "tool_use"]
            if not tool_ids:
                continue
            if index + 1 >= len(messages):
                raise InvalidRequestError("a tool_use turn needs its tool_results")
            results = messages[index + 1]["content"]
            if not isinstance(results, list):
                raise InvalidRequestError("tool_results must be a list of blocks")
            head = results[: len(tool_ids)]
            if [_field(r, "type") for r in head] != ["tool_result"] * len(tool_ids):
                raise InvalidRequestError("a tool_use turn needs one tool_result per tool_use")
            if [_field(r, "tool_use_id") for r in head] != tool_ids:
                raise InvalidRequestError("tool_results must answer the tool_use ids, in order")
    for message in messages:
        if message["role"] == "user" and isinstance(message["content"], list):
            for block in message["content"]:
                if _field(block, "type") == "tool_result":
                    try:
                        assert_tool_result_content(_field(block, "content"))
                    except MalformedToolResult as exc:
                        raise InvalidRequestError(f"invalid tool_result: {exc}") from exc
