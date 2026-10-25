"""The suite cannot spend: tests/conftest.py blanks the Anthropic key before any
settings load (backend/.env holds the real one), and the `client` fixtures hand
routes a client that fails the test if it is used."""

import os

import anthropic
import pytest
from conftest import ModelCallAttempted
from httpx import AsyncClient
from pydantic import ValidationError

from app.config import Settings, get_settings
from app.extraction import _build_client
from app.main import app
from app.scripted_model import ScriptedModelClient

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_SMOKE") == "1", reason="a live run keeps the key from backend/.env"
)

SCRIPTED = {
    "model_provider": "scripted",
    "extraction_model": "scripted",
    "transcription_model": "scripted",
    "agent_model": "scripted",
    "judge_model": "scripted",
}


def test_the_suites_settings_carry_no_key() -> None:
    assert not get_settings().anthropic_api_key
    assert not Settings().anthropic_api_key


async def test_ask_through_the_client_without_a_fake_never_reaches_a_model(client: AsyncClient) -> None:
    # No key: the route's own 503, before any client exists.
    response = await client.post("/ask", json={"question": "q"})

    assert response.status_code == 503


async def test_a_key_in_settings_without_a_fake_fails_the_test_loudly(client: AsyncClient) -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(
        anthropic_api_key="not-a-real-key", ask_rate_limit_per_minute=0, daily_budget_usd=0
    )

    with pytest.raises(ModelCallAttempted):
        await client.post("/ask", json={"question": "q"})



# --- the scripted provider (app/scripted_model.py) ---------------------------------
#
# The load stack's no-spend layers that live in Settings and the client factory;
# tests/test_scripted_model.py and tests/test_scripted_end_to_end.py cover what
# the scripted model does.


def test_a_scripted_process_cannot_hold_a_key() -> None:
    with pytest.raises(ValidationError, match="refuses a non-empty ANTHROPIC_API_KEY"):
        Settings(**SCRIPTED, anthropic_api_key="not-a-real-key")


def test_a_real_provider_cannot_name_the_scripted_model() -> None:
    with pytest.raises(ValidationError, match="only MODEL_PROVIDER=scripted can answer"):
        Settings(extraction_model="scripted", anthropic_api_key="not-a-real-key")


def test_scripted_builds_no_anthropic_client_and_the_guard_is_not_vacuous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def refuse(self, *args, **kwargs) -> None:
        raise AssertionError("a real Anthropic client was built")

    monkeypatch.setattr(anthropic.AsyncAnthropic, "__init__", refuse)

    assert isinstance(_build_client(Settings(**SCRIPTED, anthropic_api_key=None)), ScriptedModelClient)
    # The control: the same call for the real provider reaches the patched constructor.
    with pytest.raises(AssertionError, match="a real Anthropic client was built"):
        _build_client(Settings(anthropic_api_key="not-a-real-key"))
