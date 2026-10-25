"""The suite cannot spend: tests/conftest.py blanks the Anthropic key before any
settings load (backend/.env holds the real one), and the `client` fixtures hand
routes a client that fails the test if it is used."""

import os

import pytest
from conftest import ModelCallAttempted
from httpx import AsyncClient

from app.config import Settings, get_settings
from app.main import app

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_SMOKE") == "1", reason="a live run keeps the key from backend/.env"
)


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

