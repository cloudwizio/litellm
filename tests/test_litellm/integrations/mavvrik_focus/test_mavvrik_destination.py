"""Tests for Mavvrik FOCUS destination and transformer fixes."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import polars as pl
import pytest

from litellm.integrations.focus.transformer import FocusTransformer


def _make_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": ["row-1"], "date": ["2026-05-01"], "user_id": ["user-1"],
            "api_key": ["sk-hash-1"], "model": ["gpt-4o"], "model_group": ["gpt-4o"],
            "custom_llm_provider": ["openai"], "prompt_tokens": [1000],
            "completion_tokens": [500], "spend": [0.05], "api_requests": [42],
            "successful_requests": [42], "failed_requests": [0],
            "cache_creation_input_tokens": [0], "cache_read_input_tokens": [0],
            "created_at": [None], "updated_at": [None], "team_id": [None],
            "api_key_alias": [None], "team_alias": [None], "user_email": [None],
            "ChargePeriodStart": [None], "ChargePeriodEnd": [None],
        }
    )


def test_consumed_quantity_uses_api_requests():
    result = FocusTransformer().transform(_make_frame())
    assert float(result.to_dicts()[0]["ConsumedQuantity"]) == 42.0


def test_pricing_quantity_uses_api_requests():
    result = FocusTransformer().transform(_make_frame())
    assert float(result.to_dicts()[0]["PricingQuantity"]) == 42.0


def test_consumed_quantity_fallback_when_null():
    frame = _make_frame().with_columns(pl.lit(None).cast(pl.Int64).alias("api_requests"))
    result = FocusTransformer().transform(frame)
    assert float(result.to_dicts()[0]["ConsumedQuantity"]) == 1.0


def _make_window():
    from litellm.integrations.focus.destinations.base import FocusTimeWindow
    return FocusTimeWindow(
        start_time=datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc),
        end_time=datetime(2026, 5, 2, 0, 0, tzinfo=timezone.utc),
        frequency="daily",
    )


def _make_destination():
    from litellm.integrations.focus.destinations.mavvrik_destination import FocusMavvrikDestination
    return FocusMavvrikDestination(
        prefix="focus_exports",
        config={"api_key": "test-key", "api_endpoint": "https://api.mavvrik.dev/tenant-1", "connection_id": "conn-1"},
    )


def test_destination_missing_api_key_raises():
    from litellm.integrations.focus.destinations.mavvrik_destination import FocusMavvrikDestination
    with pytest.raises(ValueError, match="MAVVRIK_API_KEY"):
        FocusMavvrikDestination(prefix="p", config={"api_endpoint": "https://api.mavvrik.dev/t", "connection_id": "c"})


def test_destination_missing_api_endpoint_raises():
    from litellm.integrations.focus.destinations.mavvrik_destination import FocusMavvrikDestination
    with pytest.raises(ValueError, match="MAVVRIK_API_ENDPOINT"):
        FocusMavvrikDestination(prefix="p", config={"api_key": "k", "connection_id": "c"})


def test_destination_missing_connection_id_raises():
    from litellm.integrations.focus.destinations.mavvrik_destination import FocusMavvrikDestination
    with pytest.raises(ValueError, match="MAVVRIK_CONNECTION_ID"):
        FocusMavvrikDestination(prefix="p", config={"api_key": "k", "api_endpoint": "https://api.mavvrik.dev/t"})


def test_destination_invalid_domain_raises():
    from litellm.integrations.focus.destinations.mavvrik_destination import FocusMavvrikDestination
    with pytest.raises(ValueError, match="Mavvrik domain"):
        FocusMavvrikDestination(prefix="p", config={"api_key": "k", "api_endpoint": "https://attacker.example.com/t", "connection_id": "c"})


def test_destination_userinfo_bypass_blocked():
    from litellm.integrations.focus.destinations.mavvrik_destination import FocusMavvrikDestination
    with pytest.raises(ValueError, match="Mavvrik domain"):
        FocusMavvrikDestination(prefix="p", config={"api_key": "k", "api_endpoint": "https://api.mavvrik.dev:443@attacker.example/t", "connection_id": "c"})


def test_deliver_empty_content_skips():
    dest = _make_destination()
    asyncio.get_event_loop().run_until_complete(dest.deliver(content=b"", time_window=_make_window(), filename="f.csv"))


def test_deliver_posts_csv_to_mavvrik():
    dest = _make_destination()
    mock_response = MagicMock()
    mock_response.status_code = 200
    with patch.object(dest._http, "client") as mock_client:
        mock_client.request = AsyncMock(return_value=mock_response)
        asyncio.get_event_loop().run_until_complete(
            dest.deliver(content=b"date,spend\n2026-05-01,0.05\n", time_window=_make_window(), filename="usage.csv")
        )
        call_kwargs = mock_client.request.call_args.kwargs
        assert call_kwargs["headers"]["x-api-key"] == "test-key"
        assert call_kwargs["headers"]["Content-Type"] == "text/csv"


def test_deliver_raises_on_4xx():
    dest = _make_destination()
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.text = "Unauthorized"
    with patch.object(dest._http, "client") as mock_client:
        mock_client.request = AsyncMock(return_value=mock_response)
        with pytest.raises(RuntimeError, match="401"):
            asyncio.get_event_loop().run_until_complete(
                dest.deliver(content=b"date,spend\n2026-05-01,0.05\n", time_window=_make_window(), filename="usage.csv")
            )
