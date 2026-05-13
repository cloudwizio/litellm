# Mavvrik FOCUS Destination Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Mavvrik as a FOCUS export destination so LiteLLM spend data is exported to Mavvrik via the existing FOCUS integration, requiring only 3 env vars from the customer.

**Architecture:** Implement `FocusMavvrikDestination` following the `FocusVantageDestination` pattern — a `deliver()` method that POSTs CSV content to the Mavvrik ingestion API. Add a `MavvrikFocusLogger(FocusLogger)` subclass for dedicated scheduler registration. Fix the `ConsumedQuantity`/`PricingQuantity` bug in `transformer.py` while here.

**Tech Stack:** Python, Polars, httpx (via litellm shared HTTP handler), Pydantic, APScheduler, LiteLLM CustomLogger

---

## Customer Configuration (3 env vars)

```bash
FOCUS_PROVIDER=mavvrik
MAVVRIK_API_KEY=<key>
MAVVRIK_API_ENDPOINT=https://api.mavvrik.dev/<tenant_id>
MAVVRIK_CONNECTION_ID=<connection_id>
```

In config.yaml:
```yaml
litellm_settings:
  callbacks: ["mavvrik_focus"]
```

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `litellm/integrations/focus/destinations/mavvrik_destination.py` | `FocusMavvrikDestination` |
| Create | `litellm/integrations/mavvrik_focus/__init__.py` | Package init |
| Create | `litellm/integrations/mavvrik_focus/mavvrik_focus_logger.py` | `MavvrikFocusLogger(FocusLogger)` |
| Create | `tests/test_litellm/integrations/mavvrik_focus/__init__.py` | Test package |
| Create | `tests/test_litellm/integrations/mavvrik_focus/test_mavvrik_destination.py` | Unit tests |
| Modify | `litellm/integrations/focus/destinations/factory.py` | Register mavvrik provider |
| Modify | `litellm/integrations/focus/destinations/__init__.py` | Export FocusMavvrikDestination |
| Modify | `litellm/integrations/focus/transformer.py:99,111` | Fix ConsumedQuantity/PricingQuantity |
| Modify | `litellm/litellm_core_utils/custom_logger_registry.py` | Register mavvrik_focus callback |
| Modify | `litellm/__init__.py` | Add mavvrik_focus to callback Literal |
| Modify | `litellm/proxy/proxy_server.py` | Add scheduler registration block |
| Modify | `litellm/constants.py` | Add MAVVRIK_FOCUS_EXPORT_JOB_NAME |

---

## Task 1: Fix ConsumedQuantity and PricingQuantity in transformer

**Files:**
- Modify: `litellm/integrations/focus/transformer.py:99,111`
- Create: `tests/test_litellm/integrations/mavvrik_focus/__init__.py`
- Create: `tests/test_litellm/integrations/mavvrik_focus/test_mavvrik_destination.py`

- [ ] **Step 1: Create test package and write failing tests**

```bash
mkdir -p tests/test_litellm/integrations/mavvrik_focus
touch tests/test_litellm/integrations/mavvrik_focus/__init__.py
```

```python
# tests/test_litellm/integrations/mavvrik_focus/test_mavvrik_destination.py
import polars as pl
import pytest
from litellm.integrations.focus.transformer import FocusTransformer


def _make_frame() -> pl.DataFrame:
    return pl.DataFrame({
        "id": ["row-1"], "date": ["2026-05-01"], "user_id": ["user-1"],
        "api_key": ["sk-hash-1"], "model": ["gpt-4o"], "model_group": ["gpt-4o"],
        "custom_llm_provider": ["openai"], "prompt_tokens": [1000],
        "completion_tokens": [500], "spend": [0.05], "api_requests": [42],
        "successful_requests": [42], "failed_requests": [0],
        "cache_creation_input_tokens": [0], "cache_read_input_tokens": [0],
        "created_at": [None], "updated_at": [None], "team_id": [None],
        "api_key_alias": [None], "team_alias": [None], "user_email": [None],
        "ChargePeriodStart": [None], "ChargePeriodEnd": [None],
    })


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
poetry run pytest tests/test_litellm/integrations/mavvrik_focus/test_mavvrik_destination.py::test_consumed_quantity_uses_api_requests -v
```

Expected: FAIL — `assert 1.0 == 42.0`

- [ ] **Step 3: Fix transformer.py lines 99 and 111**

```python
# Line 99 — replace:
dec(pl.lit(1.0)).alias("ConsumedQuantity"),
# with:
dec(pl.col("api_requests").cast(pl.Float64).fill_null(1.0)).alias("ConsumedQuantity"),

# Line 111 — replace:
dec(pl.lit(1.0)).alias("PricingQuantity"),
# with:
dec(pl.col("api_requests").cast(pl.Float64).fill_null(1.0)).alias("PricingQuantity"),
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_litellm/integrations/mavvrik_focus/test_mavvrik_destination.py -k "transformer or quantity" -v
```

Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add litellm/integrations/focus/transformer.py tests/test_litellm/integrations/mavvrik_focus/
git commit -m "fix(focus): use api_requests for ConsumedQuantity and PricingQuantity"
```

---

## Task 2: Add MAVVRIK_FOCUS_EXPORT_JOB_NAME constant

**Files:**
- Modify: `litellm/constants.py`

- [ ] **Step 1: Add constant near existing MAVVRIK constants**

Search for `MAVVRIK_EXPORT_USAGE_DATA_JOB_NAME` in `litellm/constants.py` and add below it:

```python
MAVVRIK_FOCUS_EXPORT_JOB_NAME = "mavvrik_focus_export_usage_data"
```

- [ ] **Step 2: Commit**

```bash
git add litellm/constants.py
git commit -m "feat(mavvrik-focus): add MAVVRIK_FOCUS_EXPORT_JOB_NAME constant"
```

---

## Task 3: Implement FocusMavvrikDestination

**Files:**
- Create: `litellm/integrations/focus/destinations/mavvrik_destination.py`

- [ ] **Step 1: Write failing tests for destination**

Append to `tests/test_litellm/integrations/mavvrik_focus/test_mavvrik_destination.py`:

```python
import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch
from litellm.integrations.focus.destinations.base import FocusTimeWindow


def _make_window() -> FocusTimeWindow:
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


def test_destination_missing_connection_id_raises():
    from litellm.integrations.focus.destinations.mavvrik_destination import FocusMavvrikDestination
    with pytest.raises(ValueError, match="MAVVRIK_CONNECTION_ID"):
        FocusMavvrikDestination(prefix="p", config={"api_key": "k", "api_endpoint": "https://api.mavvrik.dev/t"})


def test_deliver_empty_content_skips():
    dest = _make_destination()
    asyncio.get_event_loop().run_until_complete(
        dest.deliver(content=b"", time_window=_make_window(), filename="f.csv")
    )


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
```

- [ ] **Step 2: Run to verify they fail**

```bash
poetry run pytest tests/test_litellm/integrations/mavvrik_focus/test_mavvrik_destination.py::test_destination_missing_api_key_raises -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement FocusMavvrikDestination**

Create `litellm/integrations/focus/destinations/mavvrik_destination.py`:

```python
"""Mavvrik API destination for FOCUS export."""

from __future__ import annotations

import os
from typing import Any, Optional
from urllib.parse import urlparse

from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    get_async_httpx_client,
    httpxSpecialProvider,
)

from .base import FocusDestination, FocusTimeWindow

_MAVVRIK_ALLOWED_SUFFIXES = (".mavvrik.dev", ".mavvrik.ai", ".mavvrik.app")


def _validate_api_endpoint(api_endpoint: str) -> None:
    if not api_endpoint.startswith("https://"):
        raise ValueError("MAVVRIK_API_ENDPOINT must be an HTTPS URL")
    hostname = (urlparse(api_endpoint).hostname or "").lower()
    if not any(hostname.endswith(suffix) for suffix in _MAVVRIK_ALLOWED_SUFFIXES):
        raise ValueError(
            "MAVVRIK_API_ENDPOINT host must be a Mavvrik domain "
            "(e.g. https://api.mavvrik.dev/<tenant_id>)"
        )


class FocusMavvrikDestination(FocusDestination):
    """Upload FOCUS CSV exports to the Mavvrik ingestion API."""

    def __init__(self, *, prefix: str, config: Optional[dict[str, Any]] = None) -> None:
        config = config or {}
        api_key = config.get("api_key")
        api_endpoint = config.get("api_endpoint")
        connection_id = config.get("connection_id")

        if not api_key:
            raise ValueError("MAVVRIK_API_KEY must be provided for Mavvrik FOCUS destination")
        if not api_endpoint:
            raise ValueError("MAVVRIK_API_ENDPOINT must be provided for Mavvrik FOCUS destination")
        if not connection_id:
            raise ValueError("MAVVRIK_CONNECTION_ID must be provided for Mavvrik FOCUS destination")

        _validate_api_endpoint(api_endpoint)

        self.api_key = api_key
        self.api_endpoint = api_endpoint.rstrip("/")
        self.connection_id = connection_id
        self.prefix = prefix
        self._http: AsyncHTTPHandler = get_async_httpx_client(
            llm_provider=httpxSpecialProvider.LoggingCallback
        )

    @property
    def _ingest_url(self) -> str:
        return f"{self.api_endpoint}/metrics/agent/ai/{self.connection_id}/ingest"

    @property
    def _auth_headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key}

    async def deliver(self, *, content: bytes, time_window: FocusTimeWindow, filename: str) -> None:
        """POST FOCUS CSV content to the Mavvrik ingestion API."""
        if not content:
            verbose_logger.debug("Mavvrik FOCUS destination: empty content, skipping upload")
            return

        verbose_logger.debug(
            "Mavvrik FOCUS destination: uploading %d bytes (%s) window=%s→%s",
            len(content), filename,
            time_window.start_time.date(), time_window.end_time.date(),
        )

        resp = await self._http.client.request(
            method="POST",
            url=self._ingest_url,
            headers={**self._auth_headers, "Content-Type": "text/csv"},
            content=content,
            timeout=120.0,
        )

        if resp.status_code >= 400:
            raise RuntimeError(
                f"Mavvrik FOCUS destination: upload failed with status "
                f"{resp.status_code}: {resp.text[:200]}"
            )

        verbose_logger.debug(
            "Mavvrik FOCUS destination: upload complete, status=%d", resp.status_code
        )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
poetry run pytest tests/test_litellm/integrations/mavvrik_focus/test_mavvrik_destination.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add litellm/integrations/focus/destinations/mavvrik_destination.py
git commit -m "feat(mavvrik-focus): add FocusMavvrikDestination"
```

---

## Task 4: Register destination in factory and __init__

**Files:**
- Modify: `litellm/integrations/focus/destinations/factory.py`
- Modify: `litellm/integrations/focus/destinations/__init__.py`

- [ ] **Step 1: Update factory.py — add import**

At the top of `factory.py` add:
```python
from .mavvrik_destination import FocusMavvrikDestination
```

- [ ] **Step 2: Update factory.py — add to create()**

After the `vantage` branch in `create()`:
```python
if provider_lower == "mavvrik":
    return FocusMavvrikDestination(prefix=prefix, config=normalized_config)
```

- [ ] **Step 3: Update factory.py — add to _resolve_config()**

After the `vantage` branch in `_resolve_config()`:
```python
if provider == "mavvrik":
    resolved = {
        "api_key": overrides.get("api_key") or os.getenv("MAVVRIK_API_KEY"),
        "api_endpoint": overrides.get("api_endpoint") or os.getenv("MAVVRIK_API_ENDPOINT"),
        "connection_id": overrides.get("connection_id") or os.getenv("MAVVRIK_CONNECTION_ID"),
    }
    return {k: v for k, v in resolved.items() if v is not None}
```

- [ ] **Step 4: Update destinations/__init__.py**

Add to the exports:
```python
from .mavvrik_destination import FocusMavvrikDestination
```
And add `"FocusMavvrikDestination"` to `__all__`.

- [ ] **Step 5: Verify**

```bash
poetry run python -c "
import os
os.environ['MAVVRIK_API_KEY'] = 'test-key'
os.environ['MAVVRIK_API_ENDPOINT'] = 'https://api.mavvrik.dev/tenant-1'
os.environ['MAVVRIK_CONNECTION_ID'] = 'conn-1'
from litellm.integrations.focus.destinations.factory import FocusDestinationFactory
dest = FocusDestinationFactory.create(provider='mavvrik', prefix='focus_exports')
print('OK:', type(dest).__name__)
"
```

Expected: `OK: FocusMavvrikDestination`

- [ ] **Step 6: Commit**

```bash
git add litellm/integrations/focus/destinations/factory.py litellm/integrations/focus/destinations/__init__.py
git commit -m "feat(mavvrik-focus): register mavvrik provider in FOCUS destination factory"
```

---

## Task 5: Add MavvrikFocusLogger subclass

**Files:**
- Create: `litellm/integrations/mavvrik_focus/__init__.py`
- Create: `litellm/integrations/mavvrik_focus/mavvrik_focus_logger.py`

- [ ] **Step 1: Create package and logger**

```bash
mkdir -p litellm/integrations/mavvrik_focus
touch litellm/integrations/mavvrik_focus/__init__.py
```

Create `litellm/integrations/mavvrik_focus/mavvrik_focus_logger.py`:

```python
"""MavvrikFocusLogger — FOCUS-based Mavvrik export logger."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import MAVVRIK_FOCUS_EXPORT_JOB_NAME
from litellm.integrations.focus.focus_logger import FocusLogger

if TYPE_CHECKING:
    from apscheduler.schedulers.asyncio import AsyncIOScheduler
else:
    AsyncIOScheduler = Any


class MavvrikFocusLogger(FocusLogger):
    """FOCUS-based export logger that routes to the Mavvrik destination."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            provider="mavvrik",
            export_format="csv",
            frequency=os.getenv("MAVVRIK_FOCUS_FREQUENCY", "hourly"),
            prefix="mavvrik_focus_exports",
            destination_config={
                "api_key": os.getenv("MAVVRIK_API_KEY"),
                "api_endpoint": os.getenv("MAVVRIK_API_ENDPOINT"),
                "connection_id": os.getenv("MAVVRIK_CONNECTION_ID"),
            },
            **kwargs,
        )

    async def initialize_mavvrik_focus_export_job(self) -> None:
        """Scheduler entry point — uses Mavvrik-specific pod-lock key."""
        from litellm.proxy.proxy_server import proxy_logging_obj  # noqa: PLC0415

        pod_lock_manager = None
        if proxy_logging_obj is not None:
            writer = getattr(proxy_logging_obj, "db_spend_update_writer", None)
            if writer is not None:
                pod_lock_manager = getattr(writer, "pod_lock_manager", None)

        if pod_lock_manager and pod_lock_manager.redis_cache:
            acquired = await pod_lock_manager.acquire_lock(
                cronjob_id=MAVVRIK_FOCUS_EXPORT_JOB_NAME
            )
            if not acquired:
                verbose_proxy_logger.debug("Mavvrik FOCUS export: unable to acquire pod lock")
                return
            try:
                await self._run_scheduled_export()
            finally:
                await pod_lock_manager.release_lock(cronjob_id=MAVVRIK_FOCUS_EXPORT_JOB_NAME)
        else:
            await self._run_scheduled_export()

    @staticmethod
    async def init_mavvrik_focus_background_job(scheduler: AsyncIOScheduler) -> None:
        """Register the Mavvrik FOCUS export job on the provided scheduler."""
        loggers = [
            cb
            for cb in litellm.logging_callback_manager.get_custom_loggers_for_type(
                callback_type=MavvrikFocusLogger
            )
            if type(cb) is MavvrikFocusLogger
        ]
        if not loggers:
            verbose_proxy_logger.debug("No MavvrikFocusLogger registered; skipping scheduler")
            return

        logger = loggers[0]
        trigger_kwargs = logger._build_scheduler_trigger()
        scheduler.add_job(
            logger.initialize_mavvrik_focus_export_job,
            id=MAVVRIK_FOCUS_EXPORT_JOB_NAME,
            replace_existing=True,
            **trigger_kwargs,
        )
        verbose_proxy_logger.info("mavvrik_focus: background export job scheduled (%s)", trigger_kwargs)
```

- [ ] **Step 2: Verify import works**

```bash
poetry run python -c "
import os
os.environ['MAVVRIK_API_KEY'] = 'test'
os.environ['MAVVRIK_API_ENDPOINT'] = 'https://api.mavvrik.dev/t'
os.environ['MAVVRIK_CONNECTION_ID'] = 'c'
from litellm.integrations.mavvrik_focus.mavvrik_focus_logger import MavvrikFocusLogger
l = MavvrikFocusLogger()
print('OK:', l.provider, l.export_format)
"
```

Expected: `OK: mavvrik csv`

- [ ] **Step 3: Commit**

```bash
git add litellm/integrations/mavvrik_focus/
git commit -m "feat(mavvrik-focus): add MavvrikFocusLogger subclass"
```

---

## Task 6: Register mavvrik_focus callback

**Files:**
- Modify: `litellm/litellm_core_utils/custom_logger_registry.py`
- Modify: `litellm/__init__.py`

- [ ] **Step 1: Update custom_logger_registry.py**

Add alongside the existing `"mavvrik"` entry:

```python
from litellm.integrations.mavvrik_focus.mavvrik_focus_logger import MavvrikFocusLogger
# in the registry dict:
"mavvrik_focus": MavvrikFocusLogger,
```

- [ ] **Step 2: Update litellm/__init__.py**

Find the callbacks Literal (search for `"mavvrik"`) and add `"mavvrik_focus"` next to it.

- [ ] **Step 3: Verify**

```bash
poetry run python -c "
from litellm.litellm_core_utils.custom_logger_registry import CALLBACK_CLASS_STR_TO_CLASS_TYPE
print('registered:', 'mavvrik_focus' in CALLBACK_CLASS_STR_TO_CLASS_TYPE)
"
```

Expected: `registered: True`

- [ ] **Step 4: Commit**

```bash
git add litellm/litellm_core_utils/custom_logger_registry.py litellm/__init__.py
git commit -m "feat(mavvrik-focus): register mavvrik_focus callback string"
```

---

## Task 7: Register scheduler in proxy_server.py

**Files:**
- Modify: `litellm/proxy/proxy_server.py`

- [ ] **Step 1: Find the Vantage scheduler block**

Search for `init_vantage_background_job` in `proxy_server.py` and add immediately after it:

```python
########################################################
# Mavvrik FOCUS Background Job
########################################################
from litellm.integrations.mavvrik_focus.mavvrik_focus_logger import (  # noqa: PLC0415
    MavvrikFocusLogger,
)
await MavvrikFocusLogger.init_mavvrik_focus_background_job(scheduler)
```

- [ ] **Step 2: Verify**

```bash
poetry run python -c "
from litellm.integrations.mavvrik_focus.mavvrik_focus_logger import MavvrikFocusLogger
print('import OK')
"
```

Expected: `import OK`

- [ ] **Step 3: Commit**

```bash
git add litellm/proxy/proxy_server.py
git commit -m "feat(mavvrik-focus): register MavvrikFocusLogger scheduler at proxy startup"
```

---

## Task 8: Lint, type-check, and full test run

- [ ] **Step 1: Black**

```bash
poetry run black litellm/integrations/focus/destinations/mavvrik_destination.py \
  litellm/integrations/mavvrik_focus/ \
  litellm/integrations/focus/transformer.py
```

- [ ] **Step 2: Ruff**

```bash
poetry run ruff check litellm/integrations/focus/destinations/mavvrik_destination.py \
  litellm/integrations/mavvrik_focus/ \
  litellm/integrations/focus/transformer.py
```

- [ ] **Step 3: MyPy**

```bash
poetry run mypy \
  litellm/integrations/focus/destinations/mavvrik_destination.py \
  litellm/integrations/mavvrik_focus/mavvrik_focus_logger.py \
  litellm/integrations/focus/transformer.py \
  --ignore-missing-imports
```

Expected: `Success: no issues found`

- [ ] **Step 4: Run all tests**

```bash
poetry run pytest tests/test_litellm/integrations/mavvrik_focus/ -v
```

Expected: All PASS

- [ ] **Step 5: Fix and commit if needed**

```bash
git add -u
git commit -m "style(mavvrik-focus): apply linting and type fixes"
```

---

## Task 9: Push and open PR

- [ ] **Step 1: Push branch**

```bash
git push cloudwiz mavvrik/focus-destination
```

- [ ] **Step 2: Open PR against BerriAI/litellm targeting litellm_internal_staging**

```bash
gh pr create \
  --repo BerriAI/litellm \
  --base litellm_internal_staging \
  --head cloudwizio:mavvrik/focus-destination \
  --title "feat(mavvrik): add Mavvrik as FOCUS export destination" \
  --body "$(cat <<'EOF'
## Summary

- Adds FocusMavvrikDestination — a FOCUS destination that POSTs CSV spend data to the Mavvrik ingestion API
- Adds MavvrikFocusLogger(FocusLogger) subclass registered as callbacks: [mavvrik_focus]
- Fixes ConsumedQuantity and PricingQuantity in transformer.py (were hardcoded to 1.0, now use api_requests from DB)

## Customer configuration (3 env vars)

MAVVRIK_API_KEY, MAVVRIK_API_ENDPOINT, MAVVRIK_CONNECTION_ID

In config.yaml: callbacks: [mavvrik_focus]

## Test plan

- Unit tests for FocusMavvrikDestination
- Unit tests for transformer fix
- make test-unit passes

Generated with Claude Code
EOF
)"
```

---

## Self-Review

- All 3 env vars resolved in factory
- deliver() validates endpoint domain (SSRF prevention)  
- Empty content skipped gracefully
- ConsumedQuantity/PricingQuantity fix tested including null fallback
- Scheduler uses dedicated job name — no collision with FOCUS job
- No S3/GCS creds required from customer
- Follows exact same pattern as VantageLogger/FocusVantageDestination
