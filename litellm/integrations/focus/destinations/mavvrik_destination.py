"""Mavvrik API destination for FOCUS export."""

from __future__ import annotations

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

    def __init__(
        self,
        *,
        prefix: str,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        config = config or {}
        api_key = config.get("api_key")
        api_endpoint = config.get("api_endpoint")
        connection_id = config.get("connection_id")

        if not api_key:
            raise ValueError(
                "MAVVRIK_API_KEY must be provided for Mavvrik FOCUS destination "
                "(set MAVVRIK_API_KEY env var or pass in destination_config)"
            )
        if not api_endpoint:
            raise ValueError(
                "MAVVRIK_API_ENDPOINT must be provided for Mavvrik FOCUS destination "
                "(set MAVVRIK_API_ENDPOINT env var or pass in destination_config)"
            )
        if not connection_id:
            raise ValueError(
                "MAVVRIK_CONNECTION_ID must be provided for Mavvrik FOCUS destination "
                "(set MAVVRIK_CONNECTION_ID env var or pass in destination_config)"
            )

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

    async def deliver(
        self,
        *,
        content: bytes,
        time_window: FocusTimeWindow,
        filename: str,
    ) -> None:
        """POST FOCUS CSV content to the Mavvrik ingestion API."""
        if not content:
            verbose_logger.debug(
                "Mavvrik FOCUS destination: empty content, skipping upload"
            )
            return

        verbose_logger.debug(
            "Mavvrik FOCUS destination: uploading %d bytes (%s) window=%s→%s",
            len(content),
            filename,
            time_window.start_time.date(),
            time_window.end_time.date(),
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
            "Mavvrik FOCUS destination: upload complete, status=%d",
            resp.status_code,
        )
