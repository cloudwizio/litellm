"""Mavvrik GCS destination for FOCUS export.

Flow:
  1. GET /metrics/agent/ai/{connection_id}/upload-url → GCS signed URL
  2. PUT <signed_url> with CSV content
"""

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
    """Upload FOCUS CSV exports to Mavvrik via GCS signed URL."""

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
    def _agent_url(self) -> str:
        return f"{self.api_endpoint}/metrics/agent/ai/{self.connection_id}"

    @property
    def _upload_url_endpoint(self) -> str:
        return f"{self.api_endpoint}/metrics/agent/ai/{self.connection_id}/upload-url"

    @property
    def _auth_headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "x-api-key": self.api_key}

    async def _register(self) -> None:
        """POST agent endpoint to register/initialize the connector."""
        resp = await self._http.client.request(
            method="POST",
            url=self._agent_url,
            headers=self._auth_headers,
            json={"name": self.connection_id},
            timeout=30.0,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Mavvrik FOCUS destination: register failed "
                f"({resp.status_code}): {resp.text[:200]}"
            )
        verbose_logger.debug("Mavvrik FOCUS destination: connector registered")

    async def _get_signed_url(self, date_str: str) -> str:
        """GET upload-url endpoint → GCS signed URL for the given date."""
        params = {"name": date_str, "type": "metrics", "datetime": date_str}
        resp = await self._http.client.request(
            method="GET",
            url=self._upload_url_endpoint,
            headers=self._auth_headers,
            params=params,
            timeout=30.0,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Mavvrik FOCUS destination: failed to get signed URL "
                f"({resp.status_code}): {resp.text[:200]}"
            )
        signed_url = resp.json().get("url")
        if not signed_url:
            raise RuntimeError(
                f"Mavvrik FOCUS destination: response missing 'url' field: {resp.json()}"
            )
        verbose_logger.debug(
            "Mavvrik FOCUS destination: got signed URL for date %s", date_str
        )
        return signed_url

    async def _upload_to_gcs(self, signed_url: str, content: bytes) -> None:
        """PUT content to GCS signed URL."""
        resp = await self._http.client.request(
            method="PUT",
            url=signed_url,
            headers={"Content-Type": "text/csv"},
            content=content,
            timeout=120.0,
        )
        if resp.status_code >= 400:
            raise RuntimeError(
                f"Mavvrik FOCUS destination: GCS upload failed "
                f"({resp.status_code}): {resp.text[:200]}"
            )

    async def deliver(
        self,
        *,
        content: bytes,
        time_window: FocusTimeWindow,
        filename: str,
    ) -> None:
        """Upload FOCUS CSV to Mavvrik via GCS signed URL.

        Uses the start date of the time window as the object date key.
        """
        if not content:
            verbose_logger.debug(
                "Mavvrik FOCUS destination: empty content, skipping upload"
            )
            return

        date_str = time_window.start_time.strftime("%Y-%m-%d")

        verbose_logger.debug(
            "Mavvrik FOCUS destination: uploading %d bytes for date=%s (%s)",
            len(content),
            date_str,
            filename,
        )

        await self._register()
        signed_url = await self._get_signed_url(date_str)
        await self._upload_to_gcs(signed_url, content)

        verbose_logger.debug(
            "Mavvrik FOCUS destination: upload complete for date=%s", date_str
        )
