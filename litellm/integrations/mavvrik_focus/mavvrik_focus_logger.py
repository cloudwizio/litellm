"""MavvrikFocusLogger — FOCUS-based Mavvrik export logger.

Usage in config.yaml:
    litellm_settings:
      callbacks: ["mavvrik_focus"]

Required env vars:
    MAVVRIK_API_KEY
    MAVVRIK_API_ENDPOINT
    MAVVRIK_CONNECTION_ID
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, List

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.constants import MAVVRIK_FOCUS_EXPORT_JOB_NAME
from litellm.integrations.custom_logger import CustomLogger
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
            interval_seconds=int(os.getenv("MAVVRIK_FOCUS_INTERVAL_SECONDS", 3600)),
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
                verbose_proxy_logger.debug(
                    "Mavvrik FOCUS export: unable to acquire pod lock"
                )
                return
            try:
                await self._run_scheduled_export()
            finally:
                await pod_lock_manager.release_lock(
                    cronjob_id=MAVVRIK_FOCUS_EXPORT_JOB_NAME
                )
        else:
            await self._run_scheduled_export()

    @staticmethod
    async def init_mavvrik_focus_background_job(
        scheduler: AsyncIOScheduler,
    ) -> None:
        """Register the Mavvrik FOCUS export job on the provided scheduler."""
        loggers: List[MavvrikFocusLogger] = [
            cb
            for cb in litellm.logging_callback_manager.get_custom_loggers_for_type(
                callback_type=MavvrikFocusLogger
            )
            if type(cb) is MavvrikFocusLogger
        ]
        if not loggers:
            verbose_proxy_logger.debug(
                "No MavvrikFocusLogger registered; skipping scheduler"
            )
            return

        logger = loggers[0]
        trigger_kwargs = logger._build_scheduler_trigger()
        scheduler.add_job(  # type: ignore[attr-defined]
            logger.initialize_mavvrik_focus_export_job,
            id=MAVVRIK_FOCUS_EXPORT_JOB_NAME,
            replace_existing=True,
            **trigger_kwargs,
        )
        verbose_proxy_logger.info(
            "mavvrik_focus: background export job scheduled (%s)", trigger_kwargs
        )
