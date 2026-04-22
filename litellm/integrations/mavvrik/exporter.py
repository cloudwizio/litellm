"""Exporter — fetch spend data from Postgres and transform to CSV.

Responsibility: extract data from LiteLLM's database and convert it to CSV.

Public interface:
  export(date_str, connection_id, limit) → (DataFrame, csv_str)
      Single entry point: fetch → filter → serialize. Used by Orchestrator.

  get_earliest_date() → Optional[str]
      Returns MIN(date) for first-run start date resolution.

Internal methods:
  _get_usage_data(date_str, limit) → DataFrame
  filter(df) → DataFrame
  _to_csv(df, connection_id) → str
"""

import io
from typing import Any, AsyncIterator, List, Optional, Tuple

import polars as pl

from litellm._logging import verbose_proxy_logger

# query_raw is used here instead of Prisma model methods because the query
# requires a 4-table LEFT JOIN (DailyUserSpend → VerificationToken →
# TeamTable → UserTable). Prisma's relational API cannot express a multi-hop
# JOIN in a single query without N+1 round-trips.
#
# dus.* selects all columns from LiteLLM_DailyUserSpend so that any new
# columns added to that table in future LiteLLM versions are automatically
# included in the export without requiring a code change here.
_USAGE_QUERY = """
SELECT
    dus.*,
    vt.team_id,
    vt.key_alias    AS api_key_alias,
    vt.organization_id,
    tt.team_alias,
    ut.user_email,
    ut.user_alias
FROM "LiteLLM_DailyUserSpend" dus
LEFT JOIN "LiteLLM_VerificationToken" vt  ON dus.api_key   = vt.token
LEFT JOIN "LiteLLM_TeamTable"         tt  ON vt.team_id    = tt.team_id
LEFT JOIN "LiteLLM_UserTable"         ut  ON dus.user_id   = ut.user_id
WHERE dus.date = $1
ORDER BY dus.date, dus.user_id, dus.model ASC
"""

_EARLIEST_DATE_QUERY = 'SELECT MIN(date) AS earliest FROM "LiteLLM_DailyUserSpend"'

# Fallback header used when DB is empty — derived from the known query columns.
_USAGE_HEADER = (
    "date,user_id,api_key,model,model_group,custom_llm_provider,"
    "prompt_tokens,completion_tokens,spend,api_requests,successful_requests,"
    "failed_requests,cache_creation_input_tokens,cache_read_input_tokens,"
    "created_at,updated_at,id,team_id,api_key_alias,organization_id,"
    "team_alias,user_email,user_alias\n"
)


class Exporter:
    """Fetch LiteLLM spend data from Postgres and transform to CSV."""

    # ------------------------------------------------------------------
    # DB access helper
    # ------------------------------------------------------------------

    @property
    def _prisma_client(self):
        from litellm.proxy.proxy_server import prisma_client

        if prisma_client is None:
            raise RuntimeError(
                "Database not connected. Connect a database to your proxy — "
                "https://docs.litellm.ai/docs/simple_proxy#managing-auth---virtual-keys"
            )
        return prisma_client

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def export(
        self,
        date_str: str,
        connection_id: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Tuple[pl.DataFrame, str]:
        """Fetch, filter, and serialize spend data for one calendar date.

        Args:
            date_str:      Date in YYYY-MM-DD format.
            connection_id: Added as a column in the CSV output.
            limit:         Cap on rows fetched from the database.

        Returns:
            (filtered_df, csv_str) — caller uses len(filtered_df) for the
            record count and csv_str for the upload payload.
            Returns (empty DataFrame, "") when there is no data.
        """
        df = await self._get_usage_data(date_str=date_str, limit=limit)
        df = self.filter(df)
        csv = self._to_csv(df, connection_id=connection_id)
        return df, csv

    async def _stream_pages(
        self,
        date_str: str,
        connection_id: Optional[str] = None,
        page_size: int = 10_000,
    ) -> AsyncIterator[str]:
        """Yield CSV text in pages — header first, then one page of rows at a time.

        Uses LIMIT/OFFSET pagination so only page_size rows are in memory at once.
        Stops when a page returns fewer rows than page_size (last page reached).
        Called exclusively by Uploader._stream_upload() via Orchestrator._export().
        """
        # Build a one-row DataFrame just to get the header columns from the schema.
        # We yield the header once then write subsequent pages without it.
        header_written = False
        offset = 0

        while True:
            client = self._prisma_client
            rows = await client.db.query_raw(
                _USAGE_QUERY + " LIMIT $2 OFFSET $3",
                date_str,
                page_size,
                offset,
            )

            if not rows:
                if not header_written:
                    # Empty table — yield header only so caller knows columns
                    yield _USAGE_HEADER
                break

            df = pl.DataFrame(rows, infer_schema_length=None)
            df = self.filter(df)

            if not header_written:
                # First non-empty page — derive header from actual columns
                if connection_id:
                    df = df.with_columns(pl.lit(connection_id).alias("connection_id"))
                buf = io.StringIO()
                df.write_csv(buf)
                yield buf.getvalue()
                header_written = True
            else:
                if connection_id:
                    df = df.with_columns(pl.lit(connection_id).alias("connection_id"))
                buf = io.StringIO()
                df.write_csv(buf, include_header=False)
                yield buf.getvalue()

            offset += page_size
            if len(rows) < page_size:
                break  # last page — fewer rows than requested

    async def get_earliest_date(self) -> Optional[str]:
        """Return the earliest date string (YYYY-MM-DD) in LiteLLM_DailyUserSpend, or None."""
        client = self._prisma_client
        rows = await client.db.query_raw(_EARLIEST_DATE_QUERY)
        if rows and rows[0].get("earliest") is not None:
            return str(rows[0]["earliest"])[:10]
        return None

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    async def _get_usage_data(
        self,
        date_str: str,
        limit: Optional[int] = None,
    ) -> pl.DataFrame:
        """Retrieve raw spend rows for a single calendar date."""
        client = self._prisma_client

        query = _USAGE_QUERY
        params: List[Any] = [date_str]

        if limit is not None:
            params.append(int(limit))
            query += " LIMIT $2"

        db_response = await client.db.query_raw(query, *params)
        return pl.DataFrame(db_response, infer_schema_length=None)

    def filter(self, df: pl.DataFrame) -> pl.DataFrame:
        """Drop rows with zero successful_requests — no billable output."""
        if "successful_requests" not in df.columns:
            return df
        return df.filter(pl.col("successful_requests") > 0)

    def _to_csv(self, df: pl.DataFrame, connection_id: Optional[str] = None) -> str:
        """Serialize a filtered DataFrame to CSV, adding connection_id column if provided."""
        if df.is_empty():
            verbose_proxy_logger.debug("Exporter: empty DataFrame, nothing to export")
            return ""

        if connection_id:
            df = df.with_columns(pl.lit(connection_id).alias("connection_id"))

        buf = io.StringIO()
        df.write_csv(buf)
        csv_str = buf.getvalue()

        verbose_proxy_logger.debug(
            "Exporter: %d rows → %d CSV bytes", len(df), len(csv_str)
        )
        return csv_str
