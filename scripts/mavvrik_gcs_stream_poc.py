"""Proof-of-concept: GCS chunked resumable upload with streaming gzip.

Tests whether GCS accepts a chunked PUT with on-the-fly gzip compression,
without materialising the full payload in memory first.

Usage:
    MAVVRIK_API_KEY=... \
    MAVVRIK_API_ENDPOINT=https://api.mavvrik.dev/<tenant> \
    MAVVRIK_CONNECTION_ID=... \
    poetry run python scripts/mavvrik_gcs_stream_poc.py [--rows N] [--page-size N] [--chunk-kb N]

Examples:
    # default: 50k rows (matches current production limit)
    poetry run python scripts/mavvrik_gcs_stream_poc.py

    # large dataset stress test: 500k rows — would overflow current 50k limit
    poetry run python scripts/mavvrik_gcs_stream_poc.py --rows 500000

    # tune chunk size (must stay a multiple of 256)
    poetry run python scripts/mavvrik_gcs_stream_poc.py --rows 500000 --chunk-kb 1024

Exit 0 = streaming upload works.
Exit 1 = something failed (check output).
"""

import argparse
import asyncio
import gzip
import io
import os
import sys
import time

import httpx

# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("MAVVRIK_API_KEY", "")
API_ENDPOINT = os.environ.get("MAVVRIK_API_ENDPOINT", "").rstrip("/")
CONNECTION_ID = os.environ.get("MAVVRIK_CONNECTION_ID", "")

if not all([API_KEY, API_ENDPOINT, CONNECTION_ID]):
    print("ERROR: set MAVVRIK_API_KEY, MAVVRIK_API_ENDPOINT, MAVVRIK_CONNECTION_ID")
    sys.exit(1)

# ---------------------------------------------------------------------------
# CLI args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="Mavvrik GCS streaming upload PoC")
parser.add_argument("--rows", type=int, default=50_000, help="Total CSV rows to generate (default: 50000)")
parser.add_argument("--page-size", type=int, default=10_000, help="Rows per page / DB fetch (default: 10000)")
parser.add_argument("--chunk-kb", type=int, default=256, help="GCS chunk size in KB — must be multiple of 256 (default: 256)")
parser.add_argument("--object-name", default="poc-streaming-test", help="GCS object name")
args = parser.parse_args()

if args.chunk_kb % 256 != 0:
    print(f"ERROR: --chunk-kb must be a multiple of 256 (got {args.chunk_kb})")
    sys.exit(1)

TOTAL_ROWS = args.rows
PAGE_SIZE = args.page_size
GCS_CHUNK_SIZE = args.chunk_kb * 1024
TEST_OBJECT_NAME = args.object_name

# ---------------------------------------------------------------------------
# Synthetic data generator
# ---------------------------------------------------------------------------

# Realistic LiteLLM spend row — ~120 bytes per row → 50k rows ≈ 6 MB raw CSV
# After gzip compression: ~300 KB (high repetition compresses ~95%)
# 500k rows: ~60 MB raw, ~3 MB gzipped
CSV_HEADER = (
    "date,user_id,api_key,model,model_group,custom_llm_provider,"
    "prompt_tokens,completion_tokens,spend,api_requests,successful_requests,"
    "failed_requests,cache_creation_input_tokens,cache_read_input_tokens,"
    "created_at,updated_at,team_id,api_key_alias,organization_id,"
    "team_alias,user_email,user_alias,connection_id\n"
)

def _row(i: int) -> str:
    team = i % 10
    model = ["gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro"][i % 3]
    return (
        f"2026-01-01,user-{i % 1000},sk-hash-{i % 500},{model},{model},openai,"
        f"100,50,0.0015,1,1,0,0,0,"
        f"2026-01-01T00:00:00Z,2026-01-01T00:01:00Z,"
        f"team-{team},alias-{i},org-1,"
        f"Team {team},user{i}@example.com,u{i},poc-conn\n"
    )


def generate_pages(total: int, page_size: int):
    """Yield CSV content piece by piece (header, then one row at a time).

    In production this would be replaced by async DB cursor / OFFSET pagination.
    Yields a progress log at each page boundary.
    """
    yield CSV_HEADER
    for i in range(total):
        yield _row(i)
        if (i + 1) % page_size == 0:
            pct = (i + 1) / total * 100
            print(f"  [data]  page complete: rows {i + 2 - page_size}–{i + 1} ({pct:.0f}%)")


# ---------------------------------------------------------------------------
# Step 1: Get signed URL from Mavvrik API
# ---------------------------------------------------------------------------

async def get_signed_url(http: httpx.AsyncClient) -> str:
    url = f"{API_ENDPOINT}/metrics/agent/ai/{CONNECTION_ID}/upload-url"
    today = __import__("datetime").date.today().isoformat()
    resp = await http.get(
        url,
        headers={"Content-Type": "application/json", "x-api-key": API_KEY},
        params={"name": today, "type": "metrics", "datetime": today},
        timeout=30.0,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"get_signed_url {resp.status_code}: {resp.text[:300]}")
    signed_url = resp.json().get("url")
    if not signed_url:
        raise RuntimeError(f"Response missing 'url': {resp.json()}")
    print(f"  [step 1] signed URL obtained ✓")
    return signed_url


# ---------------------------------------------------------------------------
# Step 2: Initiate GCS resumable upload session
# ---------------------------------------------------------------------------

async def initiate_session(http: httpx.AsyncClient, signed_url: str) -> str:
    metadata = b'{"contentEncoding":"gzip","contentDisposition":"attachment"}'
    resp = await http.post(
        signed_url,
        headers={"Content-Type": "application/gzip", "x-goog-resumable": "start"},
        content=metadata,
        timeout=30.0,
    )
    if resp.status_code != 201:
        raise RuntimeError(f"initiate_session {resp.status_code}: {resp.text[:300]}")
    session_uri = resp.headers.get("Location")
    if not session_uri:
        raise RuntimeError("initiate_session: missing Location header")
    print(f"  [step 2] resumable session created ✓")
    return session_uri


# ---------------------------------------------------------------------------
# Step 3: Stream gzip chunks to GCS session URI
# ---------------------------------------------------------------------------

async def stream_upload(http: httpx.AsyncClient, session_uri: str) -> dict:
    """
    Compress CSV rows on-the-fly and PUT in GCS_CHUNK_SIZE-aligned chunks.

    GCS resumable upload protocol:
      Intermediate chunks  →  Content-Range: bytes X-Y/*       →  308 Resume Incomplete
      Final chunk          →  Content-Range: bytes X-Y/<total>  →  200/201 OK

    Returns stats dict for reporting.
    """
    gz_buffer = bytearray()
    offset = 0
    chunks_sent = 0
    t_start = time.monotonic()

    # GzipFile wrapping a BytesIO lets us .write() + .flush() incrementally.
    # After each flush, raw_buf contains the newly compressed bytes.
    raw_buf = io.BytesIO()
    gz = gzip.GzipFile(fileobj=raw_buf, mode="wb")

    async def _put_chunk(chunk: bytes, final: bool) -> None:
        nonlocal offset, chunks_sent

        end = offset + len(chunk) - 1
        total_str = str(offset + len(chunk)) if final else "*"
        content_range = f"bytes {offset}-{end}/{total_str}"

        resp = await http.put(
            session_uri,
            headers={
                "Content-Type": "application/gzip",
                "Content-Range": content_range,
            },
            content=chunk,
            timeout=120.0,
        )

        expected = {200, 201} if final else {308}
        if resp.status_code not in expected:
            label = "final" if final else f"chunk {chunks_sent + 1}"
            raise RuntimeError(
                f"PUT {label} returned {resp.status_code} "
                f"(expected {expected}): {resp.text[:200]}"
            )

        offset += len(chunk)
        if not final:
            chunks_sent += 1
            elapsed = time.monotonic() - t_start
            throughput = offset / elapsed / 1024
            print(
                f"  [step 3] chunk {chunks_sent:3d} | "
                f"{offset / 1024:8,.0f} KB sent | "
                f"{throughput:6.0f} KB/s"
            )

    async def flush_full_chunks() -> None:
        nonlocal gz_buffer
        while len(gz_buffer) >= GCS_CHUNK_SIZE:
            chunk = bytes(gz_buffer[:GCS_CHUNK_SIZE])
            gz_buffer = gz_buffer[GCS_CHUNK_SIZE:]
            await _put_chunk(chunk, final=False)

    async def flush_remaining() -> None:
        nonlocal gz_buffer
        if gz_buffer:
            await _put_chunk(bytes(gz_buffer), final=True)
            gz_buffer.clear()

    # Feed CSV rows through gzip row-by-row (no full CSV in memory)
    for piece in generate_pages(TOTAL_ROWS, PAGE_SIZE):
        gz.write(piece.encode("utf-8"))
        gz.flush()

        new_compressed = raw_buf.getvalue()
        raw_buf.seek(0)
        raw_buf.truncate(0)
        gz_buffer.extend(new_compressed)

        await flush_full_chunks()

    # Close gzip (flushes internal buffer + writes gzip footer)
    gz.close()
    gz_buffer.extend(raw_buf.getvalue())
    await flush_remaining()

    elapsed = time.monotonic() - t_start
    return {
        "total_bytes": offset,
        "chunks": chunks_sent + 1,
        "elapsed_s": elapsed,
        "throughput_kbps": offset / elapsed / 1024 if elapsed > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print(f"\n{'=' * 60}")
    print(f"Mavvrik GCS Streaming Upload PoC")
    print(f"{'=' * 60}")
    print(f"  rows:          {TOTAL_ROWS:,}")
    print(f"  page size:     {PAGE_SIZE:,} rows  (simulated DB pages)")
    print(f"  chunk size:    {GCS_CHUNK_SIZE // 1024} KB")
    print(f"  object name:   {TEST_OBJECT_NAME}")
    raw_est_mb = TOTAL_ROWS * 120 / 1024 / 1024
    gz_est_mb = raw_est_mb * 0.05   # gzip typically 95% compression on repetitive CSV
    print(f"  est. raw CSV:  ~{raw_est_mb:.0f} MB")
    print(f"  est. gzipped:  ~{gz_est_mb:.1f} MB")
    print(f"  peak RAM:      ~{GCS_CHUNK_SIZE // 1024 + PAGE_SIZE * 120 // 1024 // 1024 + 1} MB  "
          f"(chunk buffer + one page)")
    print()

    async with httpx.AsyncClient() as http:
        signed_url = await get_signed_url(http)
        session_uri = await initiate_session(http, signed_url)
        print()
        stats = await stream_upload(http, session_uri)

    print()
    print(f"{'=' * 60}")
    print(f"✓  Upload complete")
    print(f"   Total bytes uploaded : {stats['total_bytes']:,} ({stats['total_bytes'] / 1024 / 1024:.2f} MB)")
    print(f"   Chunks sent          : {stats['chunks']}")
    print(f"   Elapsed              : {stats['elapsed_s']:.1f}s")
    print(f"   Throughput           : {stats['throughput_kbps']:.0f} KB/s")
    print(f"{'=' * 60}")
    print()
    print("Next step: verify in Mavvrik/GCS that the object exists")
    print("and can be decompressed:")
    print(f"  gsutil cat gs://<bucket>/{TEST_OBJECT_NAME} | gunzip | head -5")


if __name__ == "__main__":
    asyncio.run(main())
