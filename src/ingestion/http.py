"""Shared HTTP helpers: retrying GET, and a seekable remote-file adapter.

The remote-file adapter lets ``zipfile`` read a multi-hundred-megabyte archive
straight out of S3 over HTTP range requests, so the trip-history loader never
has to download and unpack a full month to local disk.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Any

import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 30
DEFAULT_RETRIES = 3


def get_with_retries(
    url: str,
    *,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    retries: int = DEFAULT_RETRIES,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    backoff_s: float = 1.5,
) -> requests.Response:
    """GET with exponential backoff.

    Raises the last exception if every attempt fails; callers in DAGs are
    expected to catch this and skip the run rather than crash the scheduler.
    """
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            # (connect, read) rather than one value. Even so, `requests` applies
            # the read timeout between socket reads, so a server trickling bytes
            # can exceed it many times over - the caller's wall-clock budget is
            # the real guard. This just fails faster on a dead connection.
            response = requests.get(
                url,
                timeout=(min(10, timeout_s), timeout_s),
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            return response
        except Exception as exc:  # noqa: BLE001 - deliberately broad, we retry everything
            last_error = exc
            if attempt < retries:
                sleep_for = backoff_s ** attempt
                logger.warning(
                    "GET %s failed (attempt %d/%d): %s - retrying in %.1fs",
                    url, attempt, retries, exc, sleep_for,
                )
                time.sleep(sleep_for)
            else:
                logger.error("GET %s failed after %d attempts: %s", url, retries, exc)
    assert last_error is not None
    raise last_error


class HttpRangeFile(io.RawIOBase):
    """A read-only, seekable file-like object backed by HTTP range requests.

    ``zipfile.ZipFile`` needs to seek to the central directory at the end of the
    archive and then to individual member offsets. Wrapping this in an
    ``io.BufferedReader`` with a large buffer keeps the number of range requests
    low while still avoiding a full download.
    """

    def __init__(self, url: str, *, timeout_s: int = 120, retries: int = DEFAULT_RETRIES) -> None:
        self._url = url
        self._timeout_s = timeout_s
        self._retries = retries
        self._pos = 0
        head = self._head()
        if "Content-Length" not in head.headers:
            raise ValueError(f"Server did not report Content-Length for {url}")
        if head.headers.get("Accept-Ranges", "").lower() != "bytes":
            # S3 supports ranges even when it does not advertise them on HEAD,
            # so this is a warning rather than a hard failure.
            logger.warning("Server did not advertise byte ranges for %s; trying anyway", url)
        self._size = int(head.headers["Content-Length"])

    def _head(self) -> requests.Response:
        last_error: Exception | None = None
        for attempt in range(1, self._retries + 1):
            try:
                response = requests.head(self._url, timeout=self._timeout_s, allow_redirects=True)
                response.raise_for_status()
                return response
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(1.5 ** attempt)
        assert last_error is not None
        raise last_error

    @property
    def size(self) -> int:
        return self._size

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            new_pos = offset
        elif whence == io.SEEK_CUR:
            new_pos = self._pos + offset
        elif whence == io.SEEK_END:
            new_pos = self._size + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        self._pos = max(0, min(new_pos, self._size))
        return self._pos

    def tell(self) -> int:
        return self._pos

    def readinto(self, buffer: bytearray) -> int:  # type: ignore[override]
        if self._pos >= self._size:
            return 0
        end = min(self._pos + len(buffer), self._size) - 1
        headers = {"Range": f"bytes={self._pos}-{end}"}
        response = get_with_retries(
            self._url, timeout_s=self._timeout_s, retries=self._retries, headers=headers
        )
        chunk = response.content
        buffer[: len(chunk)] = chunk
        self._pos += len(chunk)
        return len(chunk)


def open_remote_zip_stream(url: str, buffer_size: int = 1 << 22) -> io.BufferedReader:
    """Return a buffered, seekable stream over a remote archive."""
    return io.BufferedReader(HttpRangeFile(url), buffer_size=buffer_size)
