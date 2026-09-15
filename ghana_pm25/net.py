"""Small HTTP helpers with retries, polite back-off and on-disk caching."""
from __future__ import annotations

import time
from pathlib import Path

import requests

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "ghana-pm25-research/1.0 (air-quality downscaling)"})


def get(url: str, params: dict | None = None, timeout: int = 180, retries: int = 6, **kw) -> requests.Response:
    delay = 5.0
    last = None
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=timeout, **kw)
            if r.status_code == 429 or r.status_code >= 500:
                last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                wait = delay * (2 ** attempt)
                if r.status_code == 429:
                    wait = max(wait, 65)
                time.sleep(wait)
                continue
            return r
        except requests.RequestException as e:  # network hiccup
            last = e
            time.sleep(delay * (2 ** attempt))
    raise RuntimeError(f"GET failed after {retries} attempts: {url} ({last})")


def download(url: str, dest: Path, chunk: int = 1 << 20, overwrite: bool = False, timeout: int = 600) -> Path:
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 0 and not overwrite:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    import os
    import threading
    tmp = dest.with_suffix(dest.suffix + f".{os.getpid()}.{threading.get_ident()}.part")
    for attempt in range(5):
        try:
            with SESSION.get(url, stream=True, timeout=timeout) as r:
                if r.status_code in (403, 404):
                    raise RuntimeError(f"HTTP {r.status_code}: {url}")
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for block in r.iter_content(chunk):
                        f.write(block)
            tmp.replace(dest)
            return dest
        except requests.RequestException:
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"download failed: {url}")
