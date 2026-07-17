"""アプリ共通の定数・小さなユーティリティ。"""

from __future__ import annotations

from pathlib import Path


def default_downloads_dir() -> Path:
    downloads = Path.home() / "Downloads"
    if downloads.is_dir():
        return downloads
    return Path.home()
