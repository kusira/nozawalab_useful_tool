"""画像ファイルの階層ツリー構築。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

SUPPORTED_EXTENSIONS = {".npy", ".raw", ".png", ".jpg", ".jpeg"}


def is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


@dataclass
class DirNode:
    path: Path
    files: list[Path] = field(default_factory=list)
    children: list[DirNode] = field(default_factory=list)


def directory_has_images(path: Path) -> bool:
    if path.is_file():
        return is_supported_image(path)
    if not path.is_dir():
        return False
    try:
        for item in path.iterdir():
            if item.is_file() and is_supported_image(item):
                return True
            if item.is_dir() and directory_has_images(item):
                return True
    except OSError:
        return False
    return False


def build_dir_node(path: Path) -> DirNode | None:
    if not path.is_dir() or not directory_has_images(path):
        return None

    files: list[Path] = []
    children: list[DirNode] = []
    try:
        entries = sorted(path.iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return None

    for item in entries:
        if item.is_file() and is_supported_image(item):
            files.append(item)
        elif item.is_dir():
            child = build_dir_node(item)
            if child is not None:
                children.append(child)

    return DirNode(path=path, files=files, children=children)


def format_dir_number(index: int) -> str:
    return f"D{index:04d}"


def format_file_number(index: int) -> str:
    return f"{index + 1:04d}"
