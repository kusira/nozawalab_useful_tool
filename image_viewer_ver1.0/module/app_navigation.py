"""ファイル一覧・ナビゲーション・画像読み込み関連のミックスイン。"""

from __future__ import annotations

import random
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from module.dialogs import RawSettingsDialog
from module.file_tree import (
    DirNode,
    build_dir_node,
    directory_has_images,
    format_dir_number,
    format_file_number,
)
from module.image_loader import guess_raw_settings, is_supported_image, load_image_file


class NavigationMixin:
    """ファイルツリーの構築とファイル間の移動を担当する。"""

    @staticmethod
    def format_list_number(index: int) -> str:
        return format_file_number(index)

    def open_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="画像ファイルを選択",
            filetypes=[
                ("対応形式", "*.npy *.raw *.png *.jpg *.jpeg"),
                ("NumPy", "*.npy"),
                ("RAW", "*.raw"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("すべて", "*.*"),
            ],
        )
        if not paths:
            return
        self._load_paths([Path(p) for p in paths])

    def open_directory(self) -> None:
        path = filedialog.askdirectory(title="フォルダを選択")
        if not path:
            return
        root = Path(path)
        if not directory_has_images(root):
            messagebox.showwarning("フォルダ", "対応形式の画像が見つかりませんでした。")
            return
        self._load_paths([root])

    def add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="追加する画像ファイルを選択",
            filetypes=[("対応形式", "*.npy *.raw *.png *.jpg *.jpeg"), ("すべて", "*.*")],
        )
        if not paths:
            return
        new_paths = [Path(p) for p in paths if is_supported_image(Path(p))]
        if not new_paths:
            return
        for p in new_paths:
            if p not in self.root_sources:
                self.root_sources.append(p)
        self._rebuild_file_tree(preserve_path=self.current_path)
        if self.current_index < 0 and self.file_list:
            self.show_index(0)

    def _load_paths(self, paths: list[Path]) -> None:
        valid = [
            p
            for p in paths
            if p.exists()
            and ((p.is_file() and is_supported_image(p)) or (p.is_dir() and directory_has_images(p)))
        ]
        if not valid:
            messagebox.showwarning("読み込み", "有効なパスがありません。")
            return

        self.root_sources = valid
        self._rebuild_file_tree()
        if self.file_list:
            self.show_index(0)

    def _rebuild_file_tree(self, *, preserve_path: Path | None = None) -> None:
        self.file_list = []
        self._file_tree_iids = {}
        self.file_tree.delete(*self.file_tree.get_children(""))

        dir_counter = [0]
        for root in self.root_sources:
            if root.is_file() and is_supported_image(root):
                self._insert_file_node("", root)
            elif root.is_dir():
                node = build_dir_node(root)
                if node is not None:
                    self._insert_dir_node("", node, dir_counter)

        if preserve_path is not None and preserve_path in self.file_list:
            self.show_index(self.file_list.index(preserve_path))
        elif self.current_index >= len(self.file_list):
            if self.file_list:
                self.show_index(len(self.file_list) - 1, from_tree=True)

    def _insert_dir_node(self, parent: str, node: DirNode, dir_counter: list[int]) -> str:
        dir_counter[0] += 1
        label = f"{format_dir_number(dir_counter[0])}  {node.path.name}/"
        iid = self.file_tree.insert(parent, tk.END, text=label, tags=("dir",), open=True)
        for file_path in node.files:
            self._insert_file_node(iid, file_path)
        for child in node.children:
            self._insert_dir_node(iid, child, dir_counter)
        return iid

    def _insert_file_node(self, parent: str, path: Path) -> str:
        file_index = len(self.file_list)
        self.file_list.append(path)
        label = f"{self.format_list_number(file_index)}  {path.name}"
        iid = self.file_tree.insert(
            parent,
            tk.END,
            text=label,
            tags=("file",),
            values=(str(file_index),),
        )
        self._file_tree_iids[file_index] = iid
        return iid

    def _file_index_from_iid(self, iid: str) -> int | None:
        if not iid:
            return None
        tags = self.file_tree.item(iid, "tags")
        if "file" not in tags:
            return None
        values = self.file_tree.item(iid, "values")
        if not values:
            return None
        return int(values[0])

    def _on_tree_select(self, _event: tk.Event) -> None:
        iid = self.file_tree.focus()
        index = self._file_index_from_iid(iid) if iid else None
        if index is not None:
            self.show_index(index, from_tree=True)

    def _on_tree_activate(self, _event: tk.Event) -> None:
        iid = self.file_tree.focus()
        index = self._file_index_from_iid(iid) if iid else None
        if index is not None:
            self.show_index(index)

    def show_prev(self) -> None:
        if not self.file_list:
            return
        if self.current_index <= 0:
            self.show_index(len(self.file_list) - 1)
        else:
            self.show_index(self.current_index - 1)

    def show_next(self) -> None:
        if not self.file_list:
            return
        mode = self.browse_mode.get()
        if mode == "ランダム":
            self.show_random()
            return
        if mode == "番号指定":
            self.jump_to_index()
            return
        if self.current_index >= len(self.file_list) - 1:
            self.show_index(0)
        else:
            self.show_index(self.current_index + 1)

    def show_random(self) -> None:
        if not self.file_list:
            return
        if len(self.file_list) == 1:
            self.show_index(0)
            return
        candidates = list(range(len(self.file_list)))
        if self.current_index in candidates:
            candidates.remove(self.current_index)
        self.show_index(random.choice(candidates))

    def show_first(self) -> None:
        if self.file_list:
            self.show_index(0)

    def show_last(self) -> None:
        if self.file_list:
            self.show_index(len(self.file_list) - 1)

    def jump_to_index(self) -> None:
        if not self.file_list:
            return
        text = self.index_var.get().strip()
        if not text.isdigit():
            messagebox.showwarning("番号", "番号には整数を入力してください。")
            return
        number = int(text)
        if number < 1 or number > len(self.file_list):
            messagebox.showwarning(
                "番号",
                f"1〜{self.format_list_number(len(self.file_list) - 1)} の番号を入力してください。",
            )
            return
        self.show_index(number - 1)

    def show_index(self, index: int, *, from_tree: bool = False) -> None:
        if not self.file_list or index < 0 or index >= len(self.file_list):
            return
        self.current_index = index
        if not from_tree:
            iid = self._file_tree_iids.get(index)
            if iid:
                self.file_tree.selection_set(iid)
                self.file_tree.focus(iid)
                self.file_tree.see(iid)
        self.index_var.set(self.format_list_number(index))
        self.load_file(self.file_list[index])

    def load_file(self, path: Path) -> None:
        try:
            suffix = path.suffix.lower()
            if suffix == ".raw" or (suffix not in (".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")):
                settings = self._ask_raw_settings(path)
                if settings is None and suffix not in (".npy", ".png", ".jpg", ".jpeg"):
                    return
                if settings is not None:
                    self.raw_settings = settings
                array, image, file_type = load_image_file(path, self.raw_settings)
            else:
                array, image, file_type = load_image_file(path)
        except Exception as exc:
            messagebox.showerror("読み込みエラー", f"{path.name}\n{exc}")
            return

        self.current_path = path
        self.source_array = array
        self.base_image = image
        self.current_file_type = file_type
        self.path_label.config(text=str(path))
        self._update_meta(path, array)
        self.refresh_view()
        self.root.after(50, self.fit_view)

    def _ask_raw_settings(self, path: Path) -> dict[str, object] | None:
        suffix = path.suffix.lower()
        if suffix in (".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
            return None
        guessed = guess_raw_settings(path, self.raw_settings)
        dialog = RawSettingsDialog(self.root, path, guessed)
        self.root.wait_window(dialog)
        return dialog.result

    def reload_raw(self) -> None:
        if self.current_path is None or self.current_file_type != "raw":
            messagebox.showinfo("RAW", "RAWファイルを開いてください。")
            return
        settings = self._ask_raw_settings(self.current_path)
        if settings is None:
            return
        try:
            self.raw_settings = settings
            array, image, _ = load_image_file(self.current_path, settings)
        except Exception as exc:
            messagebox.showerror("読み込みエラー", str(exc))
            return
        self.source_array = array
        self.base_image = image
        self._update_meta(self.current_path, array)
        self.refresh_view()
