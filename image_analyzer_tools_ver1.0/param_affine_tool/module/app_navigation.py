"""ファイル一覧・ナビゲーション・画像読み込み関連のミックスイン。"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

from common.constants import FILE_DIALOG_TYPES
from common.dialogs import RawSettingsDialog
from module.file_tree import (
    DirNode,
    build_dir_node,
    directory_has_images,
    format_dir_number,
    format_file_number,
    is_supported_image,
)


class NavigationMixin:
    """ファイルツリーの構築・ファイル間の移動・読み込みの起点を担当する。"""

    def open_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="画像ファイルを選択",
            filetypes=FILE_DIALOG_TYPES,
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
            filetypes=FILE_DIALOG_TYPES,
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
            self.show_index(self.file_list.index(preserve_path), from_tree=True)
        elif self.current_index >= len(self.file_list):
            if self.file_list:
                self.show_index(len(self.file_list) - 1, from_tree=True)
            else:
                self.current_index = -1
                self.index_var.set("")

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
        label = f"{format_file_number(file_index)}  {path.name}"
        iid = self.file_tree.insert(parent, tk.END, text=label, tags=("file",), values=(str(file_index),))
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
        if self.current_index >= len(self.file_list) - 1:
            self.show_index(0)
        else:
            self.show_index(self.current_index + 1)

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
                f"1〜{format_file_number(len(self.file_list) - 1)} の番号を入力してください。",
            )
            return
        self.show_index(number - 1)

    def show_index(self, index: int, *, from_tree: bool = False) -> None:
        if not self.file_list or index < 0 or index >= len(self.file_list):
            return
        self.current_index = index
        path = self.file_list[index]
        self.index_var.set(format_file_number(index))
        if not from_tree:
            iid = self._file_tree_iids.get(index)
            if iid:
                self.file_tree.selection_set(iid)
                self.file_tree.focus(iid)
                self.file_tree.see(iid)
        self._load_image_path(path)

    def _load_image_path(self, file_path: Path) -> None:
        suffix = file_path.suffix.lower()
        try:
            if suffix == ".npy":
                array = self.load_npy(file_path)
                image = self.array_to_image(array)
                self.current_file_type = "npy"
                self.reload_button.config(state=tk.DISABLED)
            elif suffix == ".png":
                array, image = self.load_raster(file_path)
                self.current_file_type = "png"
                self.reload_button.config(state=tk.DISABLED)
            elif suffix in (".jpg", ".jpeg"):
                array, image = self.load_raster(file_path)
                self.current_file_type = "jpeg"
                self.reload_button.config(state=tk.DISABLED)
            elif suffix == ".raw":
                settings = self._ask_raw_settings(file_path)
                if settings is None:
                    return
                self.raw_settings = settings
                array = self.load_raw(file_path, settings)
                image = self.array_to_image(array)
                self.current_file_type = "raw"
                self.reload_button.config(state=tk.NORMAL)
            else:
                messagebox.showerror("読み込みエラー", f"未対応の形式です: {suffix}")
                return
        except Exception as exc:
            messagebox.showerror("読み込みエラー", f"ファイルを読み込めませんでした。\n{exc}")
            return

        self.current_path = file_path
        self.source_array = array
        self.base_image = image
        self.cropped_image = None
        self.cropped_array = None
        self.cancel_crop_mode()
        self.path_label.config(text=str(file_path))
        self._update_meta(array, file_path)
        self.reset_params(update_preview=False)
        self._clear_affine_preview()
        self.landmark_button.config(state=tk.NORMAL)
        self.save_preview_button.config(state=tk.DISABLED)
        self.crop_select_button.config(state=tk.NORMAL)
        self._update_crop_reset_button_state()
        self.update_preview()

    def reload_raw(self) -> None:
        if self.current_path is None or self.current_file_type != "raw":
            return

        settings = self._ask_raw_settings(self.current_path)
        if settings is None:
            return

        try:
            self.raw_settings = settings
            array = self.load_raw(self.current_path, settings)
            image = self.array_to_image(array)
        except Exception as exc:
            messagebox.showerror("読み込みエラー", f"RAWファイルを読み込めませんでした。\n{exc}")
            return

        self.source_array = array
        self.base_image = image
        self.cropped_image = None
        self.cropped_array = None
        self.cancel_crop_mode()
        self._update_crop_reset_button_state()
        self._update_meta(array, self.current_path)
        self._clear_affine_preview()
        self.save_preview_button.config(state=tk.DISABLED)
        self.update_preview()

    def _ask_raw_settings(self, path: Path) -> dict[str, object] | None:
        guessed = self._guess_raw_settings(path, self.raw_settings)
        dialog = RawSettingsDialog(self.root, path, guessed)
        self.root.wait_window(dialog)
        return dialog.result
