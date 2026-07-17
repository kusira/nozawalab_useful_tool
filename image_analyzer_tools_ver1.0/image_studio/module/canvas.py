"""ズーム・パン・ROI/ライン描画・ルーペ対応の画像キャンバス。

image_viewer（タイル描画・ルーペ・カーソル追従ズーム）と
image_analyzer（ROI/ラインの描画操作）の両方の機能を統合している。
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk


class MagnifierWindow(tk.Toplevel):
    """マウスホバー位置を拡大表示するルーペウィンドウ。"""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent)
        self.withdraw()
        self.overrideredirect(True)
        self.attributes("-topmost", True)
        self.label = ttk.Label(self, relief=tk.SOLID, borderwidth=1)
        self.label.pack()
        self._photo: ImageTk.PhotoImage | None = None

    def show_patch(self, patch: Image.Image, screen_x: int, screen_y: int) -> None:
        self._photo = ImageTk.PhotoImage(patch)
        self.label.config(image=self._photo)
        self.geometry(f"+{screen_x + 18}+{screen_y + 18}")
        self.deiconify()

    def hide(self) -> None:
        self.withdraw()


class ZoomableCanvas(ttk.Frame):
    """ズーム・パン・ROI/ライン描画対応の画像キャンバス。"""

    PAN_SPEED = 0.5

    def __init__(
        self,
        parent: tk.Misc,
        *,
        on_cursor_move: Callable[..., None] | None = None,
        on_drag: Callable[..., None] | None = None,
        on_drag_end: Callable[..., None] | None = None,
        on_click: Callable[..., None] | None = None,
    ) -> None:
        super().__init__(parent)
        self.on_cursor_move = on_cursor_move
        self.on_drag = on_drag
        self.on_drag_end = on_drag_end
        self.on_click = on_click

        self.canvas = tk.Canvas(self, background="#1e1e1e", highlightthickness=0)
        h_scroll = ttk.Scrollbar(self, orient=tk.HORIZONTAL, command=self._on_hscroll)
        v_scroll = ttk.Scrollbar(self, orient=tk.VERTICAL, command=self._on_vscroll)
        self.canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        self.canvas.grid(row=0, column=0, sticky="nsew")
        h_scroll.grid(row=1, column=0, sticky="ew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self._image: Image.Image | None = None
        self._source_array: np.ndarray | None = None
        self._photo: ImageTk.PhotoImage | None = None
        self._zoom = 1.0
        self._resampling: Image.Resampling = Image.Resampling.NEAREST
        self._pan_start: tuple[int, int] | None = None
        self._pan_remainder_x = 0.0
        self._pan_remainder_y = 0.0
        self._drag_start: tuple[int, int] | None = None
        self.interaction_mode = "none"  # none / rect / circle / freehand / line

        self.canvas.bind("<ButtonPress-2>", self._on_pan_start)
        self.canvas.bind("<B2-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-2>", self._on_pan_end)
        self.canvas.bind("<ButtonPress-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-3>", self._on_pan_end)
        self.canvas.bind("<Control-ButtonPress-1>", self._on_pan_start)
        self.canvas.bind("<Control-B1-Motion>", self._on_pan_move)
        self.canvas.bind("<Control-ButtonRelease-1>", self._on_pan_end)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._on_leave)
        self.canvas.bind("<ButtonPress-1>", self._on_left_press)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_release)
        self.canvas.bind("<Configure>", lambda _e: self._redraw())

    # ---------- state ----------
    def set_interaction_mode(self, mode: str) -> None:
        self.interaction_mode = mode

    def set_image(self, image: Image.Image | None, source_array: np.ndarray | None = None) -> None:
        self._image = image
        self._source_array = source_array
        self._redraw()

    def get_image(self) -> Image.Image | None:
        return self._image

    def set_zoom(self, zoom: float) -> None:
        self._zoom = max(0.05, min(32.0, zoom))
        self._redraw()

    def get_zoom(self) -> float:
        return self._zoom

    def set_resampling(self, resampling: Image.Resampling) -> None:
        if resampling != self._resampling:
            self._resampling = resampling
            self._redraw()

    # ---------- scrolling ----------
    def _on_hscroll(self, *args: str) -> None:
        self.canvas.xview(*args)
        self._redraw()

    def _on_vscroll(self, *args: str) -> None:
        self.canvas.yview(*args)
        self._redraw()

    def fit_to_window(self) -> None:
        if self._image is None:
            return
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        zx = cw / self._image.width
        zy = ch / self._image.height
        self._zoom = min(zx, zy, 1.0)
        self._redraw()
        self.canvas.xview_moveto(0)
        self.canvas.yview_moveto(0)
        self._redraw()

    def actual_size(self) -> None:
        self._zoom = 1.0
        self._redraw()

    # ---------- coordinate helpers ----------
    def canvas_to_image(self, canvas_x: int, canvas_y: int) -> tuple[int, int] | None:
        if self._image is None:
            return None
        cx = self.canvas.canvasx(canvas_x)
        cy = self.canvas.canvasy(canvas_y)
        ix = int(cx / self._zoom)
        iy = int(cy / self._zoom)
        ix = max(0, min(ix, self._image.width - 1))
        iy = max(0, min(iy, self._image.height - 1))
        return ix, iy

    def get_pixel_value(self, x: int, y: int) -> str:
        if self._source_array is not None:
            arr = self._source_array
            if arr.ndim == 2:
                return f"{arr[y, x]}"
            if arr.ndim >= 3:
                vals = arr[y, x]
                if np.ndim(vals) == 0:
                    return f"{vals}"
                return ", ".join(str(v) for v in np.asarray(vals).ravel()[:4])
        if self._image is None:
            return "-"
        px = self._image.getpixel((x, y))
        if isinstance(px, int):
            return str(px)
        return ", ".join(str(v) for v in px)

    def extract_patch(self, x: int, y: int, half_size: int, zoom_factor: int = 8) -> Image.Image | None:
        if self._image is None:
            return None
        left = max(0, x - half_size)
        top = max(0, y - half_size)
        right = min(self._image.width, x + half_size + 1)
        bottom = min(self._image.height, y + half_size + 1)
        patch = self._image.crop((left, top, right, bottom))
        target = half_size * 2 + 1
        display_size = max(target * zoom_factor, target)
        return patch.resize((display_size, display_size), Image.Resampling.NEAREST)

    # ---------- drawing ----------
    def _redraw(self) -> None:
        self.canvas.delete("all")
        if self._image is None:
            self.canvas.create_text(
                self.canvas.winfo_width() // 2 or 200,
                self.canvas.winfo_height() // 2 or 150,
                text="画像を開いてください",
                fill="#aaaaaa",
                font=("", 12),
            )
            self.canvas.config(scrollregion=(0, 0, 1, 1))
            return

        zoom = self._zoom
        img_w, img_h = self._image.width, self._image.height
        disp_w = max(1, int(round(img_w * zoom)))
        disp_h = max(1, int(round(img_h * zoom)))
        self.canvas.config(scrollregion=(0, 0, disp_w, disp_h))

        # 表示中の領域（ズーム後座標）だけを描画してメモリ消費を抑える。
        view_w = max(1, self.canvas.winfo_width())
        view_h = max(1, self.canvas.winfo_height())
        left = max(0.0, self.canvas.canvasx(0))
        top = max(0.0, self.canvas.canvasy(0))
        right = min(float(disp_w), left + view_w)
        bottom = min(float(disp_h), top + view_h)
        if right <= left or bottom <= top:
            return

        # ズーム後座標 → 元画像座標へ変換し、可視範囲を切り出す。
        src_left = max(0, min(img_w - 1, int(left / zoom)))
        src_top = max(0, min(img_h - 1, int(top / zoom)))
        src_right = max(src_left + 1, min(img_w, int(right / zoom) + 1))
        src_bottom = max(src_top + 1, min(img_h, int(bottom / zoom) + 1))

        region = self._image.crop((src_left, src_top, src_right, src_bottom))
        region_w = max(1, int(round((src_right - src_left) * zoom)))
        region_h = max(1, int(round((src_bottom - src_top) * zoom)))
        if (region.width, region.height) != (region_w, region_h):
            region = region.resize((region_w, region_h), self._resampling)

        self._photo = ImageTk.PhotoImage(region)
        self.canvas.create_image(
            int(round(src_left * zoom)),
            int(round(src_top * zoom)),
            image=self._photo,
            anchor=tk.NW,
        )

    # ---------- zoom / pan ----------
    def _on_mousewheel(self, event: tk.Event) -> None:
        if self._image is None:
            return
        factor = 1.15 if event.delta > 0 else 1 / 1.15
        old_zoom = self._zoom
        new_zoom = max(0.05, min(32.0, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-6:
            return

        # カーソル下の画像座標がズーム後も同じ位置に留まるようにする。
        img_x = self.canvas.canvasx(event.x) / old_zoom
        img_y = self.canvas.canvasy(event.y) / old_zoom
        self._zoom = new_zoom

        disp_w = max(1, int(round(self._image.width * new_zoom)))
        disp_h = max(1, int(round(self._image.height * new_zoom)))
        self.canvas.config(scrollregion=(0, 0, disp_w, disp_h))
        new_left = img_x * new_zoom - event.x
        new_top = img_y * new_zoom - event.y
        self.canvas.xview_moveto(max(0.0, new_left / disp_w))
        self.canvas.yview_moveto(max(0.0, new_top / disp_h))
        self._redraw()

    def _on_pan_start(self, event: tk.Event) -> None:
        self._pan_start = (event.x, event.y)
        self._pan_remainder_x = 0.0
        self._pan_remainder_y = 0.0
        self.canvas.config(cursor="fleur")

    def _on_pan_move(self, event: tk.Event) -> None:
        if self._pan_start is None:
            return
        dx = (event.x - self._pan_start[0]) * self.PAN_SPEED
        dy = (event.y - self._pan_start[1]) * self.PAN_SPEED
        self._pan_start = (event.x, event.y)
        self._pan_remainder_x -= dx
        self._pan_remainder_y -= dy
        scroll_x = int(self._pan_remainder_x)
        scroll_y = int(self._pan_remainder_y)
        self._pan_remainder_x -= scroll_x
        self._pan_remainder_y -= scroll_y
        if scroll_x:
            self.canvas.xview_scroll(scroll_x, "units")
        if scroll_y:
            self.canvas.yview_scroll(scroll_y, "units")
        if scroll_x or scroll_y:
            self._redraw()

    def _on_pan_end(self, _event: tk.Event) -> None:
        self._pan_start = None
        self.canvas.config(cursor="arrow")

    # ---------- cursor / interaction ----------
    def _on_motion(self, event: tk.Event) -> None:
        if self.on_cursor_move is None or self._image is None:
            return
        pt = self.canvas_to_image(event.x, event.y)
        if pt is None:
            return
        self.on_cursor_move(pt[0], pt[1], event)

    def _on_leave(self, _event: tk.Event) -> None:
        if self.on_cursor_move is not None:
            self.on_cursor_move(-1, -1, None)

    def _on_left_press(self, event: tk.Event) -> None:
        if self._image is None:
            return
        pt = self.canvas_to_image(event.x, event.y)
        if pt is None:
            return
        if self.interaction_mode == "none":
            if self.on_click is not None:
                self.on_click(pt[0], pt[1])
            return
        self._drag_start = pt
        if self.on_drag is not None:
            self.on_drag(pt[0], pt[1], pt[0], pt[1], "start")

    def _on_left_drag(self, event: tk.Event) -> None:
        if self._drag_start is None or self._image is None or self.on_drag is None:
            return
        pt = self.canvas_to_image(event.x, event.y)
        if pt is None:
            return
        self.on_drag(self._drag_start[0], self._drag_start[1], pt[0], pt[1], "move")

    def _on_left_release(self, event: tk.Event) -> None:
        if self._drag_start is None or self._image is None:
            return
        pt = self.canvas_to_image(event.x, event.y)
        start = self._drag_start
        self._drag_start = None
        if pt is None:
            return
        if self.on_drag_end is not None and self.interaction_mode != "none":
            self.on_drag_end(start[0], start[1], pt[0], pt[1])
