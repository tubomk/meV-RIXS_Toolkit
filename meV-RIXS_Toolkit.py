# To create a onefile executable, use PyInstaller with the following command:
# On Windows (Assuming the icon File is icon8.ico and icon8.png is in the same directory as this script):
# pyinstaller --noconfirm --clean --onefile --windowed --name meV-RIXS_Toolkit --icon=icon8.ico --add-data "icon8.ico;." --add-data "icon8.png;." meV-RIXS_Toolkit.py
# On Linux or macOS (Assuming the icon File is icon8.png and icon8.ico is in the same directory as this script):
# python -m PyInstaller --noconfirm --clean --onefile --name meV-RIXS_Toolkit --add-data "icon8.png:." --hidden-import PIL._tkinter_finder meV-RIXS_Toolkit.py
# Main application window for configuring and running the meV-RIXS viewer.



from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tkinter import filedialog
from typing import cast, Literal, NotRequired, overload, TypedDict
from weakref import ReferenceType, ref

import copy
import ctypes
import json
import os
import sys
import platform
import tkinter as tk
import tkinter.font as tkfont

import customtkinter as ctk
import matplotlib

# Matplotlib must use the Tk backend before pyplot and the viewer are loaded.
matplotlib.use("TkAgg")

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.figure import Figure
from PIL import (
    Image,
    ImageChops,
    ImageColor,
    ImageDraw,
    ImageEnhance,
    ImageTk,
)
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D

import mev_viewer
from mev_viewer import dark_style


# =============================================================================
# Icon and image loading
# ============================================================================

def resource_path(relative_path: str) -> Path:
    base_path = Path(
        getattr(sys, "_MEIPASS", Path(__file__).resolve().parent)
    )
    return base_path / relative_path


# =============================================================================
# Platform and display setup
# =============================================================================



class DisplayScaling(TypedDict):
    system: str
    dpi: int
    scale_factor: float
    percent: int
    source: str


def enable_windows_dpi_awareness() -> None:
    if platform.system() != "Windows":
        return

    user32 = ctypes.windll.user32  # type: ignore[attr-defined]

    try:
        # Per-monitor DPI awareness v2
        set_dpi_context = user32.SetProcessDpiAwarenessContext
        set_dpi_context.argtypes = [ctypes.c_void_p]
        set_dpi_context.restype = ctypes.c_bool

        DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = ctypes.c_void_p(-4)

        set_dpi_context(
            DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2
        )
        return

    except (AttributeError, OSError):
        pass

    try:
        # Fallback for older Windows versions
        shcore = ctypes.windll.shcore #type: ignore[attr-defined]
        shcore.SetProcessDpiAwareness.argtypes = [ctypes.c_int]
        shcore.SetProcessDpiAwareness.restype = ctypes.c_long

        PROCESS_PER_MONITOR_DPI_AWARE = 2
        shcore.SetProcessDpiAwareness(
            PROCESS_PER_MONITOR_DPI_AWARE
        )

    except (AttributeError, OSError):
        # Final fallback
        user32.SetProcessDPIAware()


def get_display_scaling() -> DisplayScaling:
    system = platform.system()

    if system == "Windows":
        user32 = ctypes.windll.user32  # type: ignore[attr-defined]

        user32.GetDpiForSystem.argtypes = []
        user32.GetDpiForSystem.restype = ctypes.c_uint

        dpi = int(user32.GetDpiForSystem())
        source = "Windows DPI API"

    else:
        hidden_root = tk.Tk()
        hidden_root.withdraw()

        try:
            dpi = round(
                float(hidden_root.winfo_fpixels("1i"))
            )
            source = "Tk reported DPI"
        finally:
            hidden_root.destroy()

    scale_factor = dpi / 96.0

    return {
        "system": system,
        "dpi": dpi,
        "scale_factor": scale_factor,
        "percent": round(scale_factor * 100),
        "source": source,
    }


enable_windows_dpi_awareness()
DISPLAY_SCALING = get_display_scaling()


SYSTEM_SCALE_FACTOR = float(
    DISPLAY_SCALING["scale_factor"]
)

print(
    f"{DISPLAY_SCALING['system']}: "
    f"{DISPLAY_SCALING['percent']}% "
    f"({DISPLAY_SCALING['dpi']} DPI, "
    f"source: {DISPLAY_SCALING['source']})"
)





# =============================================================================
# Application style and sizing
# =============================================================================

PREFERRED_GUI_MONOSPACE_FONTS = [
    "Courier New",
    "DejaVu Sans Mono",
    "Nimbus Mono PS",
    "Liberation Mono",
    "Noto Sans Mono",
    "Ubuntu Mono",
    "Cousine",
    "Consolas",
    "Cascadia Mono",
    "Lucida Console",
]

BUTTON_FG_COLOR = ("#4800C7", "#4800C7")
BUTTON_HOVER_COLOR = ("#5B14D6", "#5B14D6")
BUTTON_BORDER_COLOR = ("#8D73DF", "#7652D6")
VIEWER_ACTIVE_BORDER_COLOR = "#F59E0B"
SETUP_CARD_COLOR = "#1A1A1A"
SETUP_BORDER_COLOR = "#566173"
SETUP_CONTROL_COLOR = "#2A2F3A"
SETUP_MUTED_TEXT_COLOR = "#9DA7B8"
SETUP_ACCENT_COLOR = "#7652D6"
SETUP_PANEL_MIN_WIDTH = 440
ANALYSIS_PANEL_MIN_WIDTH = 700
WORKSPACE_SPLITTER_WIDTH = 12
APPLICATION_TITLE = "meV-RIXS Toolkit"
UNTITLED_SESSION_NAME = "UNTITLED"

PRIMARY_GRADIENT_STYLE = {
    "cmap_name": "gnuplot",
    "cmap_start": 0.0,
    "cmap_end": 0.35,
    "hover_cmap_end": 0.43,
    "border_color": "#FFFFFF",
    "border_width": 0.75,
    "border_opacity": 0.75,
}

SECONDARY_GRADIENT_STYLE = {
    "cmap_name": "gist_gray",
    "cmap_start": 0.02,
    "cmap_end": 0.20,
    "hover_cmap_end": 0.27,
    "border_color": "#FFFFFF",
    "border_width": 0.75,
    "border_opacity": 0.4,
}

SUBTLE_GRADIENT_STYLE = {
    "cmap_name": "gist_gray",
    "cmap_start": 0.015,
    "cmap_end": 0.13,
    "hover_cmap_end": 0.17,
    "border_color": "#FFFFFF",
    "border_width": 0.75,
    "border_opacity": 0.3,
}

CALIBRATION_GRADIENT_STYLE = {
    "cmap_name": "gist_gray",
    "cmap_start": 0.02,
    "cmap_end": 0.20,
    "hover_cmap_end": 0.27,
    "border_color": "#FFFFFF",
    "border_width": 0.75,
    "border_opacity": 0.4,
}

DANGER_GRADIENT_STYLE = {
    "cmap_name": "magma",
    "cmap_start": 0.02,
    "cmap_end": 0.24,
    "hover_cmap_end": 0.34,
    "border_color": "#FFFFFF",
    "border_width": 0.75,
    "border_opacity": 0.9,
}

TAB_SELECTED_GRADIENT_STYLE = {
    "cmap_name": "magma",
    "cmap_start": 0.02,
    "cmap_end": 0.34,
    "hover_cmap_end": 0.44,
    "border_color": "#FFFFFF",
    "border_width": 0.75,
    "border_opacity": 0.4,
}

TAB_UNSELECTED_GRADIENT_STYLE = {
    "cmap_name": "gist_gray",
    "cmap_start": 0.015,
    "cmap_end": 0.10,
    "hover_cmap_end": 0.16,
    "border_color": "#FFFFFF",
    "border_width": 0.75,
    "border_opacity": 0.3,
}

# Keep the existing appearance as the design baseline.
FONT_SCALE_FACTOR = 1.2
FONT_SCALE_FACTOR_VIEWER = 1.3
FONT_SCALE_FACTOR_CALIBRATION = 1.3
USER_SCALE_FACTOR = 1.125 / SYSTEM_SCALE_FACTOR
TEXT_SIZE_MULTIPLIER = 1.0
_UI_FONT_BASE_SIZES: list[tuple[ReferenceType[ctk.CTkFont], int]] = []
UI = {
    "font_status": int(FONT_SCALE_FACTOR * 14),
    "font_small": int(FONT_SCALE_FACTOR * 14),
    "font_input": int(FONT_SCALE_FACTOR * 14),
    "font_dropdown_item": int(FONT_SCALE_FACTOR * 15),
    "font_tooltip": int(FONT_SCALE_FACTOR * 15),
    "font_normal": int(FONT_SCALE_FACTOR * 15),
    "font_section": int(FONT_SCALE_FACTOR * 19),
    "font_top_tabs": int(FONT_SCALE_FACTOR * 16),
    "font_title": int(FONT_SCALE_FACTOR * 22),
    "font_tab": int(FONT_SCALE_FACTOR * 18),
    "font_dataset_tab": int(FONT_SCALE_FACTOR * 15),
    "font_placeholder": int(FONT_SCALE_FACTOR * 19),
    # Embedded viewer typography. These base sizes are registered through
    # ui_font(), so the global text-size setting updates an open viewer live.
    "font_viewer_title": int(FONT_SCALE_FACTOR_VIEWER * 18),
    "font_viewer_subtitle": int(FONT_SCALE_FACTOR_VIEWER * 11),
    "font_viewer_card_title": int(FONT_SCALE_FACTOR_VIEWER * 15),
    "font_viewer_card_note": int(FONT_SCALE_FACTOR_VIEWER * 11),
    "font_viewer_subsection": int(FONT_SCALE_FACTOR_VIEWER * 13),
    "font_viewer_control_label": int(FONT_SCALE_FACTOR_VIEWER * 13),
    "font_viewer_value": int(FONT_SCALE_FACTOR_VIEWER * 12),
    "font_viewer_button": int(FONT_SCALE_FACTOR_VIEWER * 12),
    "font_viewer_toolbar": int(FONT_SCALE_FACTOR_VIEWER * 12),
    "font_viewer_status": int(FONT_SCALE_FACTOR_VIEWER * 11),
    "font_viewer_coordinates": int(FONT_SCALE_FACTOR_VIEWER * 12),
    "font_viewer_plot_label": int(FONT_SCALE_FACTOR_VIEWER * 14),
    "font_viewer_plot_tick": int(FONT_SCALE_FACTOR_VIEWER * 12),
    "font_viewer_step": int(FONT_SCALE_FACTOR_VIEWER * 15),
    "font_calibration_plot_tab": int(FONT_SCALE_FACTOR * 14),
    "font_calibration_toolbar": int(FONT_SCALE_FACTOR_CALIBRATION * 11),
    "font_calibration_plot_title": int(FONT_SCALE_FACTOR_CALIBRATION * 13),
    "font_calibration_plot_label": int(FONT_SCALE_FACTOR_CALIBRATION * 11),
    "font_calibration_plot_tick": int(FONT_SCALE_FACTOR_CALIBRATION * 10),
    "font_calibration_plot_legend": int(FONT_SCALE_FACTOR_CALIBRATION * 11),
    "user_scale": USER_SCALE_FACTOR,
    "control_height": 32,
    "compact_control_height": 26,
    "button_height": 36,
    "start_button_height": 42,
    "status_height": 140,
    "outer_padding": 16,
    "title_image_width": 350,
}

class GlobalViewerSettingSpec(TypedDict):
    group: str
    section: str
    key: str
    label: str
    kind: str
    default: object
    options: NotRequired[list[str]]


# Each entry maps one visible Viewer control to its session-state field. The
# table keeps the global settings dialog complete without duplicating its UI.
GLOBAL_VIEWER_SETTING_SPECS: list[GlobalViewerSettingSpec] = [
    {"group": "Display", "section": "settings", "key": "display_bin_x", "label": "2D bin size X", "kind": "int", "default": 1},
    {"group": "Display", "section": "settings", "key": "display_bin_y", "label": "2D bin size Y", "kind": "int", "default": 1},
    {"group": "Display", "section": "controls", "key": "display_vmin", "label": "Display minimum", "kind": "float", "default": 0.0},
    {"group": "Display", "section": "controls", "key": "display_vmax", "label": "Display maximum", "kind": "float", "default": 100.0},
    {"group": "Display", "section": "controls", "key": "colormap", "label": "Colormap", "kind": "choice", "default": "gnuplot", "options": ["gnuplot", "viridis", "inferno", "magma", "plasma", "gray"]},
    {"group": "Display", "section": "controls", "key": "colormap_start", "label": "Colormap start", "kind": "float", "default": 0.0},
    {"group": "Display", "section": "controls", "key": "colormap_end", "label": "Colormap end", "kind": "float", "default": 1.0},
    {"group": "Alignment & ROI", "section": "settings", "key": "tilt", "label": "Tilt", "kind": "float", "default": 0.0},
    {"group": "Alignment & ROI", "section": "controls", "key": "tilt_speedup", "label": "Tilt speedup", "kind": "int", "default": 4},
    {"group": "Alignment & ROI", "section": "controls", "key": "reference_line", "label": "Reference line height", "kind": "float", "default": 2048.0},
    {"group": "Alignment & ROI", "section": "mode", "key": "reference_line_visible", "label": "Show reference line", "kind": "bool", "default": True},
    {"group": "Alignment & ROI", "section": "settings", "key": "bottom_cut", "label": "Bottom cut", "kind": "int", "default": 0},
    {"group": "Alignment & ROI", "section": "settings", "key": "top_cut", "label": "Top cut", "kind": "int", "default": 0},
    {"group": "Alignment & ROI", "section": "mode", "key": "equal_cut_rows_enabled", "label": "Symmetric ROI", "kind": "bool", "default": True},
    {"group": "Alignment & ROI", "section": "controls", "key": "symmetric_margin", "label": "Symmetric cut margin", "kind": "float", "default": 2047.0},
    {"group": "Alignment & ROI", "section": "mode", "key": "zoom_to_cut_enabled", "label": "Zoom to cut", "kind": "bool", "default": False},
    {"group": "1D Spectrum", "section": "settings", "key": "spectrum_bin", "label": "1D bin size", "kind": "int", "default": 1},
    {"group": "1D Spectrum", "section": "controls", "key": "spectrum_axis_mode", "label": "Spectrum axis", "kind": "choice", "default": "pixel", "options": ["pixel", "energy", "loss"]},
    {"group": "Filters", "section": "settings", "key": "lower_percentile", "label": "Lower percentile", "kind": "float", "default": 0.0},
    {"group": "Filters", "section": "settings", "key": "upper_percentile", "label": "Upper percentile", "kind": "float", "default": 100.0},
    {"group": "Filters", "section": "settings", "key": "symmetric_fill_enabled", "label": "Symmetric fill", "kind": "bool", "default": False},
    {"group": "Filters", "section": "settings", "key": "median_filter_enabled", "label": "Median filter", "kind": "bool", "default": False},
    {"group": "Filters", "section": "settings", "key": "median_filter_window", "label": "Median window", "kind": "int", "default": 3},
    {"group": "Filters", "section": "settings", "key": "local_filter_enabled", "label": "Local filter", "kind": "bool", "default": False},
    {"group": "Filters", "section": "settings", "key": "local_filter_window", "label": "Local window", "kind": "int", "default": 15},
    {"group": "Filters", "section": "settings", "key": "local_filter_bottom_limit", "label": "Local lower percentile", "kind": "int", "default": 0},
    {"group": "Filters", "section": "settings", "key": "local_filter_upper_limit", "label": "Local upper percentile", "kind": "int", "default": 100},
    {"group": "Filters", "section": "mode", "key": "filter_statistics_enabled", "label": "Filter statistics", "kind": "bool", "default": False},
    {"group": "Output", "section": "controls", "key": "include_metadata", "label": "Include metadata (.h5)", "kind": "bool", "default": False},
    {"group": "Output", "section": "controls", "key": "choose_save_path", "label": "Choose file name and folder", "kind": "bool", "default": False},
]

EXPCHAMBER_EMPTY_TEXT = (
    "Expchamber:   x = —    y = —    z = —    r = — (offset = —)"
)


# =============================================================================
# Shared fonts and widget factories
# =============================================================================


def scaled_plot_font_size(
    base_size: int,
    zoom_percent: int,
    minimum: int = 1,
) -> int:
    scale_correction = 1.0 / max(
        SYSTEM_SCALE_FACTOR,
        1e-6,
    )

    return max(
        minimum,
        round(
            base_size
            * TEXT_SIZE_MULTIPLIER
            * zoom_percent / 100
            * scale_correction
        ),
    )


@dataclass(frozen=True)
class CalibrationOptions:
    fit_poly_order: int
    double_gaussian_model: bool
    params_poly_orders: list[int] | None
    peak_buffer: int

    @property
    def model_name(self) -> str:
        return "Double Gaussian" if self.double_gaussian_model else "Single Gaussian"


@dataclass(frozen=True)
class ViewerOptions:
    scans: np.ndarray
    calibration_scans: np.ndarray | None
    search_roots: list[Path]
    spectra_dir: Path | None
    histogram_dir: Path | None
    calibration: CalibrationOptions
    dataset_name: str


def get_cross_platform_gui_font_family(root) -> str:
    available_fonts = {
        family.casefold(): family
        for family in tkfont.families(root)
    }
    for preferred_font in PREFERRED_GUI_MONOSPACE_FONTS:
        matched_font = available_fonts.get(preferred_font.casefold())
        if matched_font is not None:
            return matched_font
    return tkfont.nametofont("TkDefaultFont", root=root).cget("family")


def configure_native_tk_fonts(root, family: str) -> None:
    for font_name in (
        "TkDefaultFont",
        "TkTextFont",
        "TkMenuFont",
        "TkHeadingFont",
        "TkCaptionFont",
        "TkSmallCaptionFont",
        "TkIconFont",
        "TkTooltipFont",
    ):
        try:
            tkfont.nametofont(font_name, root=root).configure(family=family)
        except Exception:
            pass


def configure_platform_gui_scaling(user_scale: float = 1.0) -> None:
    if user_scale <= 0:
        raise ValueError("user_scale must be larger than zero.")
    ctk.set_widget_scaling(user_scale)
    ctk.set_window_scaling(user_scale)


def set_text_size_multiplier(multiplier: float) -> None:
    if multiplier <= 0:
        raise ValueError("Text size multiplier must be larger than zero.")

    global TEXT_SIZE_MULTIPLIER
    TEXT_SIZE_MULTIPLIER = multiplier
    live_fonts: list[tuple[ReferenceType[ctk.CTkFont], int]] = []
    for font_reference, base_size in _UI_FONT_BASE_SIZES:
        font = font_reference()
        if font is None:
            continue
        font.configure(size=max(1, round(base_size * multiplier)))
        live_fonts.append((font_reference, base_size))
    _UI_FONT_BASE_SIZES[:] = live_fonts


def ui_font(
    size: int,
    *,
    family: str | None = None,
    bold: bool = False,
) -> ctk.CTkFont:
    scaled_size = max(1, round(size * TEXT_SIZE_MULTIPLIER))
    if family is None:
        if bold:
            font = ctk.CTkFont(
                size=scaled_size,
                weight="bold",
            )
        else:
            font = ctk.CTkFont(size=scaled_size)
    elif bold:
        font = ctk.CTkFont(
            family=family,
            size=scaled_size,
            weight="bold",
        )
    else:
        font = ctk.CTkFont(family=family, size=scaled_size)

    _UI_FONT_BASE_SIZES.append((ref(font), size))
    return font


def make_label(
    master,
    text: str,
    *,
    size: int | None = None,
    family: str | None = None,
    bold: bool = False,
    **kwargs,
) -> ctk.CTkLabel:
    return ctk.CTkLabel(
        master,
        text=text,
        font=ui_font(size or UI["font_normal"], family=family, bold=bold),
        **kwargs,
    )


def bind_responsive_label_wrap(
    label: ctk.CTkLabel,
    container,
    *,
    horizontal_padding: int = 0,
    minimum: int = 40,
    maximum: int | None = None,
) -> ctk.CTkLabel:
    base_height = int(float(label.cget("height") or 0))
    pending_job: str | None = None
    updating = False

    # Tk reports physical pixels while CustomTkinter expects logical pixels.
    def update_wrap() -> None:
        nonlocal pending_job, updating
        pending_job = None
        if updating:
            return
        updating = True
        try:
            if not label.winfo_exists():
                return
            label_width = int(label.winfo_width())
            uses_label_width = label_width > 1
            width = label_width if uses_label_width else int(container.winfo_width())
            reverse_scaling = cast(
                Callable[[float], float] | None,
                getattr(label, "_reverse_widget_scaling", None),
            )
            if callable(reverse_scaling):
                width = int(reverse_scaling(width))
            padding = 2 if uses_label_width else horizontal_padding
            wraplength = max(minimum, width - padding)
            if maximum is not None:
                wraplength = min(maximum, wraplength)
            if int(float(label.cget("wraplength") or 0)) != wraplength:
                label.configure(wraplength=wraplength)
            text_widget = getattr(label, "_text_label", None)
            if text_widget is not None and text_widget.winfo_exists():
                required_height = int(text_widget.winfo_reqheight())
                if callable(reverse_scaling):
                    required_height = int(reverse_scaling(required_height))
                target_height = max(base_height, required_height + 6)
                if int(float(label.cget("height") or 0)) != target_height:
                    label.configure(height=target_height)
        except tk.TclError:
            return
        finally:
            updating = False

    def schedule_update(_event=None) -> None:
        nonlocal pending_job
        if updating or pending_job is not None:
            return
        try:
            # One pending callback is enough, even during a Configure storm.
            pending_job = label.after(16, update_wrap)
        except tk.TclError:
            pending_job = None

    container.bind("<Configure>", schedule_update, add="+")
    label.bind("<Configure>", schedule_update, add="+")
    setattr(label, "_responsive_wrap_update", schedule_update)
    schedule_update()
    return label


# =============================================================================
# Custom widgets
# =============================================================================


class GradientButton(ctk.CTkFrame):

    def __init__(
        self,
        master,
        *,
        text: str,
        command: Callable[[], None],
        width: int = 140,
        height: int = UI["button_height"],
        font: ctk.CTkFont | None = None,
        text_color="white",
        disabled_text_color="#A0A0A0",
        border_color=BUTTON_BORDER_COLOR,
        border_width: float = 0.75,
        border_opacity: float = 0.55,
        corner_radius: int = 10,
        anchor: Literal["center", "w", "e"] = "center",
        text_padding: int = 12,
        leading_text: str | None = None,
        leading_command: Callable[[], None] | None = None,
        leading_width: int = 28,
        leading_visible: bool = True,
        leading_font: ctk.CTkFont | None = None,
        leading_tooltip_text: str | None = None,
        secondary_text: str | None = None,
        secondary_command: Callable[[], None] | None = None,
        secondary_width: int = 26,
        secondary_visible: bool = True,
        secondary_font: ctk.CTkFont | None = None,
        secondary_tooltip_text: str | None = None,
        tooltip_text: str | None = None,
        cmap_name: str = "gnuplot",
        cmap_start: float = 0.0,
        cmap_end: float = 0.35,
        hover_cmap_end: float = 0.43,
        reverse_gradient: bool = False,
        **kwargs,
    ):
        super().__init__(
            master,
            width=width,
            height=height,
            fg_color="transparent",
            corner_radius=0,
            **kwargs,
        )

        self._text = text
        self._command = command
        self._button_width = width
        self._button_height = height
        self._font = font or ui_font(UI["font_small"])
        self._text_color = text_color
        self._disabled_text_color = disabled_text_color
        self._border_color = border_color
        self._border_width = float(border_width)
        self._border_opacity = float(
            np.clip(border_opacity, 0.0, 1.0)
        )
        self._corner_radius = corner_radius

        if anchor not in ("center", "w", "e"):
            raise ValueError(
                "GradientButton anchor must be "
                "'center', 'w', or 'e'."
            )

        self._text_anchor = anchor
        self._text_padding = int(text_padding)
        self._leading_text = leading_text
        self._leading_command = leading_command
        self._leading_width = int(leading_width)
        self._leading_visible = bool(leading_visible)
        self._leading_font = leading_font or self._font
        self._leading_tooltip_text = leading_tooltip_text
        self._leading_hovered = False
        self._secondary_text = secondary_text
        self._secondary_command = secondary_command
        self._secondary_width = int(secondary_width)
        self._secondary_visible = bool(secondary_visible)
        self._secondary_font = secondary_font or self._font
        self._secondary_tooltip_text = secondary_tooltip_text
        self._secondary_hovered = False
        self._tooltip_text = tooltip_text
        self._tooltip_after_id = None
        self._tooltip_window = None
        self._tooltip_font: ctk.CTkFont | None = None
        # Used only by reorderable dataset-tab buttons.
        self._tab_drag_name = ""

        if cmap_name not in matplotlib.colormaps:
            raise ValueError(
                f"Unknown button colormap: {cmap_name!r}"
            )

        self._cmap_name = cmap_name

        self._cmap_start = float(
            np.clip(cmap_start, 0.0, 1.0)
        )
        self._cmap_end = float(
            np.clip(cmap_end, 0.0, 1.0)
        )
        self._hover_cmap_end = float(
            np.clip(hover_cmap_end, 0.0, 1.0)
        )
        self._reverse_gradient = bool(
            reverse_gradient
        )

        self._state = "normal"
        self._hovered = False
        self._pressed = False
        self._photo_image = None
        self._redraw_after_id = None

        self.pack_propagate(False)
        self.grid_propagate(False)

        self._canvas = tk.Canvas(
            self,
            width=width,
            height=height,
            bd=0,
            highlightthickness=0,
            relief="flat",
            takefocus=True,
        )
        self._canvas.pack(fill="both", expand=True)

        self._update_canvas_background()

        self._canvas.bind(
            "<Configure>",
            self._schedule_redraw,
        )
        self._canvas.bind(
            "<Enter>",
            self._on_enter,
        )
        self._canvas.bind(
            "<Leave>",
            self._on_leave,
        )
        self._canvas.bind(
            "<Motion>",
            self._on_motion,
        )
        self._canvas.bind(
            "<ButtonPress-1>",
            self._on_press,
        )
        self._canvas.bind(
            "<ButtonRelease-1>",
            self._on_release,
        )
        self._canvas.bind(
            "<Return>",
            self._on_keyboard_activate,
        )
        self._canvas.bind(
            "<space>",
            self._on_keyboard_activate,
        )

        self._observed_fonts: list[ctk.CTkFont] = []
        for observed_font in (
            self._font,
            self._leading_font,
            self._secondary_font,
        ):
            self._observe_font(observed_font)

        self.after_idle(self._redraw_gradient)

    # Color resolution and image rendering

    def _resolve_color(self, color) -> str:
        """Resolve a CustomTkinter light/dark color."""

        return self._apply_appearance_mode(color)

    def _update_canvas_background(self) -> None:
        background = self._apply_appearance_mode(
            self._bg_color
        )
        self._canvas.configure(bg=background)

    def _set_appearance_mode(
        self,
        mode_string: str,
    ) -> None:
        super()._set_appearance_mode(mode_string)

        if hasattr(self, "_canvas"):
            self._update_canvas_background()
            self._schedule_redraw()

    def _schedule_redraw(self, _event=None) -> None:
        if self._redraw_after_id is not None:
            self.after_cancel(self._redraw_after_id)

        self._redraw_after_id = self.after_idle(
            self._redraw_gradient
        )

    def _observe_font(self, font: ctk.CTkFont) -> None:
        if any(observed_font is font for observed_font in self._observed_fonts):
            return
        font.add_size_configure_callback(self._schedule_redraw)
        self._observed_fonts.append(font)

    def destroy(self) -> None:
        for font in self._observed_fonts:
            font.remove_size_configure_callback(self._schedule_redraw)
        self._observed_fonts.clear()
        self._hide_tooltip()
        super().destroy()

    def _create_gradient_image(
        self,
        width: int,
        height: int,
    ) -> Image.Image:
        """
        Create an antialiased rounded gradient image.

        Rendering takes place at twice the displayed resolution.
        """

        scale = 4
        render_width = max(2, width * scale)
        render_height = max(2, height * scale)

        x = np.linspace(-1, 1, render_width)
        y = np.linspace(-1, 1, render_height)
        xx, yy = np.meshgrid(x, y)

        radius = np.sqrt(
            xx**2 + (yy / 2.5) ** 2
        )

        falloff = 1 - np.clip(
            radius,
            0,
            1,
        )

        if self._reverse_gradient:
            falloff = 1 - falloff

        current_cmap_end = (
            self._hover_cmap_end
            if self._hovered
            else self._cmap_end
        )

        gradient = (
            self._cmap_start
            + (
                current_cmap_end
                - self._cmap_start
            )
            * falloff
        )

        rgba = (
            matplotlib.colormaps[
                self._cmap_name
            ](gradient)
            * 255
        ).astype(np.uint8)

        image = Image.fromarray(
            rgba,
            mode="RGBA",
        )

        radius_pixels = min(
            self._corner_radius * scale,
            render_height // 2,
        )

        mask = Image.new(
            "L",
            (render_width, render_height),
            0,
        )
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rounded_rectangle(
            (
                0,
                0,
                render_width - 1,
                render_height - 1,
            ),
            radius=radius_pixels,
            fill=255,
        )
        image.putalpha(mask)

        if self._state == "disabled":
            image = ImageEnhance.Color(
                image
            ).enhance(0.15)
            image = ImageEnhance.Brightness(
                image
            ).enhance(0.55)

        if self._border_width > 0:
            border_color = self._resolve_color(
                self._border_color
            )
            border_rgb = ImageColor.getrgb(
                border_color
            )

            border_pixels = max(
                1,
                round(self._border_width * scale),
            )

            # The existing mask describes the outer rounded shape.
            outer_mask = mask.copy()

            inner_mask = Image.new(
                "L",
                (render_width, render_height),
                0,
            )

            inner_x0 = border_pixels
            inner_y0 = border_pixels
            inner_x1 = (
                render_width - 1 - border_pixels
            )
            inner_y1 = (
                render_height - 1 - border_pixels
            )

            if (
                inner_x1 >= inner_x0
                and inner_y1 >= inner_y0
            ):
                inner_radius = max(
                    0,
                    radius_pixels - border_pixels,
                )

                inner_draw = ImageDraw.Draw(
                    inner_mask
                )
                inner_draw.rounded_rectangle(
                    (
                        inner_x0,
                        inner_y0,
                        inner_x1,
                        inner_y1,
                    ),
                    radius=inner_radius,
                    fill=255,
                )

                # Subtract the inner shape from the outer shape.
                # What remains is a smooth rounded border ring.
                border_mask = ImageChops.subtract(
                    outer_mask,
                    inner_mask,
                )
            else:
                border_mask = outer_mask

            opacity = self._border_opacity

            if opacity < 1.0:
                opacity_lut = [
                    int(round(value * opacity))
                    for value in range(256)
                ]
                border_mask = border_mask.point(
                    opacity_lut
                )

            border_layer = Image.new(
                "RGBA",
                (render_width, render_height),
                (
                    border_rgb[0],
                    border_rgb[1],
                    border_rgb[2],
                    0,
                ),
            )
            border_layer.putalpha(border_mask)

            image = Image.alpha_composite(
                image,
                border_layer,
            )

        image = image.resize(
            (width, height),
            Image.Resampling.LANCZOS,
        )

        return image

    def _redraw_gradient(self) -> None:
        self._redraw_after_id = None

        if not self.winfo_exists():
            return

        width = max(
            2,
            self._canvas.winfo_width(),
        )
        height = max(
            2,
            self._canvas.winfo_height(),
        )

        image = self._create_gradient_image(
            width,
            height,
        )
        self._photo_image = ImageTk.PhotoImage(
            image
        )

        self._canvas.delete("all")
        self._canvas.create_image(
            0,
            0,
            anchor="nw",
            image=self._photo_image,
        )

        text_color = (
            self._disabled_text_color
            if self._state == "disabled"
            else self._text_color
        )

        leading_text = self._leading_text
        leading_command = self._leading_command
        secondary_text = self._secondary_text
        secondary_command = self._secondary_command
        has_leading_space = leading_text is not None and leading_command is not None
        has_leading_action = (
            has_leading_space and self._leading_visible
        )
        has_secondary_space = (
            secondary_text is not None
            and secondary_command is not None
        )
        has_secondary_action = (
            has_secondary_space and self._secondary_visible
        )
        text_area_left = self._leading_width if has_leading_space else 0
        text_area_right = (
            width - self._secondary_width if has_secondary_space else width
        )

        if self._text_anchor == "w":
            text_x = text_area_left + self._text_padding
            canvas_anchor = "w"
        elif self._text_anchor == "e":
            text_x = text_area_right - self._text_padding
            canvas_anchor = "e"
        else:
            text_x = (text_area_left + text_area_right) // 2
            canvas_anchor = "center"

        self._canvas.create_text(
            text_x,
            height // 2,
            text=self._text,
            fill=self._resolve_color(text_color),
            font=self._apply_font_scaling(self._font),
            anchor=canvas_anchor,
        )

        if has_leading_action and leading_text is not None:
            leading_x = self._leading_width // 2
            self._canvas.create_text(
                leading_x,
                height // 2,
                text=leading_text,
                fill=self._resolve_color(text_color),
                font=self._apply_font_scaling(self._leading_font),
                anchor="center",
            )

        if has_secondary_action and secondary_text is not None:
            secondary_x = width - self._secondary_width // 2
            self._canvas.create_text(
                secondary_x,
                height // 2,
                text=secondary_text,
                fill=self._resolve_color(text_color),
                font=self._apply_font_scaling(self._secondary_font),
                anchor="center",
            )

        self._canvas.configure(
            cursor=(
                "arrow"
                if self._state == "disabled"
                else "hand2"
            )
        )

    # Tooltip and input handling

    def _current_tooltip_text(self) -> str | None:
        if self._leading_hovered:
            return self._leading_tooltip_text
        if self._secondary_hovered:
            return self._secondary_tooltip_text
        return self._tooltip_text

    def _schedule_tooltip(self) -> None:
        if self._current_tooltip_text() is None or self._tooltip_after_id is not None:
            return
        self._tooltip_after_id = self.after(450, self._show_tooltip)

    def _show_tooltip(self) -> None:
        self._tooltip_after_id = None
        tooltip_text = self._current_tooltip_text()
        if not self._hovered or tooltip_text is None:
            return

        self._hide_tooltip()
        tooltip = tk.Toplevel(self)
        tooltip.overrideredirect(True)
        try:
            tooltip.attributes("-topmost", True)
        except tk.TclError:
            pass

        if self._tooltip_font is None:
            tooltip_family = str(self._font.cget("family"))
            self._tooltip_font = ui_font(
                UI["font_tooltip"],
                family=tooltip_family,
            )
        tooltip_font = self._tooltip_font

        label = tk.Label(
            tooltip,
            text=tooltip_text,
            bg="#25252B",
            fg="#F2F2F4",
            relief="solid",
            borderwidth=1,
            padx=10,
            pady=6,
            font=self._apply_font_scaling(tooltip_font),
        )
        label.pack()
        tooltip.update_idletasks()

        if self._leading_hovered:
            anchor_x = self._leading_width // 2
        elif self._secondary_hovered:
            anchor_x = self.winfo_width() - self._secondary_width // 2
        else:
            anchor_x = self.winfo_width() // 2
        x = self.winfo_rootx() + max(0, anchor_x - tooltip.winfo_reqwidth() // 2)
        y = self.winfo_rooty() + self.winfo_height() + 4
        tooltip.geometry(f"+{x}+{y}")
        self._tooltip_window = tooltip

    def _hide_tooltip(self) -> None:
        if self._tooltip_after_id is not None:
            try:
                self.after_cancel(self._tooltip_after_id)
            except tk.TclError:
                pass
            self._tooltip_after_id = None
        if self._tooltip_window is not None:
            try:
                self._tooltip_window.destroy()
            except tk.TclError:
                pass
            self._tooltip_window = None

    def _on_enter(self, event=None) -> None:
        if self._state == "disabled":
            return

        self._hovered = True
        if event is not None:
            self._update_action_hover(event.x)
        self._schedule_tooltip()
        self._schedule_redraw()

    def _on_leave(self, _event=None) -> None:
        self._hovered = False
        self._leading_hovered = False
        self._secondary_hovered = False
        self._pressed = False
        self._hide_tooltip()
        self._schedule_redraw()

    def _update_action_hover(self, pointer_x: int) -> None:
        leading_hovered = (
            self._leading_visible
            and self._leading_command is not None
            and pointer_x < self._leading_width
        )
        secondary_hovered = (
            self._secondary_visible
            and self._secondary_command is not None
            and pointer_x >= self._canvas.winfo_width() - self._secondary_width
        )
        if (
            leading_hovered != self._leading_hovered
            or secondary_hovered != self._secondary_hovered
        ):
            self._leading_hovered = leading_hovered
            self._secondary_hovered = secondary_hovered
            self._hide_tooltip()
            self._schedule_tooltip()

    def _on_motion(self, event) -> None:
        self._update_action_hover(event.x)

    def _on_press(self, _event=None) -> None:
        if self._state == "disabled":
            return

        self._pressed = True
        self._canvas.focus_set()

    def _on_release(self, event=None) -> None:
        if (
            self._state == "disabled"
            or not self._pressed
        ):
            return

        self._pressed = False

        if event is not None:
            width = self._canvas.winfo_width()
            height = self._canvas.winfo_height()

            pointer_is_inside = (
                0 <= event.x < width
                and 0 <= event.y < height
            )

            if not pointer_is_inside:
                return

        leading_command = self._leading_command
        secondary_command = self._secondary_command
        leading_clicked = (
            event is not None
            and self._leading_visible
            and leading_command is not None
            and event.x < self._leading_width
        )
        secondary_clicked = (
            event is not None
            and self._secondary_visible
            and secondary_command is not None
            and event.x >= self._canvas.winfo_width() - self._secondary_width
        )
        self._hide_tooltip()
        if leading_clicked and leading_command is not None:
            leading_command()
        elif secondary_clicked and secondary_command is not None:
            secondary_command()
        elif self._command is not None:
            self._command()

    def _on_keyboard_activate(
        self,
        _event=None,
    ):
        if (
            self._state != "disabled"
            and self._command is not None
        ):
            self._command()

        return "break"

    # Public configuration API used throughout the launcher

    def configure_button(
        self,
        **kwargs,
    ) -> None:
        """Change GradientButton-specific properties."""

        redraw_required = False

        if "text" in kwargs:
            self._text = str(
                kwargs.pop("text")
            )
            redraw_required = True

        if "command" in kwargs:
            self._command = kwargs.pop(
                "command"
            )

        if "leading_command" in kwargs:
            self._leading_command = kwargs.pop("leading_command")
            redraw_required = True

        if "leading_visible" in kwargs:
            self._leading_visible = bool(kwargs.pop("leading_visible"))
            if not self._leading_visible:
                self._leading_hovered = False
                self._hide_tooltip()
            redraw_required = True

        if "leading_tooltip_text" in kwargs:
            self._hide_tooltip()
            self._leading_tooltip_text = kwargs.pop("leading_tooltip_text")

        if "secondary_command" in kwargs:
            self._secondary_command = kwargs.pop("secondary_command")
            redraw_required = True

        if "secondary_visible" in kwargs:
            self._secondary_visible = bool(kwargs.pop("secondary_visible"))
            if not self._secondary_visible:
                self._secondary_hovered = False
                self._hide_tooltip()
            redraw_required = True

        if "secondary_tooltip_text" in kwargs:
            self._hide_tooltip()
            self._secondary_tooltip_text = kwargs.pop("secondary_tooltip_text")

        if "tooltip_text" in kwargs:
            self._hide_tooltip()
            self._tooltip_text = kwargs.pop("tooltip_text")

        if "state" in kwargs:
            state = kwargs.pop("state")

            if state not in ("normal", "disabled"):
                raise ValueError(
                    "GradientButton state must be "
                    "'normal' or 'disabled'."
                )

            self._state = state
            redraw_required = True

        if "text_color" in kwargs:
            self._text_color = kwargs.pop(
                "text_color"
            )
            redraw_required = True

        if "disabled_text_color" in kwargs:
            self._disabled_text_color = kwargs.pop(
                "disabled_text_color"
            )
            redraw_required = True

        if "border_color" in kwargs:
            self._border_color = kwargs.pop(
                "border_color"
            )
            redraw_required = True

        if "border_width" in kwargs:
            self._border_width = float(
                kwargs.pop("border_width")
            )
            redraw_required = True

        if "border_opacity" in kwargs:
            self._border_opacity = float(
                np.clip(
                    kwargs.pop("border_opacity"),
                    0.0,
                    1.0,
                )
            )
            redraw_required = True

        if "corner_radius" in kwargs:
            self._corner_radius = int(
                kwargs.pop("corner_radius")
            )
            redraw_required = True

        if "cmap_name" in kwargs:
            cmap_name = str(
                kwargs.pop("cmap_name")
            )

            if cmap_name not in matplotlib.colormaps:
                raise ValueError(
                    "Unknown button colormap: "
                    f"{cmap_name!r}"
                )

            self._cmap_name = cmap_name
            redraw_required = True

        if "cmap_start" in kwargs:
            self._cmap_start = float(
                np.clip(
                    kwargs.pop("cmap_start"),
                    0.0,
                    1.0,
                )
            )
            redraw_required = True

        if "cmap_end" in kwargs:
            self._cmap_end = float(
                np.clip(
                    kwargs.pop("cmap_end"),
                    0.0,
                    1.0,
                )
            )
            redraw_required = True

        if "hover_cmap_end" in kwargs:
            self._hover_cmap_end = float(
                np.clip(
                    kwargs.pop("hover_cmap_end"),
                    0.0,
                    1.0,
                )
            )
            redraw_required = True

        if "reverse_gradient" in kwargs:
            self._reverse_gradient = bool(
                kwargs.pop("reverse_gradient")
            )
            redraw_required = True

        if "font" in kwargs:
            self._font = kwargs.pop("font")
            self._observe_font(self._font)
            redraw_required = True

        if "width" in kwargs:
            width = int(kwargs.pop("width"))
            self._button_width = width

            super().configure(width=width)
            self._canvas.configure(width=width)
            redraw_required = True

        if "height" in kwargs:
            height = int(kwargs.pop("height"))
            self._button_height = height

            super().configure(height=height)
            self._canvas.configure(height=height)
            redraw_required = True

        if "anchor" in kwargs:
            anchor = kwargs.pop("anchor")

            if anchor not in ("center", "w", "e"):
                raise ValueError(
                    "GradientButton anchor must be "
                    "'center', 'w', or 'e'."
                )

            self._text_anchor = anchor
            redraw_required = True

        if "text_padding" in kwargs:
            self._text_padding = int(
                kwargs.pop("text_padding")
            )
            redraw_required = True

        if kwargs:
            unsupported = ", ".join(
                sorted(kwargs)
            )
            raise TypeError(
                "Unsupported GradientButton options: "
                f"{unsupported}"
            )

        if redraw_required:
            self._schedule_redraw()

def make_gradient_button(
    master,
    *,
    text: str,
    command: Callable[[], None],
    width: int | None = None,
    height: int | None = None,
    font_size: int | None = None,
    family: str | None = None,
    bold: bool = False,
    **kwargs,
) -> GradientButton:
    return GradientButton(
        master,
        text=text,
        command=command,
        width=width or 140,
        height=height or UI["button_height"],
        font=ui_font(
            font_size or UI["font_normal"],
            family=family,
            bold=bold,
        ),
        **kwargs,
    )

def make_combobox(
    master,
    *,
    variable: ctk.StringVar,
    values: list[str],
    family: str | None = None,
    state: str = "readonly",
    font_size: int | None = None,
    dropdown_font_size: int | None = None,
    **kwargs,
) -> ctk.CTkComboBox:
    options = {
        "variable": variable,
        "values": values,
        "state": state,
        "height": UI["control_height"],
        "font": ui_font(font_size or UI["font_normal"], family=family),
        "dropdown_font": ui_font(
            dropdown_font_size or UI["font_dropdown_item"],
            family=family),
        "fg_color": ("#E8E8E8", "#171717"),
        "border_color": ("#B8B8B8", "#444444"),
        "border_width": 1,
        "button_color": ("#D0D0D0", "#252525"),
        "button_hover_color": ("#C0C0C0", "#333333"),
        "text_color": ("#181818", "#F0F0F0"),
        "dropdown_fg_color": ("#F0F0F0", "#171717"),
        "dropdown_hover_color": ("#D8D8D8", "#292929"),
        "dropdown_text_color": ("#181818", "#F0F0F0"),
        "corner_radius": 8,
    }
    options.update(kwargs)
    return ctk.CTkComboBox(master, **options)


def make_entry(
    master,
    *,
    family: str | None = None,
    font_size: int | None = None,
    variable: ctk.StringVar | None = None,
    placeholder: str = "",
    **kwargs,
) -> ctk.CTkEntry:
    options = {
        "height": UI["control_height"],
        "font": ui_font(font_size or UI["font_input"], family=family),
    }
    if variable is not None:
        options["textvariable"] = variable
    if placeholder:
        options["placeholder_text"] = placeholder
        options["placeholder_text_color"] = "gray55"
    options.update(kwargs)
    return ctk.CTkEntry(master, **options)


# =============================================================================
# Input parsing and calibration result validation
# =============================================================================


def validate_directory(path: Path | None, label: str) -> None:
    if path is not None and (not path.exists() or not path.is_dir()):
        raise ValueError(f"{label} does not exist or is not a folder:\n{path}")


def get_scan_search_roots(
    text: str,
    *,
    max_parent_levels: int = 3,
) -> list[Path]:
    explicit_root = parse_optional_path(text)
    if explicit_root is not None:
        return [explicit_root]
    cwd = Path.cwd()
    return [cwd, *list(cwd.parents)[:max_parent_levels]]


def parse_scans(text: str) -> np.ndarray:
    text = text.strip()
    if not text:
        raise ValueError("Please enter at least one scan number.")

    normalized = (
        text.replace(";", ",")
        .replace("\n", ",")
        .replace("\t", ",")
        .replace(" ", ",")
    )
    tokens = [token.strip() for token in normalized.split(",") if token.strip()]
    scans: list[int] = []

    for token in tokens:
        if ":" in token:
            left, right = token.split(":", 1)
            start, stop = int(left), int(right)
            step = 1 if stop >= start else -1
            scans.extend(range(start, stop + step, step))
        elif "-" in token and not token.startswith("-"):
            left, right = token.split("-", 1)
            start, stop = int(left), int(right)
            step = 1 if stop >= start else -1
            scans.extend(range(start, stop + step, step))
        else:
            scans.append(int(token))

    if not scans:
        raise ValueError("No valid scan numbers found.")
    return np.asarray(scans, dtype=int)


def parse_optional_path(text: str) -> Path | None:
    clean_text = text.strip().strip('"')
    return Path(clean_text).expanduser() if clean_text else None


def _unpack_calibration_output(output: object) -> tuple[dict, dict]:
    # Calibration has several return modes. The launcher always requests a
    # pair containing serializable details and Matplotlib figures.
    if not isinstance(output, tuple) or len(output) != 2:
        raise TypeError(
            "The calibration helper did not return details and figures."
        )

    details, figures = output
    if not isinstance(details, dict):
        raise TypeError(
            "The calibration helper did not return calibration details."
        )
    if not isinstance(figures, dict):
        raise TypeError(
            "The calibration helper did not return calibration figures."
        )
    return details, figures


def ask_dataset_name(
    parent,
    *,
    title: str = "Rename Dataset",
    prompt: str = "New Dataset name:",
    initialvalue: str = "",
    existing_names: set[str] | None = None,
    old_name: str | None = None,
) -> str | None:
    existing_names = set(existing_names or ())

    dialog = ctk.CTkToplevel(parent)
    dialog.withdraw()
    dialog.title(title)
    dialog.transient(parent)
    dialog.grab_set()
    dialog.resizable(False, False)

    dialog.grid_columnconfigure(0, weight=1)

    result: dict[str, str | None] = {"value": None}
    name_var = ctk.StringVar(value=initialvalue)

    make_label(
        dialog,
        title,
        size=UI["font_title"],
        bold=True,
        anchor="w",
    ).grid(
        row=0,
        column=0,
        sticky="ew",
        padx=24,
        pady=(18, 4),
    )

    make_label(
        dialog,
        prompt,
        anchor="w",
    ).grid(
        row=1,
        column=0,
        sticky="ew",
        padx=24,
        pady=(0, 8),
    )

    entry = make_entry(
        dialog,
        variable=name_var,
        font=ui_font(UI["font_section"]),
        height=UI["start_button_height"],
        width=440,
    )
    entry.grid(
        row=2,
        column=0,
        sticky="ew",
        padx=24,
        pady=(0, 6),
    )

    error_label = make_label(
        dialog,
        "",
        size=UI["font_small"],
        text_color="#ff6b6b",
        anchor="w",
    )
    error_label.grid(
        row=3,
        column=0,
        sticky="ew",
        padx=24,
        pady=(0, 8),
    )

    button_frame = ctk.CTkFrame(
        dialog,
        fg_color="transparent",
    )
    button_frame.grid(
        row=4,
        column=0,
        pady=(2, 18),
    )

    def close(value: str | None) -> None:
        result["value"] = value
        dialog.destroy()

    def confirm() -> None:
        new_name = name_var.get().strip()

        if not new_name:
            error_label.configure(
                text="Dataset name cannot be empty."
            )
            entry.focus_set()
            return

        if old_name is not None and new_name == old_name:
            close(None)
            return

        if new_name in existing_names:
            error_label.configure(
                text=f"A dataset named '{new_name}' already exists."
            )
            entry.focus_set()
            return

        close(new_name)

    make_gradient_button(
        button_frame,
        text="Cancel",
        width=120,
        command=lambda: close(None),
        **SUBTLE_GRADIENT_STYLE,
    ).grid(
        row=0,
        column=0,
        padx=(0, 6),
    )

    make_gradient_button(
        button_frame,
        text="Apply",
        width=120,
        command=confirm,
        bold=True,
        **SECONDARY_GRADIENT_STYLE,
    ).grid(
        row=0,
        column=1,
        padx=(6, 0),
    )

    dialog.bind(
        "<Return>",
        lambda _event: confirm(),
    )
    dialog.bind(
        "<Escape>",
        lambda _event: close(None),
    )

    # Calculate the required window size after all widgets exist.
    dialog.update_idletasks()

    width = dialog.winfo_reqwidth()
    height = dialog.winfo_reqheight()

    x = (
        parent.winfo_rootx()
        + (parent.winfo_width() - width) // 2
    )
    y = (
        parent.winfo_rooty()
        + (parent.winfo_height() - height) // 2
    )

    dialog.geometry(
        f"{width}x{height}+{x}+{y}"
    )
    dialog.deiconify()

    entry.focus_set()
    entry.select_range(0, "end")

    parent.wait_window(dialog)

    return result["value"]


# =============================================================================
# Reusable dropdown and tab controls
# =============================================================================


class ScrollableDropdown(ctk.CTkFrame):
    """Scrollable dropdown with keyboard navigation and automatic closing."""

    def __init__(
        self,
        master,
        *,
        variable: ctk.StringVar,
        values: list[str],
        max_visible_items: int = 10,
        width: int = 220,
        height: int = UI["control_height"],
        font_size: int = UI["font_normal"],
        dropdown_font_size: int = UI["font_dropdown_item"],
        command=None,
    ):
        super().__init__(master, fg_color="transparent")
        self.variable = variable
        self.values = list(values)
        self.max_visible_items = max_visible_items
        self.command = command
        self.dropdown_font_size = dropdown_font_size
        self.popup: ctk.CTkToplevel | None = None
        self._listbox: tk.Listbox | None = None
        self._global_click_bound = False
        self._root_focus_bind_id = None
        self._root_configure_bind_id = None

        self.grid_columnconfigure(0, weight=1)
        self.button = make_gradient_button(
            self,
            text=self.variable.get() or "Select...",
            command=self.toggle_popup,
            width=width,
            height=height,
            font_size=font_size,
            anchor="w",
            text_padding=12,
            corner_radius=8,
            **SUBTLE_GRADIENT_STYLE,
        )
        self.button.grid(row=0, column=0, sticky="ew")
        self.button.bind(
            "<Button-1>",
            lambda _event: self.button.focus_set(),
            add=True,
        )

        for key, direction in (
            ("<Up>", -1),
            ("<Left>", -1),
            ("<Down>", 1),
            ("<Right>", 1),
        ):
            self.button.bind(key, lambda _event, d=direction: self._step_value(d))
        self.button.bind("<Home>", lambda _event: self._select_index(0))
        self.button.bind(
            "<End>",
            lambda _event: self._select_index(len(self.values) - 1),
        )
        self.variable.trace_add("write", self._sync_button_text)

    def _sync_button_text(self, *_args) -> None:
        self.button.configure_button(
            text=self.variable.get() or "Select..."
        )

    def _select_index(self, index: int):
        if not self.values:
            return "break"
        index = max(0, min(index, len(self.values) - 1))
        value = self.values[index]
        if value != self.variable.get():
            self.variable.set(value)
            if self.command is not None:
                self.command(value)
        return "break"

    def _step_value(self, direction: int):
        if not self.values:
            return "break"
        current = self.variable.get()
        index = self.values.index(current) if current in self.values else 0
        return self._select_index(index + direction)

    @staticmethod
    def _is_descendant(widget, parent) -> bool:
        while widget is not None:
            if widget == parent:
                return True
            widget = getattr(widget, "master", None)
        return False

    # Popup lifecycle and focus handling

    def toggle_popup(self) -> None:
        if self.popup is not None and self.popup.winfo_exists():
            self.close_popup()
        else:
            self.open_popup()

    def open_popup(self) -> None:
        self.close_popup()
        self.update_idletasks()

        popup_width = max(self.winfo_width(), 220)
        visible_items = min(len(self.values), self.max_visible_items)
        font_family = getattr(
            self.winfo_toplevel(),
            "gui_font_family",
            "Courier New",
        )
        current_font_size = max(
            1,
            round(self.dropdown_font_size * TEXT_SIZE_MULTIPLIER),
        )
        # Native Tk uses positive font sizes as points. Reuse CustomTkinter's
        # negative pixel size so list entries match the closed selector button.
        listbox_font = self._apply_font_scaling(
            (font_family, current_font_size)
        )
        row_height = tkfont.Font(
            root=self,
            family=listbox_font[0],
            size=listbox_font[1],
        ).metrics("linespace")
        popup_height = max(1, visible_items) * row_height + 10
        x = self.winfo_rootx()
        y = self.winfo_rooty() + self.winfo_height()

        self.popup = ctk.CTkToplevel(self)
        self.popup.geometry(f"{popup_width}x{popup_height}+{x}+{y}")
        self.popup.overrideredirect(True)
        self.popup.attributes("-topmost", True)

        popup_frame = ctk.CTkFrame(
            self.popup,
            fg_color="#1A1A1A",
            corner_radius=0,
        )
        popup_frame.pack(fill="both", expand=True)
        popup_frame.grid_rowconfigure(0, weight=1)
        popup_frame.grid_columnconfigure(0, weight=1)

        self._listbox = tk.Listbox(
            popup_frame,
            activestyle="none",
            background="#1A1A1A",
            borderwidth=0,
            exportselection=False,
            font=listbox_font,
            foreground="white",
            highlightbackground="#393939",
            highlightcolor="#7652D6",
            highlightthickness=1,
            relief="flat",
            selectbackground="#4800C7",
            selectforeground="white",
        )
        self._listbox.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(4, 0),
            pady=4,
        )

        for value in self.values:
            self._listbox.insert("end", value)

        if len(self.values) > self.max_visible_items:
            scrollbar = ctk.CTkScrollbar(
                popup_frame,
                command=self._listbox.yview,
                width=14,
            )
            scrollbar.grid(
                row=0,
                column=1,
                sticky="ns",
                padx=(2, 4),
                pady=4,
            )
            self._listbox.configure(yscrollcommand=scrollbar.set)

        selected_value = self.variable.get()

        if selected_value in self.values:
            selected_index = self.values.index(selected_value)
            self._listbox.selection_set(selected_index)
            self._listbox.activate(selected_index)
            self._listbox.see(selected_index)

        def on_mouse_wheel(event):
            if self._listbox is None:
                return "break"
            if getattr(event, "num", None) == 4:
                direction = -3
            elif getattr(event, "num", None) == 5:
                direction = 3
            else:
                direction = int(-event.delta / 120) * 3
            self._listbox.yview_scroll(direction, "units")
            return "break"

        self._listbox.bind("<ButtonRelease-1>", self._accept_listbox_selection)
        self._listbox.bind("<Return>", self._accept_listbox_selection)
        self._listbox.bind("<KP_Enter>", self._accept_listbox_selection)
        self._listbox.bind("<Escape>", lambda _event: self.close_popup())
        for event_name in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self._listbox.bind(event_name, on_mouse_wheel)
        self.popup.bind("<Escape>", lambda _event: self.close_popup())
        self._listbox.focus_force()
        self._bind_auto_close_events()

    def _accept_listbox_selection(self, _event=None):
        if self._listbox is None:
            return "break"
        selection = self._listbox.curselection()
        if selection:
            self.select_value(self.values[selection[0]])
        return "break"

    def _bind_auto_close_events(self) -> None:
        root = self.winfo_toplevel()
        root.bind_all("<ButtonPress-1>", self._handle_global_click, add="+")
        self._global_click_bound = True
        self._root_focus_bind_id = root.bind(
            "<FocusOut>",
            self._handle_focus_out,
            add="+",
        )
        self._root_configure_bind_id = root.bind(
            "<Configure>",
            lambda _event: self.close_popup(),
            add="+",
        )

    def _unbind_auto_close_events(self) -> None:
        root = self.winfo_toplevel()
        if self._global_click_bound:
            try:
                root.unbind_all("<ButtonPress-1>")
            except Exception:
                pass
            self._global_click_bound = False
        if self._root_focus_bind_id is not None:
            try:
                root.unbind("<FocusOut>", self._root_focus_bind_id)
            except Exception:
                pass
        if self._root_configure_bind_id is not None:
            try:
                root.unbind("<Configure>", self._root_configure_bind_id)
            except Exception:
                pass
        self._root_focus_bind_id = None
        self._root_configure_bind_id = None

    def _handle_global_click(self, event) -> None:
        if self.popup is None or not self.popup.winfo_exists():
            return
        if self._is_descendant(event.widget, self):
            return
        if self._is_descendant(event.widget, self.popup):
            return
        self.close_popup()

    def _handle_focus_out(self, _event) -> None:
        self.after(100, self._close_if_focus_is_outside)

    def _close_if_focus_is_outside(self) -> None:
        if self.popup is None or not self.popup.winfo_exists():
            return
        focused = self.focus_get()
        if focused is None:
            self.close_popup()
            return
        if not self._is_descendant(focused, self) and not self._is_descendant(
            focused,
            self.popup,
        ):
            self.close_popup()

    def close_popup(self) -> None:
        self._unbind_auto_close_events()
        if self.popup is not None and self.popup.winfo_exists():
            self.popup.destroy()
        self.popup = None
        self._listbox = None

    def select_value(self, value: str) -> None:
        self.variable.set(value)
        self.close_popup()
        if self.command is not None:
            self.command(value)
        self.after_idle(self.button.focus_set)

    def get(self) -> str:
        return self.variable.get()

    def set(self, value: str) -> None:
        if value not in self.values:
            raise ValueError(f"Unknown dropdown value: {value}")
        self.variable.set(value)

    def set_values(
        self,
        values: list[str],
        preferred_value: str | None = None,
    ) -> None:
        self.values = list(values)
        self.close_popup()
        current = self.variable.get()
        if preferred_value is not None and preferred_value in self.values:
            self.variable.set(preferred_value)
        elif current in self.values:
            self.variable.set(current)
        elif self.values:
            self.variable.set(self.values[0])
        else:
            self.variable.set("")

class ButtonTabView(ctk.CTkFrame):
    """Button-based tab view with overflow handling."""

    @staticmethod
    def _bind_tab_command(
        command: Callable[[str], None] | None,
        tab_name: str,
    ) -> Callable[[], None] | None:
        if command is None:
            return None
        return lambda: command(tab_name)

    def __init__(
        self,
        master,
        *,
        button_width=120,
        max_button_width_factor: float | None = None,
        button_text_padding: int = 28,
        button_height=UI["button_height"],
        button_gap=6,
        button_strip_sticky: Literal["", "e", "w"] = "w",
        button_corner_radius=10,
        button_border_width=1,
        overflow_width=110,
        overflow_text="More...",
        wrap_buttons: bool = False,
        reorderable: bool = False,
        reorder_command: Callable[[list[str]], None] | None = None,
        add_command: Callable[[], None] | None = None,
        close_command: Callable[[str], None] | None = None,
        close_button_width: int = 26,
        close_font: ctk.CTkFont | None = None,
        rename_command: Callable[[str], None] | None = None,
        rename_button_width: int = 30,
        rename_font: ctk.CTkFont | None = None,
        font: ctk.CTkFont | None = None,
        selected_color=("#6236D9", "#4800C7"),
        selected_hover_color=("#7A46E8", "#6624E8"),
        selected_border_color=("#8D73DF", "#7652D6"),
        selected_text_color="white",
        unselected_color=("gray90", "gray14"),
        unselected_hover_color=("#CEC2E9", "#382653"),
        unselected_border_color=("gray75", "gray30"),
        unselected_text_color=("black", "white"),
        selected_gradient_style: dict | None = None,
        unselected_gradient_style: dict | None = None,
        fg_color="#1A1A1A",
        content_fg_color="#1A1A1A",
        corner_radius=50,
        **kwargs,
    ):
        super().__init__(master, fg_color=fg_color, corner_radius=50, **kwargs)
        if max_button_width_factor is not None and max_button_width_factor < 1.0:
            raise ValueError("max_button_width_factor must be at least 1.0.")

        self.button_width = button_width
        self.max_button_width_factor = max_button_width_factor
        self.button_text_padding = button_text_padding
        self.button_height = button_height
        self.button_gap = button_gap
        self.button_strip_sticky = button_strip_sticky
        self.button_corner_radius = button_corner_radius
        self.button_border_width = button_border_width
        self.overflow_width = overflow_width
        self.overflow_text = overflow_text
        self.wrap_buttons = bool(wrap_buttons)
        self.reorderable = bool(reorderable)
        self.reorder_command = reorder_command
        self.add_command = add_command
        self.close_command = close_command
        self.close_button_width = close_button_width
        self.close_font = close_font
        self.rename_command = rename_command
        self.rename_button_width = rename_button_width
        self.rename_font = rename_font
        self.font = font

        self.selected_color = selected_color
        self.selected_hover_color = selected_hover_color
        self.selected_border_color = selected_border_color
        self.selected_text_color = selected_text_color
        self.unselected_color = unselected_color
        self.unselected_hover_color = unselected_hover_color
        self.unselected_border_color = unselected_border_color
        self.unselected_text_color = unselected_text_color

        self.selected_gradient_style = dict(
            selected_gradient_style
            or TAB_SELECTED_GRADIENT_STYLE
        )
        self.unselected_gradient_style = dict(
            unselected_gradient_style
            or TAB_UNSELECTED_GRADIENT_STYLE
        )

        self._tabs: dict[str, ctk.CTkFrame] = {}
        self._buttons: dict[str, GradientButton] = {}
        self._tab_border_highlights: dict[
            str,
            tuple[object, float, float],
        ] = {}
        self._tab_order: list[str] = []
        self._button_hidden_tabs: set[str] = set()
        self._selected_name: str | None = None
        self._reflow_job = None
        self._drag_name: str | None = None
        self._drag_start_root: tuple[int, int] | None = None
        self._drag_started = False

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.button_bar = ctk.CTkFrame(
            self,
            fg_color="transparent",
            corner_radius=50,
        )
        self.button_bar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        self.button_bar.grid_columnconfigure(0, weight=1)

        self.button_strip = ctk.CTkFrame(
            self.button_bar,
            fg_color="transparent",
            corner_radius=50,
        )
        self.button_strip.grid(
            row=0,
            column=0,
            sticky=("ew" if self.wrap_buttons else self.button_strip_sticky),
        )
        if self.wrap_buttons:
            self.button_strip.grid_columnconfigure(0, weight=0)

        self.overflow_var = ctk.StringVar(value=self.overflow_text)
        self.overflow_menu = ctk.CTkOptionMenu(
            self.button_bar,
            variable=self.overflow_var,
            values=[""],
            width=self.overflow_width,
            height=self.button_height,
            corner_radius=self.button_corner_radius,
            font=self.font,
            fg_color=self.unselected_color,
            button_color=self.selected_color,
            button_hover_color=self.selected_hover_color,
            text_color=self.unselected_text_color,
            dropdown_fg_color=("gray90", "gray14"),
            dropdown_hover_color=self.unselected_hover_color,
            dropdown_text_color=self.unselected_text_color,
            command=self._select_from_overflow,
        )
        self._hide_overflow_menu()

        self.add_button = None
        if self.add_command is not None:
            self.add_button = GradientButton(
                self.button_strip,
                text="+",
                command=self.add_command,
                width=self.button_height,
                height=self.button_height,
                corner_radius=self.button_corner_radius,
                font=self.font,
                text_color=self.unselected_text_color,
                **self.unselected_gradient_style,
            )

        self.content_frame = ctk.CTkFrame(
            self,
            fg_color=content_fg_color,
            corner_radius=50,
        )
        self.content_frame.grid(row=1, column=0, sticky="nsew")
        self.content_frame.grid_columnconfigure(0, weight=1)
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.bind("<Configure>", self._on_size_changed, add="+")  # type: ignore

    def _measurement_font(self) -> ctk.CTkFont | tkfont.Font:
        if isinstance(self.font, ctk.CTkFont):
            return self.font
        return tkfont.nametofont("TkDefaultFont", root=self)

    def _get_button_text_and_width(self, full_name: str) -> tuple[str, int]:
        leading_width = (
            self.rename_button_width
            if self.rename_command is not None
            else 0
        )
        secondary_width = (
            self.close_button_width
            if self.close_command is not None
            else 0
        )
        if self.max_button_width_factor is None:
            button_width = (
                self.button_width
                + leading_width
                + secondary_width
            )
        else:
            button_width = (
                round(self.button_width * self.max_button_width_factor)
                + leading_width
                + secondary_width
            )

        available_width = max(
            1,
            button_width
            - self.button_text_padding
            - leading_width
            - secondary_width,
        )

        font = self._measurement_font()
        if font.measure(full_name) <= available_width:
            return full_name, button_width

        lower, upper = 0, len(full_name)
        while lower < upper:
            middle = (lower + upper + 1) // 2
            candidate = full_name[:middle].rstrip() + "..."
            if font.measure(candidate) <= available_width:
                lower = middle
            else:
                upper = middle - 1
        return full_name[:lower].rstrip() + "...", button_width

    def add(
        self,
        name: str,
        *,
        defer_reflow: bool = False,
        show_button: bool = True,
    ) -> None:
        if name in self._tabs:
            raise ValueError(f"Tab already exists: {name}")

        tab_frame = ctk.CTkFrame(
            self.content_frame,
            fg_color="transparent",
            corner_radius=50,
        )
        tab_frame.grid(row=0, column=0, sticky="nsew")
        display_name, display_width = self._get_button_text_and_width(name)
        button = GradientButton(
            self.button_strip,
            text=display_name,
            width=display_width,
            height=self.button_height,
            corner_radius=self.button_corner_radius,
            font=self.font,
            text_color=self.unselected_text_color,
            leading_text="⋮" if self.rename_command is not None else None,
            leading_command=self._bind_tab_command(self.rename_command, name),
            leading_width=self.rename_button_width,
            leading_font=self.rename_font,
            leading_tooltip_text="Rename",
            leading_visible=False,
            secondary_text="×" if self.close_command is not None else None,
            secondary_command=self._bind_tab_command(self.close_command, name),
            secondary_width=self.close_button_width,
            secondary_font=self.close_font,
            secondary_tooltip_text="Close",
            secondary_visible=(
                self.close_command is not None and len(self._tabs) >= 1
            ),
            tooltip_text=(name if display_name != name else None),
            command=lambda tab_name=name: self.set(
                tab_name
            ),
            **self.unselected_gradient_style,
        )
        if self.reorderable:
            self._bind_button_drag_events(name, button)

        self._tabs[name] = tab_frame
        self._buttons[name] = button
        self._tab_order.append(name)
        if not show_button:
            self._button_hidden_tabs.add(name)
        if self._selected_name is None:
            self.set(name)
        else:
            self._tabs[self._selected_name].tkraise()
            self._update_button_styles()
            if not defer_reflow:
                self._schedule_reflow()

    def tab(self, name: str) -> ctk.CTkFrame:
        if name not in self._tabs:
            raise KeyError(f"Unknown tab: {name}")
        return self._tabs[name]

    def set(self, name: str) -> None:
        if name not in self._tabs:
            raise KeyError(f"Unknown tab: {name}")

        previous_name = self._selected_name

        self._selected_name = name
        self._tabs[name].tkraise()
        self._update_button_styles()
        self._schedule_reflow()

        if previous_name != name:
            self.event_generate("<<TabChanged>>", when="tail")

    def get(self) -> str:
        return self._selected_name or ""

    def ordered_names(self) -> list[str]:
        """Return tab names in their current visual order."""

        return list(self._tab_order)

    def _bind_button_drag_events(
        self,
        name: str,
        button: GradientButton,
    ) -> None:
        button._tab_drag_name = name
        canvas = button._canvas
        canvas.bind(
            "<ButtonPress-1>",
            lambda event, tab_button=button: self._start_button_drag(
                tab_button._tab_drag_name,
                event,
            ),
            add="+",
        )
        canvas.bind(
            "<B1-Motion>",
            self._continue_button_drag,
            add="+",
        )
        canvas.bind(
            "<ButtonRelease-1>",
            self._finish_button_drag,
            add="+",
        )

    def _start_button_drag(self, name: str, event) -> None:
        button = self._buttons.get(name)
        if button is None:
            return

        # Rename and close remain regular click targets and never start a drag.
        if button._leading_visible and event.x < button._leading_width:
            return
        if (
            button._secondary_visible
            and event.x >= button._canvas.winfo_width() - button._secondary_width
        ):
            return

        self._drag_name = name
        self._drag_start_root = (event.x_root, event.y_root)
        self._drag_started = False

    def _continue_button_drag(self, event) -> None:
        if self._drag_name is None or self._drag_start_root is None:
            return

        if not self._drag_started:
            distance = abs(event.x_root - self._drag_start_root[0]) + abs(
                event.y_root - self._drag_start_root[1]
            )
            if distance < 8:
                return
            self._drag_started = True
            self._buttons[self._drag_name]._canvas.configure(cursor="fleur")
            self._update_button_styles()

        candidates = [
            name
            for name in self._tab_order
            if name not in self._button_hidden_tabs
            and name != self._drag_name
            and self._buttons[name].winfo_ismapped()
        ]
        if not candidates:
            return

        target_name = min(
            candidates,
            key=lambda name: (
                event.x_root
                - (
                    self._buttons[name].winfo_rootx()
                    + self._buttons[name].winfo_width() / 2
                )
            ) ** 2
            + (
                event.y_root
                - (
                    self._buttons[name].winfo_rooty()
                    + self._buttons[name].winfo_height() / 2
                )
            ) ** 2,
        )
        target_button = self._buttons[target_name]
        inside_target = (
            target_button.winfo_rootx()
            <= event.x_root
            < target_button.winfo_rootx() + target_button.winfo_width()
            and target_button.winfo_rooty()
            <= event.y_root
            < target_button.winfo_rooty() + target_button.winfo_height()
        )
        if not inside_target:
            return

        old_index = self._tab_order.index(self._drag_name)
        target_index = self._tab_order.index(target_name)
        if old_index == target_index:
            return
        self._tab_order.pop(old_index)
        self._tab_order.insert(target_index, self._drag_name)
        self._reflow_buttons()
        if self.reorder_command is not None:
            self.reorder_command(list(self._tab_order))

    def _finish_button_drag(self, _event=None) -> None:
        if self._drag_name in self._buttons:
            self._buttons[self._drag_name]._canvas.configure(cursor="hand2")
        self._drag_name = None
        self._drag_start_root = None
        self._drag_started = False
        self._update_button_styles()

    def set_tab_border_highlight(
        self,
        name: str,
        enabled: bool,
        *,
        color: object = VIEWER_ACTIVE_BORDER_COLOR,
        width: float = 1.0,
        opacity: float = 1.0,
    ) -> None:
        """Persistently highlight one tab independently of selection state."""

        if name not in self._tabs:
            raise KeyError(f"Unknown tab: {name}")
        if enabled:
            self._tab_border_highlights[name] = (
                color,
                float(width),
                float(np.clip(opacity, 0.0, 1.0)),
            )
        else:
            self._tab_border_highlights.pop(name, None)
        self._update_button_styles()

    def refresh_text_layout(self) -> None:
        for name, button in self._buttons.items():
            display_name, display_width = self._get_button_text_and_width(name)
            button.configure_button(
                text=display_name,
                width=display_width,
                tooltip_text=(name if display_name != name else None),
            )
        self._schedule_reflow()

    def rename(self, old_name: str, new_name: str) -> None:
        if old_name not in self._tabs:
            raise KeyError(f"Unknown tab: {old_name}")
        if new_name in self._tabs:
            raise ValueError(f"Tab already exists: {new_name}")

        tab_frame = self._tabs.pop(old_name)
        button = self._buttons.pop(old_name)
        self._tabs[new_name] = tab_frame
        self._buttons[new_name] = button
        if old_name in self._tab_border_highlights:
            self._tab_border_highlights[new_name] = (
                self._tab_border_highlights.pop(old_name)
            )
        if old_name in self._button_hidden_tabs:
            self._button_hidden_tabs.remove(old_name)
            self._button_hidden_tabs.add(new_name)
        self._tab_order[self._tab_order.index(old_name)] = new_name

        display_name, display_width = self._get_button_text_and_width(new_name)
        button.configure_button(
            text=display_name,
            width=display_width,
            command=lambda tab_name=new_name: self.set(
                tab_name
            ),
            leading_command=self._bind_tab_command(
                self.rename_command,
                new_name,
            ),
            secondary_command=self._bind_tab_command(
                self.close_command,
                new_name,
            ),
            tooltip_text=(new_name if display_name != new_name else None),
        )
        if self.reorderable:
            button._tab_drag_name = new_name
        if self._selected_name == old_name:
            self._selected_name = new_name
        self._update_button_styles()
        self._schedule_reflow()

    def delete(self, name: str) -> None:
        if name not in self._tabs:
            raise KeyError(f"Unknown tab: {name}")

        deleted_index = self._tab_order.index(name)
        self._tabs.pop(name).destroy()
        self._buttons.pop(name).destroy()
        self._tab_border_highlights.pop(name, None)
        self._button_hidden_tabs.discard(name)
        self._tab_order.remove(name)

        if self._selected_name == name:
            if not self._tab_order:
                self._selected_name = None
            else:
                self.set(self._tab_order[min(deleted_index, len(self._tab_order) - 1)])
        else:
            self._update_button_styles()
            self._schedule_reflow()

    def _update_button_styles(self) -> None:
        for name, button in self._buttons.items():
            selected = (
                name == self._selected_name
            )
            leading_visible = (
                self.rename_command is not None and selected
            )
            secondary_visible = (
                self.close_command is not None and len(self._tabs) > 1
            )

            style = dict(
                self.selected_gradient_style
                if selected
                else self.unselected_gradient_style
            )
            dragging = self._drag_started and name == self._drag_name
            if dragging:
                # Tk widgets do not support real per-widget transparency.
                # A darker neutral fill creates the same lifted/dimmed effect.
                style.update(
                    cmap_name="gist_gray",
                    cmap_start=0.03,
                    cmap_end=0.14,
                    hover_cmap_end=0.14,
                    border_color=VIEWER_ACTIVE_BORDER_COLOR,
                    border_width=2.25,
                    border_opacity=1.0,
                )
            border_highlight = self._tab_border_highlights.get(name)
            if border_highlight is not None and not dragging:
                border_color, border_width, border_opacity = border_highlight
                style.update(
                    border_color=border_color,
                    border_width=border_width,
                    border_opacity=border_opacity,
                )

            button.configure_button(
                **style,
                leading_visible=leading_visible,
                secondary_visible=secondary_visible,
                text_color=(
                    self.selected_text_color
                    if selected
                    else self.unselected_text_color
                ),
            )

    def _on_size_changed(self, _event=None) -> None:
        self._schedule_reflow()

    def _show_overflow_menu(self) -> None:
        self.overflow_menu.grid(
            row=0,
            column=1,
            sticky="e",
            padx=(self.button_gap, 0),
        )

    def _hide_overflow_menu(self) -> None:
        self.overflow_menu.grid_forget()

    def _schedule_reflow(self) -> None:
        if self._reflow_job is not None:
            try:
                self.after_cancel(self._reflow_job)
            except Exception:
                pass
        self._reflow_job = self.after_idle(self._reflow_buttons)

    def _reflow_buttons(self) -> None:
        self._reflow_job = None
        button_names = [
            name
            for name in self._tab_order
            if name not in self._button_hidden_tabs
        ]
        if self.wrap_buttons:
            self._reflow_wrapped_buttons(button_names)
            return

        if not button_names:
            self._hide_overflow_menu()
            if self.add_button is not None:
                self.add_button.pack(side="left")
            return

        for button in self._buttons.values():
            button.pack_forget()
        if self.add_button is not None:
            self.add_button.pack_forget()
        self._hide_overflow_menu()
        self.update_idletasks()

        available_width = self.button_bar.winfo_width()
        if available_width <= 1:
            self._reflow_job = self.after(50, self._reflow_buttons)
            return

        add_button_width = 0
        if self.add_button is not None:
            add_button_width = (
                self.add_button.winfo_reqwidth()
                + self.button_gap
            )
        available_tab_width = max(0, available_width - add_button_width)

        widths = {
            name: self._buttons[name].winfo_reqwidth()
            for name in button_names
        }
        required_width = sum(widths.values()) + max(
            len(button_names) - 1,
            0,
        ) * self.button_gap

        if required_width <= available_tab_width:
            visible_names = list(button_names)
        else:
            self._show_overflow_menu()
            self.update_idletasks()
            usable_width = max(
                0,
                available_tab_width
                - self.overflow_menu.winfo_reqwidth()
                - self.button_gap,
            )
            visible_names = []
            used_width = 0
            for name in button_names:
                additional = widths[name] + (self.button_gap if visible_names else 0)
                if used_width + additional > usable_width:
                    break
                visible_names.append(name)
                used_width += additional

            selected_button_name = (
                self._selected_name
                if self._selected_name in button_names
                else None
            )
            if not visible_names and selected_button_name is not None:
                visible_names = [selected_button_name]
            elif (
                selected_button_name is not None
                and selected_button_name not in visible_names
            ):
                visible_names[-1] = selected_button_name
                visible_names = [
                    name for name in button_names if name in visible_names
                ]

        hidden_names = [
            name for name in button_names if name not in visible_names
        ]
        for index, name in enumerate(visible_names):
            right_padding = self.button_gap if index < len(visible_names) - 1 else 0
            self._buttons[name].pack(side="left", padx=(0, right_padding))

        if self.add_button is not None:
            self.add_button.pack(
                side="left",
                padx=(self.button_gap if visible_names else 0, 0),
            )

        if hidden_names:
            self.overflow_menu.configure(values=hidden_names)
            self.overflow_var.set(self.overflow_text)
            self._show_overflow_menu()
        else:
            self._hide_overflow_menu()

    def _reflow_wrapped_buttons(self, button_names: list[str]) -> None:
        """Place all visible tabs in as many rows as the window needs."""

        self._hide_overflow_menu()
        for button in self._buttons.values():
            button.pack_forget()
            button.grid_forget()
        if self.add_button is not None:
            self.add_button.pack_forget()
            self.add_button.grid_forget()

        self.update_idletasks()
        available_width = self.button_bar.winfo_width()
        if available_width <= 1:
            self._reflow_job = self.after(50, self._reflow_buttons)
            return

        items: list[GradientButton] = [
            self._buttons[name] for name in button_names
        ]
        if self.add_button is not None:
            items.append(self.add_button)

        row = 0
        column = 0
        used_width = 0
        for item in items:
            item_width = max(item.winfo_reqwidth(), 1)
            extra_width = item_width + (self.button_gap if column else 0)
            if column and used_width + extra_width > available_width:
                row += 1
                column = 0
                used_width = 0
                extra_width = item_width

            item.grid(
                row=row,
                column=column,
                sticky="w",
                padx=(self.button_gap if column else 0, 0),
                pady=(0, self.button_gap if row >= 0 else 0),
            )
            used_width += extra_width
            column += 1

        # Do not leave an unnecessary gap below the final row.
        for item in items:
            if int(item.grid_info().get("row", -1)) == row:
                item.grid_configure(pady=(0, 0))

    def _select_from_overflow(self, name: str) -> None:
        if name not in self._tabs:
            return
        self.set(name)
        self.after_idle(lambda: self.overflow_var.set(self.overflow_text))

# =============================================================================
# Dataset tab
# =============================================================================


class SpectraTab(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent")
        self.app = app

        self.viewer_figure = None
        self.viewer_axes = None
        self._calibration_cache: dict[str, object] | None = None
        self._viewer_applied_calibration: dict[str, object] | None = None
        self._pending_calibration_for_viewer: dict[str, object] | None = None
        self._calibration_figures: dict[str, Figure] = {}
        self._calibration_canvases: dict[str, FigureCanvasTkAgg] = {}
        self._calibration_toolbars: dict[str, NavigationToolbar2Tk] = {}
        self._pending_viewer_session_state: dict | None = None
        self._pending_content_tab = "Viewer"
        self._scan_file_cache: dict[int,tuple[Path, Path],] = {}
        self._metadata_cache: dict[int,dict[str, float | str],] = {}
        self._spec_file_cache: dict[int,Path,] = {}
        self._calibration_cache: dict[str, object] | None = None

        self._init_variables()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self._build_layout()

    def _init_variables(self) -> None:
        self.search_root_var = ctk.StringVar(value="")
        self.spec_file_var = ctk.StringVar(value="")
        self.spec_search_root_var = ctk.StringVar(value="")

        self.calibration_poly_order_var = ctk.StringVar(value="3")
        self.gaussian_model_var = ctk.StringVar(value="Double Gaussian")
        self.calibration_buffer_var = ctk.StringVar(value="0")
        self.sigma1_poly_order_var = ctk.StringVar(value="2")
        self.sigma2_poly_order_var = ctk.StringVar(value="2")
        self.r_poly_order_var = ctk.StringVar(value="2")
        self.delta_poly_order_var = ctk.StringVar(value="0")

        self.spectra_dir_var = ctk.StringVar(value="")
        self.histogram_dir_var = ctk.StringVar(value="")
        self.apply_global_preferences_var = ctk.BooleanVar(value=False)
        self.metadata_scan_var = ctk.StringVar(value="")
        self.metadata_motor_var = ctk.StringVar(value="")
        self.metadata_value_var = ctk.StringVar(value="—")
        self.expchamber_value_var = ctk.StringVar(value=EXPCHAMBER_EMPTY_TEXT)

        self.search_root_var.trace_add(
            "write",
            self._invalidate_search_caches,
        )
        self.spec_search_root_var.trace_add(
            "write",
            self._invalidate_spec_caches,
        )
        self.spec_file_var.trace_add(
            "write",
            self._invalidate_spec_caches,
        )

    def _build_layout(self) -> None:
        self.workspace = ctk.CTkFrame(
            self,
            fg_color="#1A1A1A",
            corner_radius=0,
        )
        self.workspace.grid(row=0, column=0, sticky="nsew")
        self.workspace.grid_columnconfigure(0, weight=0, minsize=480)
        self.workspace.grid_columnconfigure(
            1,
            weight=0,
            minsize=WORKSPACE_SPLITTER_WIDTH,
        )
        self.workspace.grid_columnconfigure(
            2,
            weight=1,
            minsize=ANALYSIS_PANEL_MIN_WIDTH,
        )
        self.workspace.grid_rowconfigure(0, weight=1)

        # Use the same widget structure and card styling as SPECTRA VIEWER.
        self.setup_panel = ctk.CTkScrollableFrame(
            self.workspace,
            width=480,
            fg_color="#121214",
            border_color=SETUP_BORDER_COLOR,
            border_width=2,
            corner_radius=14,
        )
        self.setup_panel.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=(8, 0),
            # Match the plot and Viewer cards below the content-tab strip:
            # outer margin + tab height + tab gap + inner card margin.
            pady=(50, 12),
        )
        self._build_workspace_splitter()

        self.content_tabs = ButtonTabView(
            self.workspace,
            button_width=180,
            button_height=UI["compact_control_height"] + 6,
            button_gap=6,
            button_strip_sticky="w",
            button_corner_radius=9,
            button_border_width=1,
            selected_text_color="white",
            unselected_text_color=("black", "white"),
            selected_gradient_style={
                **TAB_SELECTED_GRADIENT_STYLE,
            },
            unselected_gradient_style={
                **TAB_UNSELECTED_GRADIENT_STYLE,
            },
            font=ui_font(
                UI["font_tab"],
                family=self.app.gui_font_family,
                bold=True,
            ),
        )
        self.content_tabs.bind(
            "<<TabChanged>>",
            self._on_content_tab_changed,
            add=True,
        )
        self.content_tabs.grid(
            row=0,
            column=2,
            sticky="nsew",
            padx=(0, 4),
            pady=4,
        )
        self.content_tabs.add("Viewer")
        self.content_tabs.add("Diagnostics")

        viewer_tab = self.content_tabs.tab("Viewer")
        calibration_tab = self.content_tabs.tab("Diagnostics")

        # Keep the existing setup-layout code readable while using the outer
        # card itself as the scrolling surface.
        self.setup_frame = self.setup_panel
        self.setup_frame.grid_columnconfigure(0, weight=1)
        self.setup_frame.grid_rowconfigure(1, weight=1)

        self._setup_cards: dict[str, ctk.CTkFrame] = {}
        self._setup_layout_mode: str | None = None

        self.setup_frame._parent_canvas.bind(
            "<Configure>",
            self._on_setup_viewport_configure,
            add="+",
        )

        self.viewer_frame = ctk.CTkFrame(
            viewer_tab,
            fg_color="#1A1A1A",
            corner_radius=0,
        )
        self.viewer_frame.pack(fill="both", expand=True)
        self._build_viewer_placeholder()

        self._build_calibration_results_area(calibration_tab)

        self._build_setup_header()

        self.setup_workspace = ctk.CTkFrame(
            self.setup_frame,
            fg_color="transparent",
            corner_radius=0,
        )
        self.setup_workspace.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=UI["outer_padding"],
            pady=(0, 16),
        )

        self._build_top_inputs(self.setup_workspace)
        self._build_start_button(self.setup_workspace)
        self._build_options_and_status(self.setup_workspace)
        self.after_idle(self._refresh_setup_layout)
        self._install_live_setup_sync()
        self.content_tabs.set("Viewer")

    def _build_workspace_splitter(self) -> None:
        """Create the draggable divider between Setup and analysis."""

        splitter = tk.Frame(
            self.workspace,
            width=WORKSPACE_SPLITTER_WIDTH,
            bd=0,
            highlightthickness=0,
            bg="#1A1A1A",
            cursor="sb_h_double_arrow",
        )
        splitter.grid(row=0, column=1, sticky="ns", pady=10)
        splitter.grid_propagate(False)

        line = ctk.CTkFrame(
            splitter,
            width=3,
            corner_radius=2,
            fg_color=SETUP_BORDER_COLOR,
        )
        line.place(relx=0.5, rely=0.5, relheight=0.96, anchor="center")
        guide = ctk.CTkFrame(
            self.workspace,
            width=3,
            corner_radius=2,
            fg_color=SETUP_ACCENT_COLOR,
        )
        drag_state = {
            "start_x": 0,
            "start_width": 480,
            "pending_width": 480,
        }

        def width_for_pointer(pointer_x: int) -> int:
            available = max(1, self.workspace.winfo_width())
            maximum = max(
                SETUP_PANEL_MIN_WIDTH,
                available
                - ANALYSIS_PANEL_MIN_WIDTH
                - WORKSPACE_SPLITTER_WIDTH,
            )
            return int(np.clip(
                drag_state["start_width"] + pointer_x - drag_state["start_x"],
                SETUP_PANEL_MIN_WIDTH,
                maximum,
            ))

        def begin_drag(event) -> None:
            drag_state["start_x"] = int(event.x_root)
            drag_state["start_width"] = max(
                SETUP_PANEL_MIN_WIDTH,
                self.setup_panel.winfo_width(),
            )
            drag_state["pending_width"] = drag_state["start_width"]
            line.configure(fg_color=SETUP_ACCENT_COLOR)
            guide.place(
                x=drag_state["start_width"] + 4,
                rely=0.5,
                relheight=0.96,
                anchor="center",
            )
            guide.lift()

        def drag(event) -> None:
            width = width_for_pointer(int(event.x_root))
            drag_state["pending_width"] = width
            guide.place_configure(x=width + 4)

        def end_drag(_event=None) -> None:
            width = int(drag_state["pending_width"])
            guide.place_forget()
            cover = self._cover_layout_update(self.workspace)

            def apply_width() -> None:
                if not self._layout_cover_exists(cover):
                    return
                self.workspace.grid_columnconfigure(0, minsize=width)
                self.setup_panel.configure(width=width)
                line.configure(fg_color=SETUP_BORDER_COLOR)
                self.after(35, finish_layout)

            def finish_layout() -> None:
                if not self._layout_cover_exists(cover):
                    return
                self._refresh_setup_layout()
                self._reveal_layout_after_idle(cover)

            # Return from the mouse callback before Tk processes the resize.
            self.after(16, apply_width)

        for widget in (splitter, line):
            widget.bind("<ButtonPress-1>", begin_drag)
            widget.bind("<B1-Motion>", drag)
            widget.bind("<ButtonRelease-1>", end_drag)

        self.workspace_splitter = splitter
        self.workspace_splitter_line = line
        self.workspace_splitter_guide = guide

    def set_setup_panel_width(self, width: int) -> None:
        """Set the Setup width programmatically; also used by session tests."""

        minimum_total_width = (
            SETUP_PANEL_MIN_WIDTH
            + ANALYSIS_PANEL_MIN_WIDTH
            + WORKSPACE_SPLITTER_WIDTH
        )
        available = max(minimum_total_width, self.workspace.winfo_width())
        maximum = max(
            SETUP_PANEL_MIN_WIDTH,
            available - ANALYSIS_PANEL_MIN_WIDTH - WORKSPACE_SPLITTER_WIDTH,
        )
        resolved_width = int(np.clip(width, SETUP_PANEL_MIN_WIDTH, maximum))
        self.workspace.grid_columnconfigure(0, minsize=resolved_width)
        self.setup_panel.configure(width=resolved_width)

    def _cover_layout_update(self, target) -> tk.Frame:
        # Keep intermediate widget reflow hidden until Tk has settled.
        cover = tk.Frame(
            target,
            bg="#1A1A1A",
            bd=0,
            highlightthickness=0,
        )
        cover.place(relx=0, rely=0, relwidth=1, relheight=1)
        cover.lift()
        return cover

    @staticmethod
    def _layout_cover_exists(cover: tk.Frame) -> bool:
        try:
            return bool(cover.winfo_exists())
        except tk.TclError:
            return False

    def _reveal_layout_after_idle(self, cover: tk.Frame) -> None:
        def reveal() -> None:
            try:
                if cover.winfo_exists():
                    cover.destroy()
            except tk.TclError:
                return

        # Give pending wrapping and canvas resize callbacks time to complete.
        self.after(50, reveal)

    def _begin_local_loading(self, target, message: str) -> dict[str, object]:
        # Cover only the area affected by a blocking operation.
        overlay = ctk.CTkFrame(
            target,
            fg_color="#151820",
            border_color=SETUP_BORDER_COLOR,
            border_width=2,
            corner_radius=12,
        )
        overlay.grid_columnconfigure(0, weight=1)
        overlay.grid_rowconfigure(0, weight=1)
        overlay.grid_rowconfigure(3, weight=1)

        make_label(
            overlay,
            "WORKING…",
            size=UI["font_viewer_card_title"],
            bold=True,
            text_color="#F4F6FA",
        ).grid(row=1, column=0, padx=18, pady=(18, 3))
        message_label = make_label(
            overlay,
            str(message),
            size=UI["font_viewer_status"],
            text_color=SETUP_MUTED_TEXT_COLOR,
            anchor="center",
            justify="center",
        )
        message_label.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 18))
        bind_responsive_label_wrap(
            message_label,
            overlay,
            horizontal_padding=36,
        )
        overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        overlay.lift()
        self.update_idletasks()
        return {"overlay": overlay, "message_label": message_label}

    def _update_local_loading(
        self,
        token: dict[str, object],
        message: str,
    ) -> None:
        label = token.get("message_label")
        overlay = token.get("overlay")
        if isinstance(label, ctk.CTkLabel) and label.winfo_exists():
            label.configure(text=str(message))
            refresh_wrap = getattr(label, "_responsive_wrap_update", None)
            if callable(refresh_wrap):
                refresh_wrap()
        if isinstance(overlay, ctk.CTkFrame) and overlay.winfo_exists():
            overlay.lift()
        self.update_idletasks()

    @staticmethod
    def _end_local_loading(token: dict[str, object] | None) -> None:
        if not isinstance(token, dict):
            return
        overlay = token.get("overlay")
        if isinstance(overlay, ctk.CTkFrame) and overlay.winfo_exists():
            overlay.destroy()

    def _build_viewer_placeholder(self) -> None:
        """Reserve the final plot/control geometry before data is loaded."""

        placeholder = ctk.CTkFrame(
            self.viewer_frame,
            fg_color="#1A1A1A",
            corner_radius=0,
        )
        placeholder.pack(fill="both", expand=True)
        placeholder.grid_columnconfigure(0, weight=1)
        placeholder.grid_columnconfigure(1, weight=0, minsize=10)
        placeholder.grid_columnconfigure(2, weight=0, minsize=350)
        placeholder.grid_rowconfigure(0, weight=1)

        plot_card = ctk.CTkFrame(
            placeholder,
            fg_color="#121214",
            border_color=SETUP_BORDER_COLOR,
            border_width=2,
            corner_radius=14,
        )
        plot_card.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=8)
        plot_card.grid_columnconfigure(0, weight=1)
        plot_card.grid_rowconfigure(0, weight=1)

        figure = Figure(figsize=(8.4, 7.2))
        figure.patch.set_facecolor("#121214")
        axes = figure.subplots(2, 1, sharex=True)
        for axis, title in zip(axes, ("Histogram", "1D Spectrum"), strict=True):
            axis.set_facecolor("#121214")
            axis.set_title(title, color="#727C8D", pad=10)
            axis.tick_params(colors="#586170")
            for spine in axis.spines.values():
                spine.set_color("#465162")
            axis.text(
                0.5,
                0.5,
                "Waiting for Viewer data",
                transform=axis.transAxes,
                ha="center",
                va="center",
                color="#727C8D",
                fontsize=13,
            )
        figure.subplots_adjust(
            left=0.10,
            right=0.97,
            top=0.94,
            bottom=0.09,
            hspace=0.24,
        )
        canvas = FigureCanvasTkAgg(figure, master=plot_card)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.configure(background="#121214", highlightthickness=0)
        canvas_widget.grid(row=0, column=0, sticky="nsew", padx=7, pady=7)
        canvas.draw()

        viewer_splitter = tk.Frame(
            placeholder,
            width=10,
            bd=0,
            highlightthickness=0,
            bg="#1A1A1A",
            cursor="sb_h_double_arrow",
        )
        viewer_splitter.grid(row=0, column=1, sticky="ns", pady=14)
        viewer_splitter.grid_propagate(False)
        splitter_line = ctk.CTkFrame(
            viewer_splitter,
            width=3,
            fg_color=SETUP_BORDER_COLOR,
            corner_radius=2,
        )
        splitter_line.place(relx=0.5, rely=0.5, relheight=0.94, anchor="center")

        sidebar = ctk.CTkScrollableFrame(
            placeholder,
            width=342,
            fg_color="#121214",
            border_color=SETUP_BORDER_COLOR,
            border_width=2,
            corner_radius=14,
        )
        sidebar.grid(row=0, column=2, sticky="nsew", padx=(0, 8), pady=8)
        sidebar.grid_columnconfigure(0, weight=1)

        split_guide = ctk.CTkFrame(
            placeholder,
            width=3,
            fg_color=SETUP_ACCENT_COLOR,
            corner_radius=2,
        )
        split_drag_state = {
            "start_x": 0,
            "start_width": 350,
            "pending_width": 350,
        }

        def sidebar_width_for_pointer(pointer_x: int) -> int:
            available = max(1, int(placeholder.winfo_width()))
            maximum = max(300, available - 440)
            delta = pointer_x - split_drag_state["start_x"]
            return int(np.clip(
                split_drag_state["start_width"] - delta,
                300,
                maximum,
            ))

        def begin_split_drag(event) -> None:
            split_drag_state["start_x"] = int(event.x_root)
            split_drag_state["start_width"] = max(
                300,
                int(sidebar.winfo_width()) + 8,
            )
            split_drag_state["pending_width"] = split_drag_state["start_width"]
            splitter_line.configure(fg_color=SETUP_ACCENT_COLOR)
            available = max(1, int(placeholder.winfo_width()))
            split_guide.place(
                x=available - split_drag_state["start_width"] - 5,
                rely=0.5,
                relheight=0.94,
                anchor="center",
            )
            split_guide.lift()

        def drag_split(event) -> None:
            available = max(1, int(placeholder.winfo_width()))
            sidebar_width = sidebar_width_for_pointer(int(event.x_root))
            split_drag_state["pending_width"] = sidebar_width
            split_guide.place_configure(x=available - sidebar_width - 5)

        def end_split_drag(_event=None) -> None:
            sidebar_width = int(split_drag_state["pending_width"])
            split_guide.place_forget()
            cover = self._cover_layout_update(placeholder)

            def apply_width() -> None:
                if not self._layout_cover_exists(cover):
                    return
                placeholder.grid_columnconfigure(2, minsize=sidebar_width)
                sidebar.configure(width=max(280, sidebar_width - 8))
                splitter_line.configure(fg_color=SETUP_BORDER_COLOR)
                canvas.draw_idle()
                self._reveal_layout_after_idle(cover)

            self.after(16, apply_width)

        for widget in (viewer_splitter, splitter_line):
            widget.bind("<ButtonPress-1>", begin_split_drag)
            widget.bind("<B1-Motion>", drag_split)
            widget.bind("<ButtonRelease-1>", end_split_drag)

        make_label(
            sidebar,
            "SPECTRA VIEWER",
            size=UI["font_viewer_title"],
            bold=True,
            text_color="#727C8D",
        ).grid(row=0, column=0, sticky="w", padx=3, pady=(1, 0))
        placeholder_subtitle = make_label(
            sidebar,
            "Controls become available after Start Viewer",
            size=UI["font_viewer_subtitle"],
            text_color="#687181",
            anchor="w",
            justify="left",
        )
        placeholder_subtitle.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=3,
            pady=(0, 10),
        )
        bind_responsive_label_wrap(
            placeholder_subtitle,
            sidebar,
            horizontal_padding=24,
        )

        placeholder_sections = (
            ("Display", "Color mapping, intensity and display rebinning"),
            ("Alignment & ROI", "Tilt, reference line and detector region"),
            ("1D Spectrum", "Binning, axis mode and vertical scale"),
            ("Filters", "Percentile, median and local cleanup"),
            ("Output", "Spectrum and histogram export"),
        )
        for row, (title, note) in enumerate(placeholder_sections, start=2):
            card = ctk.CTkFrame(
                sidebar,
                fg_color="#18191C",
                border_color=SETUP_BORDER_COLOR,
                border_width=2,
                corner_radius=13,
            )
            card.grid(row=row, column=0, sticky="ew", pady=(0, 9))
            make_label(
                card,
                f"▸  {title.upper()}",
                size=UI["font_viewer_card_title"],
                bold=True,
                text_color="#707989",
            ).pack(fill="x", padx=12, pady=(10, 2))
            note_label = make_label(
                card,
                note,
                size=UI["font_small"],
                text_color="#596271",
                anchor="w",
                justify="left",
            )
            note_label.pack(fill="x", padx=12, pady=(0, 10))
            bind_responsive_label_wrap(
                note_label,
                card,
                horizontal_padding=24,
            )

        self.viewer_placeholder = placeholder
        self.viewer_placeholder_splitter = viewer_splitter
        self.viewer_placeholder_figure = figure
        self.viewer_placeholder_canvas = canvas

    def _on_content_tab_changed(self, _event=None) -> None:
        selected = self.content_tabs.get()
        if selected == "Diagnostics":
            self.after_idle(self.refresh_calibration_plot_layout)
            return
        if selected != "Viewer":
            return

        if self.viewer_figure is None:
            return

        self.after_idle(
            self._refresh_open_viewer_axes_after_show
        )


    def _refresh_open_viewer_axes_after_show(self) -> None:
        if self.viewer_figure is None:
            return

        self.update_idletasks()

        keepalive = getattr(
            self.viewer_figure,
            "_view_spectra_keepalive",
            None,
        )

        if not isinstance(keepalive, dict):
            return

        refresh_axes = keepalive.get(
            "refresh_axes_after_show"
        )

        if callable(refresh_axes):
            refresh_axes()

    def _build_calibration_results_area(self, parent) -> None:
        self.calibration_frame = ctk.CTkFrame(
            parent,
            fg_color="#121214",
            border_color=SETUP_BORDER_COLOR,
            border_width=2,
            corner_radius=14,
        )
        self.calibration_frame.pack(fill="both", expand=True, padx=8, pady=8)
        self.calibration_frame.grid_columnconfigure(0, weight=1)
        self.calibration_frame.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(
            self.calibration_frame,
            fg_color="transparent",
            corner_radius=0,
        )
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 6))
        header.grid_columnconfigure(0, weight=1)

        title_frame = ctk.CTkFrame(header, fg_color="transparent")
        title_frame.grid(row=0, column=0, sticky="w")
        make_label(
            title_frame,
            "CALIBRATION DIAGNOSTICS",
            size=UI["font_section"],
            bold=True,
        ).pack(anchor="w")
        make_label(
            title_frame,
            "Diagnostics for the current calibration model",
            size=UI["font_small"],
            text_color=("gray35", "gray65"),
        ).pack(anchor="w", pady=(1, 0))

        self.diagnostics_apply_calibration_button = make_gradient_button(
            header,
            text="Apply New Model",
            command=self.apply_new_calibration_model,
            font_size=UI["font_viewer_button"],
            width=265,
            corner_radius=9,
            **CALIBRATION_GRADIENT_STYLE,
        )
        self.diagnostics_apply_calibration_button.grid(
            row=0,
            column=1,
            sticky="e",
        )
        self.diagnostics_apply_calibration_button.configure_button(
            state="disabled"
        )

        self.calibration_plot_tabs = ButtonTabView(
            self.calibration_frame,
            button_width=175,
            max_button_width_factor=1.35,
            button_height=UI["compact_control_height"] + 6,
            button_gap=6,
            button_strip_sticky="w",
            button_corner_radius=9,
            button_border_width=1,
            fg_color="#1A1A1A",
            content_fg_color="#1A1A1A",
            font=ui_font(
                UI["font_calibration_plot_tab"],
                family=self.app.gui_font_family,
                bold=True,
            ),
        )
        self.calibration_plot_tabs.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=10,
            pady=(0, 10),
        )

        self._calibration_plot_hosts: dict[str, ctk.CTkFrame] = {}
        plot_tabs = (
            ("parameters", "Line-shape parameters"),
            ("line_fits", "Calibration lines"),
            ("energy_calibration", "Energy calibration"),
            ("energy_deviation", "Energy deviation"),
        )
        for plot_key, tab_name in plot_tabs:
            self.calibration_plot_tabs.add(tab_name)
            host = self.calibration_plot_tabs.tab(tab_name)
            host.grid_columnconfigure(0, weight=1)
            host.grid_rowconfigure(0, weight=1)
            self._calibration_plot_hosts[plot_key] = host
            make_label(
                host,
                "Run calibration and diagnostics to generate this plot.",
                size=UI["font_placeholder"],
                bold=True,
                text_color=("gray45", "gray60"),
            ).grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

        self.calibration_plot_tabs.set("Line-shape parameters")

    def _on_setup_viewport_configure(self, event) -> None:
        self._stretch_setup_content_to_viewport(event)
        self._reflow_setup_cards(event.width)
        # Wrapped labels may change card heights without changing layout mode.
        self.after_idle(self._sync_setup_scroll_height)

    def _stretch_setup_content_to_viewport(self, event) -> None:
        """Stretch the setup content to at least the visible canvas height."""

        canvas = self.setup_frame._parent_canvas
        window_id = self.setup_frame._create_window_id

        required_height = self.setup_frame.winfo_reqheight()
        target_height = max(required_height, event.height)

        canvas.itemconfigure(
            window_id,
            height=target_height,
        )

    def _build_setup_header(self) -> None:
        """Create the compact page heading used above the setup cards."""

        header = ctk.CTkFrame(
            self.setup_frame,
            fg_color="transparent",
            corner_radius=0,
        )
        header.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=UI["outer_padding"] + 4,
            pady=(4, 12),
        )
        header.grid_columnconfigure(0, weight=1)

        make_label(
            header,
            "DATASET SETUP",
            size=UI["font_viewer_title"],
            bold=True,
        ).grid(row=0, column=0, sticky="w")
        setup_subtitle = make_label(
            header,
            "Configure data sources, calibration, metadata and output for this dataset.",
            size=UI["font_viewer_subtitle"],
            text_color=SETUP_MUTED_TEXT_COLOR,
            anchor="w",
            justify="left",
        )
        setup_subtitle.grid(row=1, column=0, sticky="ew", pady=(1, 0))
        bind_responsive_label_wrap(
            setup_subtitle,
            header,
            horizontal_padding=4,
        )

    def _refresh_setup_layout(self) -> None:
        width = self.setup_frame._parent_canvas.winfo_width()
        self._reflow_setup_cards(width, force=True)

    def _sync_setup_scroll_height(self) -> None:
        """Refresh the scrollable window after cards change rows or columns."""

        self.setup_frame.update_idletasks()
        canvas = self.setup_frame._parent_canvas
        required_height = self.setup_frame.winfo_reqheight()
        canvas.itemconfigure(
            self.setup_frame._create_window_id,
            height=max(required_height, canvas.winfo_height()),
        )
        canvas.configure(scrollregion=canvas.bbox("all"))

    def _reflow_setup_cards(self, viewport_width: int, *, force: bool = False) -> None:
        """Use two columns when useful and a single column on narrow windows."""

        if not hasattr(self, "setup_workspace"):
            return

        layout_mode = "wide" if viewport_width >= 1180 else "stacked"
        if not force and layout_mode == self._setup_layout_mode:
            return
        self._setup_layout_mode = layout_mode

        for card in self._setup_cards.values():
            card.grid_forget()

        workspace = self.setup_workspace
        if layout_mode == "wide":
            workspace.grid_columnconfigure(0, weight=5)
            workspace.grid_columnconfigure(1, weight=3)
            positions = {
                "inputs": (0, 0),
                "launch": (0, 1),
                "calibration": (1, 0),
                "metadata": (1, 1),
                "output": (2, 0),
                "log": (2, 1),
            }
            for name, (row, column) in positions.items():
                card = self._setup_cards.get(name)
                if card is None:
                    continue
                card.grid(
                    row=row,
                    column=column,
                    sticky="nsew",
                    padx=(0, 5) if column == 0 else (5, 0),
                    pady=5,
                )
        else:
            workspace.grid_columnconfigure(0, weight=1)
            workspace.grid_columnconfigure(1, weight=0)
            order = ("inputs", "launch", "calibration", "metadata", "output", "log")
            for row, name in enumerate(order):
                card = self._setup_cards.get(name)
                if card is not None:
                    card.grid(
                        row=row,
                        column=0,
                        sticky="nsew",
                        padx=0,
                        pady=5,
                    )

        self.after_idle(self._sync_setup_scroll_height)

    def _new_section_frame(
        self,
        parent,
        *,
        name: str,
        title: str,
        subtitle: str,
    ) -> ctk.CTkFrame:
        card = ctk.CTkFrame(
            parent,
            fg_color=SETUP_CARD_COLOR,
            border_color=SETUP_BORDER_COLOR,
            border_width=2,
            corner_radius=13,
        )
        self._setup_cards[name] = card
        card.grid(
            row=len(self._setup_cards) - 1,
            column=0,
            sticky="nsew",
            pady=5,
        )
        card.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(
            card,
            fg_color="transparent",
            corner_radius=0,
        )
        header.grid(row=0, column=0, sticky="ew", padx=13, pady=(10, 2))
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkFrame(
            header,
            width=4,
            height=34,
            fg_color=SETUP_ACCENT_COLOR,
            corner_radius=2,
        ).grid(row=0, column=0, rowspan=2, sticky="ns", padx=(0, 9))

        make_label(
            header,
            title.upper(),
            size=UI["font_viewer_card_title"],
            bold=True,
        ).grid(row=0, column=1, sticky="w")
        subtitle_label = make_label(
            header,
            subtitle,
            size=UI["font_viewer_card_note"],
            text_color=SETUP_MUTED_TEXT_COLOR,
            anchor="w",
            justify="left",
        )
        subtitle_label.grid(row=1, column=1, sticky="ew", pady=(0, 1))
        bind_responsive_label_wrap(
            subtitle_label,
            header,
            horizontal_padding=36,
        )

        content = ctk.CTkFrame(
            card,
            fg_color="transparent",
            corner_radius=0,
        )
        content.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=13,
            pady=(7, 13),
        )
        card.grid_rowconfigure(1, weight=1)
        return content

    @overload
    def _add_path_row(
        self,
        parent,
        *,
        row: int,
        label: str,
        variable: ctk.StringVar,
        choose_command: Callable[[], None],
        combo: Literal[True],
        placeholder: str = "",
        choose_text: str = "Choose...",
        pady: int = 8,
    ) -> ctk.CTkComboBox:
        ...


    @overload
    def _add_path_row(
        self,
        parent,
        *,
        row: int,
        label: str,
        variable: ctk.StringVar,
        choose_command: Callable[[], None],
        combo: Literal[False],
        placeholder: str = "",
        choose_text: str = "Choose...",
        pady: int = 8,
    ) -> ctk.CTkEntry:
        ...

    def _add_path_row(
        self,
        parent,
        *,
        row: int,
        label: str,
        variable: ctk.StringVar,
        choose_command: Callable[[], None],
        combo: bool,
        placeholder: str = "",
        choose_text: str = "Choose...",
        pady: int = 8,
    ) -> ctk.CTkComboBox | ctk.CTkEntry:
        row_frame = ctk.CTkFrame(
            parent,
            fg_color="transparent",
            corner_radius=0,
        )
        row_frame.grid(
            row=row,
            column=0,
            sticky="ew",
            pady=pady,
        )
        row_frame.grid_columnconfigure(0, weight=1)

        make_label(
            row_frame,
            label,
            size=UI["font_viewer_control_label"],
        ).grid(
            row=0,
            column=0,
            sticky="w",
            pady=(0, 4),
        )

        picker = ctk.CTkFrame(
            row_frame,
            fg_color="transparent",
            corner_radius=0,
        )
        picker.grid(row=1, column=0, sticky="ew")
        picker.grid_columnconfigure(0, weight=1)

        if combo:
            field: ctk.CTkComboBox | ctk.CTkEntry = make_combobox(
                picker,
                variable=variable,
                values=[],
                family=self.app.gui_font_family,
                state="normal",
                font_size=UI["font_viewer_value"],
                dropdown_font_size=UI["font_viewer_value"],
            )
        else:
            field = make_entry(
                picker,
                variable=variable,
                placeholder=placeholder,
                family=self.app.gui_font_family,
                font_size=UI["font_viewer_value"],
            )

        field.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 7),
        )

        make_gradient_button(
            picker,
            text=choose_text,
            width=108,
            command=choose_command,
            font_size=UI["font_viewer_button"],
            corner_radius=8,
            **SUBTLE_GRADIENT_STYLE,
        ).grid(
            row=0,
            column=1,
            padx=(0, 7),
        )

        make_gradient_button(
            picker,
            text="Clear",
            width=72,
            command=lambda: variable.set(""),
            font_size=UI["font_viewer_button"],
            corner_radius=8,
            **SUBTLE_GRADIENT_STYLE,
        ).grid(
            row=0,
            column=2,
        )

        return field

    def _invalidate_search_caches(
        self,
        *_args,
    ) -> None:
        """Discard caches affected by a scan search-root change."""

        self._scan_file_cache.clear()
        self._metadata_cache.clear()
        self._spec_file_cache.clear()

        self._calibration_cache = None
        self._pending_calibration_for_viewer = None
        self._sync_apply_calibration_button_state()


    def _invalidate_spec_caches(
        self,
        *_args,
    ) -> None:
        """Discard caches affected by a SPEC search-root change."""

        self._metadata_cache.clear()
        self._spec_file_cache.clear()

        self._calibration_cache = None
        self._pending_calibration_for_viewer = None
        self._sync_apply_calibration_button_state()

    def _build_top_inputs(self, parent) -> None:
        frame = self._new_section_frame(
            parent,
            name="inputs",
            title="Data Sources",
            subtitle="Scans and locations used to discover detector and SPEC files",
        )
        frame.grid_columnconfigure(0, weight=1)

        make_label(
            frame,
            "Scans",
            size=UI["font_viewer_control_label"],
            bold=True,
        ).grid(
            row=0,
            column=0,
            sticky="w",
            pady=(0, 4),
        )
        self.scans_entry = make_entry(
            frame,
            family=self.app.gui_font_family,
            font_size=UI["font_viewer_value"],
            placeholder=(
                "Examples: 101:103 | 101,102,1003 | 101-103 | "
                "101:103, 105:107, 109:110"
            ),
        )
        self.scans_entry.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(0, 9),
        )

        make_label(
            frame,
            "Calibration Scans",
            size=UI["font_viewer_control_label"],
            bold=True,
        ).grid(
            row=2,
            column=0,
            sticky="w",
            pady=(0, 4),
        )
        self.calibration_scans_entry = make_entry(
            frame,
            family=self.app.gui_font_family,
            font_size=UI["font_viewer_value"],
            placeholder="Optional. Same format as Scans.",
        )
        self.calibration_scans_entry.grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(0, 7),
        )

        for entry in (self.scans_entry, self.calibration_scans_entry):
            entry.bind(
                "<FocusOut>",
                self._on_scan_input_focus_out,
                add="+",  # type: ignore
            )

        self.search_root_combo = self._add_path_row(
            frame,
            row=4,
            label="Scan Files Search Root",
            variable=self.search_root_var,
            choose_command=self.choose_search_root,
            combo=True,
        )
        self.spec_file_entry = self._add_path_row(
            frame,
            row=5,
            label="Spec File",
            variable=self.spec_file_var,
            choose_command=self.choose_spec_file,
            combo=False,
            placeholder="Optional. Overrides Spec File Search Root.",
            choose_text="Choose...",
        )
        self.spec_search_root_combo = self._add_path_row(
            frame,
            row=6,
            label="Spec File Search Root",
            variable=self.spec_search_root_var,
            choose_command=self.choose_spec_search_root,
            combo=True,
        )

        self.spec_search_root_combo.configure(
            command=lambda _value: (
                self._on_spec_search_root_committed()
            )
        )

        self.spec_search_root_combo.bind(
            "<Return>",
            self._on_spec_search_root_committed,
                add="+",  # type: ignore
        )

        self.spec_search_root_combo.bind(
            "<FocusOut>",
            self._on_spec_search_root_committed,
            add="+",  # type: ignore
        )

        for event_name in ("<Return>", "<FocusOut>"):
            self.spec_file_entry.bind(
                event_name,
                self._on_spec_file_committed,
                add="+",  # type: ignore
            )

    def _build_start_button(self, parent) -> None:
        frame = self._new_section_frame(
            parent,
            name="launch",
            title="Launch Viewer",
            subtitle="Open this dataset in the embedded Spectra Viewer",
        )
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)

        launch_note = make_label(
            frame,
            "The current scan sources and calibration settings are used when the Viewer starts.",
            size=UI["font_viewer_card_note"],
            text_color=SETUP_MUTED_TEXT_COLOR,
            justify="left",
            anchor="w",
        )
        launch_note.grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 12),
        )
        bind_responsive_label_wrap(launch_note, frame)

        workflow = ctk.CTkFrame(
            frame,
            fg_color=SETUP_CONTROL_COLOR,
            corner_radius=9,
        )
        workflow.grid(row=1, column=0, sticky="new", pady=(0, 12))
        workflow.grid_columnconfigure(1, weight=1)
        for row, (number, text) in enumerate(
            (
                ("1", "Select scans and data sources"),
                ("2", "Adjust calibration if required"),
                ("3", "Start the embedded Viewer"),
            )
        ):
            make_label(
                workflow,
                number,
                size=UI["font_viewer_control_label"],
                bold=True,
                width=26,
                height=26,
                fg_color=SETUP_ACCENT_COLOR,
                corner_radius=13,
            ).grid(row=row, column=0, padx=(10, 8), pady=6)
            workflow_label = make_label(
                workflow,
                text,
                size=UI["font_viewer_control_label"],
                anchor="w",
                justify="left",
            )
            workflow_label.grid(
                row=row,
                column=1,
                sticky="ew",
                padx=(0, 10),
                pady=6,
            )
            bind_responsive_label_wrap(
                workflow_label,
                workflow,
                horizontal_padding=58,
            )

        self.apply_global_preferences_checkbox = ctk.CTkCheckBox(
            frame,
            text="Apply global preferences",
            variable=self.apply_global_preferences_var,
            onvalue=True,
            offvalue=False,
            width=220,
            checkbox_width=20,
            checkbox_height=20,
            font=ui_font(
                UI["font_viewer_control_label"],
                family=self.app.gui_font_family,
            ),
        )
        self.apply_global_preferences_checkbox.grid(
            row=2,
            column=0,
            sticky="w",
            pady=(0, 12),
        )

        self.run_button = make_gradient_button(
            frame,
            text="Start Viewer",
            command=self.run_viewer,
            height=UI["start_button_height"],
            font_size=UI["font_viewer_card_title"],
            bold=True,
            corner_radius=10,
            text_color="white",
            **PRIMARY_GRADIENT_STYLE,
        )
        self.run_button.grid(
            row=3,
            column=0,
            sticky="ew",
        )


    def _build_options_and_status(self, parent) -> None:
        self._build_calibration_section(parent)
        self._build_metadata_section(parent)
        self._build_save_section(parent)
        self._build_log_section(parent)

    def _build_calibration_section(self, parent) -> None:
        frame = self._new_section_frame(
            parent,
            name="calibration",
            title="Calibration",
            subtitle="Peak model, polynomial orders and diagnostic workflow",
        )
        frame.grid_columnconfigure(0, weight=1)

        top_controls = ctk.CTkFrame(
            frame,
            fg_color="transparent",
            corner_radius=0,
        )
        top_controls.grid(row=0, column=0, sticky="ew", pady=(0, 7))
        for column, weight in enumerate((2, 3, 2)):
            top_controls.grid_columnconfigure(column, weight=weight)

        polynomial_group = ctk.CTkFrame(
            top_controls,
            fg_color="transparent",
            corner_radius=0,
        )
        polynomial_group.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        polynomial_group.grid_columnconfigure(0, weight=1)
        polynomial_label = make_label(
            polynomial_group,
            "Polynomial Order",
            size=UI["font_viewer_control_label"],
            anchor="w",
            justify="left",
        )
        polynomial_label.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        bind_responsive_label_wrap(polynomial_label, polynomial_group)
        self.calibration_poly_order_dropdown = make_combobox(
            polynomial_group,
            variable=self.calibration_poly_order_var,
            values=["1", "2", "3"],
            family=self.app.gui_font_family,
            font_size=UI["font_viewer_value"],
            dropdown_font_size=UI["font_viewer_value"],
        )
        self.calibration_poly_order_dropdown.grid(
            row=1,
            column=0,
            sticky="ew",
        )

        model_group = ctk.CTkFrame(
            top_controls,
            fg_color="transparent",
            corner_radius=0,
        )
        model_group.grid(row=0, column=1, sticky="ew", padx=8)
        model_group.grid_columnconfigure(0, weight=1)
        model_label = make_label(
            model_group,
            "Peak Model",
            size=UI["font_viewer_control_label"],
            anchor="w",
            justify="left",
        )
        model_label.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        bind_responsive_label_wrap(model_label, model_group)
        self.gaussian_model_dropdown = make_combobox(
            model_group,
            variable=self.gaussian_model_var,
            values=["Double Gaussian", "Single Gaussian"],
            family=self.app.gui_font_family,
            font_size=UI["font_viewer_value"],
            dropdown_font_size=UI["font_viewer_value"],
        )
        self.gaussian_model_dropdown.grid(
            row=1,
            column=0,
            sticky="ew",
        )

        buffer_group = ctk.CTkFrame(
            top_controls,
            fg_color="transparent",
            corner_radius=0,
        )
        buffer_group.grid(row=0, column=2, sticky="ew", padx=(8, 0))
        buffer_group.grid_columnconfigure(0, weight=1)
        buffer_label = make_label(
            buffer_group,
            "Additional Buffer (px)",
            size=UI["font_viewer_control_label"],
            anchor="w",
            justify="left",
        )
        buffer_label.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        bind_responsive_label_wrap(buffer_label, buffer_group)
        self.calibration_buffer_dropdown = make_combobox(
            buffer_group,
            variable=self.calibration_buffer_var,
            values=["10", "20", "30", "50", "75", "100", "125", "150", "200", "250", "300"],
            family=self.app.gui_font_family,
            font_size=UI["font_viewer_value"],
            dropdown_font_size=UI["font_viewer_value"],
        )
        self.calibration_buffer_dropdown.grid(
            row=1,
            column=0,
            sticky="ew",
        )

        self._build_parameter_order_controls(frame)

        calibration_note = make_label(
            frame,
            "Run diagnostics before applying a new model to an open Viewer.",
            size=UI["font_viewer_card_note"],
            text_color=SETUP_MUTED_TEXT_COLOR,
            anchor="w",
            justify="left",
        )
        calibration_note.grid(row=2, column=0, sticky="ew", pady=(5, 8))
        bind_responsive_label_wrap(calibration_note, frame)

        actions = ctk.CTkFrame(frame, fg_color="transparent", corner_radius=0)
        actions.grid(row=3, column=0, sticky="ew")
        for column in range(2):
            actions.grid_columnconfigure(column, weight=1)

        self.calibration_check_button = make_gradient_button(
            actions,
            text="Run Diagnostics",
            command=self.run_calibration_check,
            font_size=UI["font_viewer_button"],
            corner_radius=10,
            **CALIBRATION_GRADIENT_STYLE,
        )
        self.calibration_check_button.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 6),
        )

        self.open_calibration_button = make_gradient_button(
            actions,
            text="Open Diagnostics",
            command=self.open_calibration_results,
            font_size=UI["font_viewer_button"],
            corner_radius=10,
            **CALIBRATION_GRADIENT_STYLE,
        )
        self.open_calibration_button.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=6,
        )

        self.apply_calibration_button = make_gradient_button(
            actions,
            text="Apply New Model",
            command=self.apply_new_calibration_model,
            font_size=UI["font_viewer_button"],
            corner_radius=10,
            **CALIBRATION_GRADIENT_STYLE,
        )
        self.apply_calibration_button.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
            pady=(7, 0),
        )
        self.apply_calibration_button.configure_button(state="disabled")

        self._sync_calibration_plot_button_indicator()

        self.gaussian_model_var.trace_add(
            "write",
            self._update_params_poly_order_visibility,
        )
        self._update_params_poly_order_visibility()

    def _build_parameter_order_controls(self, parent) -> None:
        self.params_poly_order_frame = ctk.CTkFrame(
            parent,
            fg_color=("gray88", "gray17"),
            corner_radius=8,
        )
        self.params_poly_order_frame.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=4,
        )
        self.params_poly_order_frame.grid_columnconfigure(0, weight=1)

        heading = make_label(
            self.params_poly_order_frame,
            "Parameter Polynomial Orders",
            size=UI["font_viewer_subsection"],
            bold=True,
            anchor="w",
            justify="left",
        )
        heading.grid(row=0, column=0, sticky="ew", padx=10, pady=(8, 5))
        bind_responsive_label_wrap(
            heading,
            self.params_poly_order_frame,
            horizontal_padding=20,
        )

        controls_grid = ctk.CTkFrame(
            self.params_poly_order_frame,
            fg_color="transparent",
            corner_radius=0,
        )
        controls_grid.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 9))
        for column in range(4):
            controls_grid.grid_columnconfigure(column, weight=1, uniform="orders")

        controls = (
            ("σ₁:", self.sigma1_poly_order_var),
            ("σ₂:", self.sigma2_poly_order_var),
            ("R:", self.r_poly_order_var),
            ("δ:", self.delta_poly_order_var),
        )
        self.params_poly_order_dropdowns = []

        for index, (label_text, variable) in enumerate(controls):
            control = ctk.CTkFrame(
                controls_grid,
                fg_color="transparent",
                corner_radius=0,
            )
            control.grid(
                row=0,
                column=index,
                sticky="ew",
                padx=(0, 5) ,
                # pady=(0, 7),
            )
            control.grid_columnconfigure(0, weight=1)
            make_label(
                control,
                label_text,
                size=UI["font_viewer_control_label"],
                bold=True,
            ).grid(
                row=0,
                column=0,
                sticky="w",
                pady=(0, 4),
            )
            dropdown = make_combobox(
                control,
                variable=variable,
                values=["0", "1", "2", "3"],
                family=self.app.gui_font_family,
                font_size=UI["font_viewer_value"],
                dropdown_font_size=UI["font_viewer_value"],
                height=UI["compact_control_height"],
            )
            dropdown.grid(row=1, column=0, sticky="ew")
            self.params_poly_order_dropdowns.append(dropdown)

    def _build_metadata_section(self, parent) -> None:
        frame = self._new_section_frame(
            parent,
            name="metadata",
            title="Metadata",
            subtitle="Motor values from the selected SPEC source",
        )
        frame.grid_columnconfigure(0, weight=1)

        self.load_metadata_button = make_gradient_button(
            frame,
            text="Load Metadata",
            command=lambda: self.load_metadata(
                force_reload=True,
            ),
            font_size=UI["font_viewer_button"],
            width=170,
            bold=True,
            corner_radius=10,
            **SECONDARY_GRADIENT_STYLE,
        )
        self.load_metadata_button.grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 11),
        )

        make_label(
            frame,
            "Metadata Scan",
            size=UI["font_viewer_control_label"],
        ).grid(
            row=1,
            column=0,
            sticky="w",
            pady=(0, 4),
        )
        self.metadata_scan_dropdown = ScrollableDropdown(
            frame,
            variable=self.metadata_scan_var,
            values=[],
            max_visible_items=10,
            font_size=UI["font_viewer_value"],
            dropdown_font_size=UI["font_viewer_value"],
            command=lambda _value: self.update_metadata_motors(),
        )
        self.metadata_scan_dropdown.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(0, 9),
        )

        make_label(
            frame,
            "Motor",
            size=UI["font_viewer_control_label"],
        ).grid(
            row=3,
            column=0,
            sticky="w",
            pady=(0, 4),
        )
        self.metadata_motor_dropdown = ScrollableDropdown(
            frame,
            variable=self.metadata_motor_var,
            values=[],
            max_visible_items=10,
            font_size=UI["font_viewer_value"],
            dropdown_font_size=UI["font_viewer_value"],
            command=lambda _value: self.update_metadata_motor_value(),
        )
        self.metadata_motor_dropdown.grid(
            row=4,
            column=0,
            sticky="ew",
            pady=(0, 9),
        )

        make_label(
            frame,
            "Value",
            size=UI["font_viewer_control_label"],
        ).grid(
            row=5,
            column=0,
            sticky="w",
            pady=(0, 4),
        )
        self.metadata_value_entry = self._make_readonly_entry(
            frame,
            self.metadata_value_var,
            self.app.readonly_value_font,
        )
        self.metadata_value_entry.grid(
            row=6,
            column=0,
            sticky="ew",
            pady=(0, 9),
        )

        self.expchamber_value_entry = self._make_readonly_entry(
            frame,
            self.expchamber_value_var,
            self.app.readonly_value_bold_font,
            allow_wrap=True,
        )
        self.expchamber_value_entry.grid(
            row=7,
            column=0,
            sticky="ew",
            pady=(2, 0),
        )
        def refresh_expchamber_wrap(*_args) -> None:
            refresh_wrap = getattr(
                self.expchamber_value_entry,
                "_responsive_wrap_update",
                None,
            )
            if callable(refresh_wrap):
                refresh_wrap()
            self.after_idle(self._sync_setup_scroll_height)

        self.expchamber_value_var.trace_add(
            "write",
            refresh_expchamber_wrap,
        )




    @staticmethod
    def _make_readonly_entry(
        parent,
        variable: ctk.StringVar,
        font: ctk.CTkFont,
        *,
        allow_wrap: bool = False,
    ) -> ctk.CTkLabel:
        label = ctk.CTkLabel(
            parent,
            textvariable=variable,
            fg_color=("#F2F2F2", "#212121"),
            text_color=("#000000", "#FFFFFF"),
            corner_radius=7,
            height=54 if allow_wrap else UI["button_height"],
            justify="left",
            anchor="w",
            font=font,
        )
        if allow_wrap:
            bind_responsive_label_wrap(
                label,
                parent,
                horizontal_padding=18,
            )
        return label

    def _add_directory_picker(
        self,
        parent,
        *,
        row: int,
        label_column: int,
        field_column: int,
        label: str,
        variable: ctk.StringVar,
        choose_command: Callable[[], None],
        placeholder: str,
        field_padx=(0, 0),
    ) -> ctk.CTkEntry:
        make_label(
            parent,
            label,
            size=UI["font_viewer_control_label"],
        ).grid(
            row=row,
            column=label_column,
            sticky="w",
            padx=(0, 10),
            pady=5,
        )

        # Entry and browse button share one visual frame.
        picker_frame = ctk.CTkFrame(
            parent,
            height=UI["button_height"],
            fg_color=("#F9F9FA", "#343638"),
            border_color=("#979DA2", "#565B5E"),
            border_width=1,
            corner_radius=7,
        )
        picker_frame.grid(
            row=row,
            column=field_column,
            sticky="ew",
            padx=field_padx,
            pady=5,
        )
        picker_frame.grid_columnconfigure(0, weight=1)
        picker_frame.grid_propagate(False)

        entry = ctk.CTkEntry(
            picker_frame,
            textvariable=variable,
            placeholder_text=placeholder,
            placeholder_text_color="gray55",
            height=UI["button_height"] - 2,
            fg_color="transparent",
            border_width=0,
            corner_radius=0,
            font=ui_font(
                UI["font_viewer_value"],
                family=self.app.gui_font_family,
            ),
        )
        entry.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(8, 2),
            pady=1,
        )

        make_gradient_button(
            picker_frame,
            text="Browse...",
            command=choose_command,
            width=120,
            height=UI["button_height"] - 4,
            font_size=UI["font_viewer_button"],
            corner_radius=5,
            **SUBTLE_GRADIENT_STYLE,
        ).grid(
            row=0,
            column=1,
            padx=2,
            pady=2,
        )

        return entry

    def _build_save_section(self, parent) -> None:
        frame = self._new_section_frame(
            parent,
            name="output",
            title="Output",
            subtitle="Default destinations used when files are saved directly",
        )

        frame.grid_columnconfigure(1, weight=1)

        self.spectra_dir_entry = self._add_directory_picker(
            frame,
            row=0,
            label_column=0,
            field_column=1,
            label="Spectra",
            variable=self.spectra_dir_var,
            choose_command=self.choose_spectra_dir,
            placeholder="Default spectra folder",
        )

        self.histogram_dir_entry = self._add_directory_picker(
            frame,
            row=1,
            label_column=0,
            field_column=1,
            label="Histograms",
            variable=self.histogram_dir_var,
            choose_command=self.choose_histogram_dir,
            placeholder="Default histograms folder",
        )

    def _build_log_section(self, parent) -> None:
        frame = self._new_section_frame(
            parent,
            name="log",
            title="Activity",
            subtitle="Metadata, calibration and Viewer status messages",
        )

        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(0, weight=1)

        self.status_box = ctk.CTkTextbox(
            frame,
            height=max(110, UI["status_height"] // 2),
            font=ui_font(UI["font_viewer_status"]),
            fg_color=SETUP_CONTROL_COLOR,
            border_color=SETUP_BORDER_COLOR,
            border_width=1,
            corner_radius=8,
        )
        self.status_box.grid(
            row=0,
            column=0,
            sticky="nsew",
        )

        self.status_box.insert(
            "end",
            "meV-RIXS Toolkit\n",
            # f"Launcher: {Path(__file__).name}\n",
        )
        self.status_box.configure(state="disabled")

    # Calibration options, caches, and viewer synchronization

    def _update_params_poly_order_visibility(self, *_args) -> None:
        if self.gaussian_model_var.get() == "Double Gaussian":
            self.params_poly_order_frame.grid()
        else:
            self.params_poly_order_frame.grid_remove()

    def _get_params_poly_orders(self) -> list[int]:
        return [
            int(self.sigma1_poly_order_var.get()),
            int(self.sigma2_poly_order_var.get()),
            int(self.r_poly_order_var.get()),
            int(self.delta_poly_order_var.get()),
        ]

    def _get_calibration_options(self) -> CalibrationOptions:
        double_gaussian = self.gaussian_model_var.get() == "Double Gaussian"
        try:
            peak_buffer = int(self.calibration_buffer_var.get())
        except ValueError as exc:
            raise ValueError(
                "Peak Selection Buffer must be a non-negative integer."
            ) from exc
        if peak_buffer < 0:
            raise ValueError(
                "Peak Selection Buffer must be a non-negative integer."
            )
        return CalibrationOptions(
            fit_poly_order=int(self.calibration_poly_order_var.get()),
            double_gaussian_model=double_gaussian,
            params_poly_orders=(
                self._get_params_poly_orders() if double_gaussian else None
            ),
            peak_buffer=peak_buffer,
        )

    def _get_valid_search_roots(self) -> list[Path]:
        roots = get_scan_search_roots(
            self.search_root_var.get(),
            max_parent_levels=3,
        )
        for root in roots:
            validate_directory(root, "Scan Files Search Root")
        return roots

    def _ensure_scan_file_cache(
        self,
        scans: np.ndarray,
        search_roots: list[Path],
    ) -> None:
        requested_scans = [
            int(scan)
            for scan in np.asarray(scans, dtype=int)
        ]

        # Remove stale entries.
        for scan in requested_scans:
            cached_pair = self._scan_file_cache.get(scan)

            if cached_pair is None:
                continue

            x_file, y_file = cached_pair

            if not x_file.is_file() or not y_file.is_file():
                self._scan_file_cache.pop(scan, None)

        missing_scans = [
            scan
            for scan in requested_scans
            if scan not in self._scan_file_cache
        ]

        if not missing_scans:
            return

        viewer_module = mev_viewer

        resolved = viewer_module.resolve_scan_file_map(
            np.asarray(missing_scans, dtype=int),
            search_roots=search_roots,
        )

        self._scan_file_cache.update(resolved)


    def _build_calibration_kwargs(
        self,
        calibration_scans: np.ndarray,
        search_roots: list[Path],
        options: CalibrationOptions,
        *,
        show_calibration_plots: bool,
        show_test_plot: bool,
    ) -> dict:
        calibration_scans = np.asarray(
            calibration_scans,
            dtype=int,
        )

        self._ensure_scan_file_cache(
            calibration_scans,
            search_roots,
        )

        self._ensure_spec_metadata(
            calibration_scans,
        )

        missing_energy_scans = [
            int(scan)
            for scan in calibration_scans
            if (
                int(scan) not in self._metadata_cache
                or "pgm_en"
                not in self._metadata_cache[int(scan)]
            )
        ]

        if missing_energy_scans:
            raise ValueError(
                "Could not determine calibration energies for scans: "
                f"{missing_energy_scans}. Missing motor value: pgm_en"
            )

        kwargs = {
            "calibration_scans": calibration_scans,
            "search_roots": search_roots,
            "motor_name": "pgm_en",
            "scan_files_by_scan": self._scan_file_cache,
            "motor_values_by_scan": self._metadata_cache,
            "buffer": options.peak_buffer,
            "fit_poly_order": options.fit_poly_order,
            "double_gaussian_model": options.double_gaussian_model,
            "show_calibration_plots": show_calibration_plots,
            "show_test_plot": show_test_plot,
        }

        if options.params_poly_orders is not None:
            kwargs["params_poly_orders"] = (
                options.params_poly_orders
            )

        return kwargs

    @staticmethod
    def _cached_energy_coefficients(
        cache: dict[str, object],
    ) -> tuple[float, float, float, float]:
        values = cache.get("energy_coefficients")
        if not isinstance(values, (list, tuple)) or len(values) != 4:
            raise ValueError(
                "The stored calibration does not contain four valid "
                "energy coefficients."
            )
        converted = tuple(float(value) for value in values)
        return converted[0], converted[1], converted[2], converted[3]

    @staticmethod
    def _calibration_cache_matches(
        cache: object,
        calibration_scans: np.ndarray,
        options: CalibrationOptions,
    ) -> bool:
        if not isinstance(cache, dict):
            return False

        expected_model = (
            "double_gaussian"
            if options.double_gaussian_model
            else "single_gaussian"
        )
        expected_orders = (
            {
                "sigma1": int(options.params_poly_orders[0]),
                "sigma2": int(options.params_poly_orders[1]),
                "R": int(options.params_poly_orders[2]),
                "delta": int(options.params_poly_orders[3]),
            }
            if options.params_poly_orders is not None
            else {"sigma": 2}
        )

        try:
            cached_scans = [int(scan) for scan in cache["calibration_scans"]]
            SpectraTab._cached_energy_coefficients(cache)
            cached_orders = {
                str(name): int(order)
                for name, order in cache[
                    "line_shape_polynomial_orders"
                ].items()
            }
            cached_buffer = int(cache.get("peak_selection_buffer", -1))
        except (KeyError, TypeError, ValueError, AttributeError):
            return False

        return (
            cached_scans == [int(scan) for scan in calibration_scans]
            and int(cache.get("energy_polynomial_order", -1))
            == options.fit_poly_order
            and cache.get("line_shape_model") == expected_model
            and cached_orders == expected_orders
            and cached_buffer == options.peak_buffer
        )

    def _collect_viewer_options(self) -> ViewerOptions:
        scans = parse_scans(self.scans_entry.get())
        calibration_text = self.calibration_scans_entry.get().strip()
        calibration_scans = parse_scans(calibration_text) if calibration_text else None
        search_roots = self._get_valid_search_roots()

        spectra_dir = parse_optional_path(self.spectra_dir_var.get())
        histogram_dir = parse_optional_path(self.histogram_dir_var.get())
        validate_directory(spectra_dir, "Spectra Save Dir.")
        validate_directory(histogram_dir, "Histogram Save Dir.")

        return ViewerOptions(
            scans=scans,
            calibration_scans=calibration_scans,
            search_roots=search_roots,
            spectra_dir=spectra_dir,
            histogram_dir=histogram_dir,
            calibration=self._get_calibration_options(),
            dataset_name=self.app.get_tab_name_for_tab(self),
        )

    def _install_live_setup_sync(self) -> None:
        """Connect Setup values that can update an already open Viewer."""

        self.spectra_dir_var.trace_add(
            "write",
            self._sync_open_viewer_save_directories,
        )
        self.histogram_dir_var.trace_add(
            "write",
            self._sync_open_viewer_save_directories,
        )

        calibration_variables = (
            self.calibration_poly_order_var,
            self.gaussian_model_var,
            self.calibration_buffer_var,
            self.sigma1_poly_order_var,
            self.sigma2_poly_order_var,
            self.r_poly_order_var,
            self.delta_poly_order_var,
        )
        for variable in calibration_variables:
            variable.trace_add(
                "write",
                self._invalidate_pending_calibration_model,
            )

        self.calibration_scans_entry.bind(
            "<KeyRelease>",
            self._invalidate_pending_calibration_model,
            add="+",  # type: ignore
        )

    def _get_open_viewer_keepalive(self) -> dict | None:
        if self.viewer_figure is None:
            return None

        keepalive = getattr(
            self.viewer_figure,
            "_view_spectra_keepalive",
            None,
        )
        return keepalive if isinstance(keepalive, dict) else None

    def _sync_open_viewer_save_directories(self, *_args) -> None:
        """Mirror current Setup save paths into a running Viewer."""

        keepalive = self._get_open_viewer_keepalive()
        if keepalive is None:
            return

        setter = keepalive.get("set_save_directories")
        if not callable(setter):
            return

        setter(
            parse_optional_path(self.spectra_dir_var.get()),
            parse_optional_path(self.histogram_dir_var.get()),
        )

    @staticmethod
    def _calibration_model_signature(cache: object) -> tuple | None:
        """Return the Setup-level identity of a computed calibration model."""

        if not isinstance(cache, dict):
            return None

        try:
            calibration_scans = tuple(
                int(scan) for scan in cache["calibration_scans"]
            )
            polynomial_order = int(cache["energy_polynomial_order"])
            line_shape_model = str(cache["line_shape_model"])
            peak_buffer = int(cache.get("peak_selection_buffer", -1))
            raw_orders = cache["line_shape_polynomial_orders"]
            if not isinstance(raw_orders, dict):
                return None
            parameter_orders = tuple(
                sorted(
                    (str(name), int(order))
                    for name, order in raw_orders.items()
                )
            )
        except (KeyError, TypeError, ValueError):
            return None

        return (
            calibration_scans,
            polynomial_order,
            line_shape_model,
            parameter_orders,
            peak_buffer,
        )

    def _invalidate_pending_calibration_model(self, *_args) -> None:
        """Require a fresh diagnostic run after any calibration input change."""

        self._pending_calibration_for_viewer = None
        self._sync_apply_calibration_button_state()

    def _sync_apply_calibration_button_state(self) -> None:
        pending_signature = self._calibration_model_signature(
            self._pending_calibration_for_viewer
        )
        applied_signature = self._calibration_model_signature(
            self._viewer_applied_calibration
        )
        can_apply = (
            self.viewer_figure is not None
            and pending_signature is not None
            and pending_signature != applied_signature
        )
        buttons = (
            getattr(self, "apply_calibration_button", None),
            getattr(self, "diagnostics_apply_calibration_button", None),
        )
        for button in buttons:
            if button is not None:
                button.configure_button(
                    state="normal" if can_apply else "disabled",
                    text="Apply New Model",
                )

    def _record_open_viewer_calibration(self) -> None:
        """Remember which calibration model the current Viewer actually uses."""

        self._viewer_applied_calibration = None
        keepalive = self._get_open_viewer_keepalive()
        if keepalive is not None:
            getter = keepalive.get("get_calibration_state")
            if callable(getter):
                state = getter()
                if isinstance(state, dict):
                    metadata = state.get("calibration_metadata")
                    if isinstance(metadata, dict):
                        self._viewer_applied_calibration = copy.deepcopy(metadata)

        if (
            self._viewer_applied_calibration is None
            and isinstance(self._calibration_cache, dict)
        ):
            self._viewer_applied_calibration = copy.deepcopy(
                self._calibration_cache
            )

        self._pending_calibration_for_viewer = None
        self._sync_apply_calibration_button_state()

    def apply_new_calibration_model(self) -> None:
        """Transfer the latest successful diagnostic calibration to the Viewer."""

        pending = self._pending_calibration_for_viewer
        keepalive = self._get_open_viewer_keepalive()
        if not isinstance(pending, dict) or keepalive is None:
            self._sync_apply_calibration_button_state()
            return

        apply_model = keepalive.get("apply_calibration_model")
        get_state = keepalive.get("get_calibration_state")
        if not callable(apply_model) or not callable(get_state):
            self.app.show_styled_message(
                title="Calibration update error",
                message=(
                    "The open Viewer does not support live calibration updates. "
                    "Please restart it once with the updated mev_viewer.py."
                ),
                kind="error",
            )
            return

        try:
            coefficients = self._cached_energy_coefficients(pending)
            current_state = get_state()
            if not isinstance(current_state, dict):
                raise TypeError("The Viewer returned an invalid calibration state.")

            incident_energy_value = current_state.get("incident_energy")
            incident_energy = (
                None
                if incident_energy_value is None
                else float(incident_energy_value)
            )

            applied_details = copy.deepcopy(pending)
            current_metadata = current_state.get("calibration_metadata")
            if isinstance(current_metadata, dict):
                incident_metadata = current_metadata.get("incident_energy")
                if isinstance(incident_metadata, dict):
                    applied_details["incident_energy"] = copy.deepcopy(
                        incident_metadata
                    )

            apply_model(
                coefficients,
                incident_energy,
                applied_details,
            )
        except Exception as exc:
            self.app.show_styled_message(
                title="Calibration update error",
                message=str(exc),
                kind="error",
            )
            self.log(f"Calibration update error: {exc}")
            return

        self._calibration_cache = copy.deepcopy(applied_details)
        self._viewer_applied_calibration = copy.deepcopy(applied_details)
        self._pending_calibration_for_viewer = None
        self._sync_apply_calibration_button_state()

        self.log(
            "Applied new calibration model to the running Viewer: "
            + ", ".join(f"{value:.8g}" for value in coefficients)
        )

    def export_session_state(self) -> dict:
        viewer_state = None
        if self.viewer_figure is not None:
            keepalive = getattr(
                self.viewer_figure,
                "_view_spectra_keepalive",
                None,
            )
            if isinstance(keepalive, dict):
                exporter = keepalive.get("export_session_state")
                if callable(exporter):
                    viewer_state = exporter()

        return {
            "setup": {
                "scans": self.scans_entry.get(),
                "calibration_scans": self.calibration_scans_entry.get(),
                "search_root": self.search_root_var.get(),
                "spec_file": self.spec_file_var.get(),
                "spec_search_root": self.spec_search_root_var.get(),
                "calibration_poly_order": self.calibration_poly_order_var.get(),
                "gaussian_model": self.gaussian_model_var.get(),
                "calibration_buffer": self.calibration_buffer_var.get(),
                "sigma1_poly_order": self.sigma1_poly_order_var.get(),
                "sigma2_poly_order": self.sigma2_poly_order_var.get(),
                "r_poly_order": self.r_poly_order_var.get(),
                "delta_poly_order": self.delta_poly_order_var.get(),
                "spectra_dir": self.spectra_dir_var.get(),
                "histogram_dir": self.histogram_dir_var.get(),
                "apply_global_preferences": bool(
                    self.apply_global_preferences_var.get()
                ),
                "metadata_scan": self.metadata_scan_var.get(),
                "metadata_motor": self.metadata_motor_var.get(),
                "metadata_value": self.metadata_value_var.get(),
                "expchamber_value": self.expchamber_value_var.get(),
            },
            "viewer_open": viewer_state is not None,
            "viewer": viewer_state,
            "calibration_cache": self._calibration_cache,
            "active_content_tab": self.content_tabs.get(),
        }

    @staticmethod
    def _set_entry_text(entry: ctk.CTkEntry, value: object) -> None:
        entry.delete(0, "end")
        if value is not None:
            entry.insert(0, str(value))

    def restore_setup_session_state(self, state: dict) -> None:
        setup = state.get("setup", {}) if isinstance(state, dict) else {}
        if not isinstance(setup, dict):
            setup = {}

        self._set_entry_text(self.scans_entry, setup.get("scans", ""))
        self._set_entry_text(
            self.calibration_scans_entry,
            setup.get("calibration_scans", ""),
        )

        variable_values = (
            (self.search_root_var, "search_root", ""),
            (self.spec_file_var, "spec_file", ""),
            (self.spec_search_root_var, "spec_search_root", ""),
            (self.calibration_poly_order_var, "calibration_poly_order", "3"),
            (self.gaussian_model_var, "gaussian_model", "Double Gaussian"),
            (self.calibration_buffer_var, "calibration_buffer", "0"),
            (self.sigma1_poly_order_var, "sigma1_poly_order", "2"),
            (self.sigma2_poly_order_var, "sigma2_poly_order", "2"),
            (self.r_poly_order_var, "r_poly_order", "2"),
            (self.delta_poly_order_var, "delta_poly_order", "0"),
            (self.spectra_dir_var, "spectra_dir", ""),
            (self.histogram_dir_var, "histogram_dir", ""),
            (self.metadata_scan_var, "metadata_scan", ""),
            (self.metadata_motor_var, "metadata_motor", ""),
            (self.metadata_value_var, "metadata_value", "—"),
            (
                self.expchamber_value_var,
                "expchamber_value",
                EXPCHAMBER_EMPTY_TEXT,
            ),
        )
        for variable, key, default in variable_values:
            variable.set(str(setup.get(key, default)))
        self.apply_global_preferences_var.set(
            bool(setup.get("apply_global_preferences", False))
        )

        calibration_cache = state.get("calibration_cache")
        self._calibration_cache = (
            calibration_cache if isinstance(calibration_cache, dict) else None
        )
        self._update_params_poly_order_visibility()
        # Setup is always visible in the unified workspace.  Returning to the
        # Viewer keeps restored sessions compatible with the previous layout.
        self.content_tabs.set("Viewer")

    def restore_session_metadata(self) -> None:
        """Load metadata when a saved explicit SPEC source is available."""

        spec_source = (
            parse_optional_path(self.spec_file_var.get())
            or parse_optional_path(self.spec_search_root_var.get())
        )
        if spec_source is None:
            return
        spec_source = spec_source.expanduser()
        if not spec_source.exists():
            self.log(f"Saved SPEC source is unavailable: {spec_source}")
            return

        self.refresh_metadata_scan_choices(auto_load=False)
        self.load_metadata(
            force_reload=True,
            refresh_scan_choices=False,
        )

    def restore_calibration_plots_from_session(self) -> None:
        """Restore cached calibration diagnostics, refitting only legacy caches."""

        if not isinstance(self._calibration_cache, dict):
            return

        calibration_text = self.calibration_scans_entry.get().strip()
        if not calibration_text:
            return

        viewer_module = mev_viewer
        builder = getattr(
            viewer_module,
            "build_calibration_figures_from_details",
            None,
        )
        if callable(builder):
            try:
                with plt.rc_context():
                    cached_figures = builder(self._calibration_cache)
                if isinstance(cached_figures, dict):
                    self._embed_calibration_figures(
                        cached_figures,
                        open_results=False,
                    )
                    self.log("Restored calibration diagnostic plots from session data.")
                    return
            except (KeyError, TypeError, ValueError) as exc:
                self.log(f"Calibration plot cache requires migration: {exc}")

        calibration_scans = parse_scans(calibration_text)
        options = self._get_calibration_options()
        if not self._calibration_cache_matches(
            self._calibration_cache,
            calibration_scans,
            options,
        ):
            return

        search_roots = self._get_valid_search_roots()

        kwargs = self._build_calibration_kwargs(
            calibration_scans,
            search_roots,
            options,
            show_calibration_plots=False,
            show_test_plot=False,
        )
        kwargs["return_details"] = True
        kwargs["return_figures"] = True
        previous_incident_energy = self._calibration_cache.get("incident_energy")
        with plt.rc_context():
            calibration_output = viewer_module.compute_energy_calibration_2(**kwargs)
        calibration_details, calibration_figures = _unpack_calibration_output(
            calibration_output
        )
        if previous_incident_energy is not None:
            calibration_details["incident_energy"] = previous_incident_energy
        self._calibration_cache = calibration_details
        self._embed_calibration_figures(
            calibration_figures,
            open_results=False,
        )
        self.log(
            "Migrated legacy calibration diagnostics; future sessions can "
            "restore them without refitting."
        )

    def start_viewer_from_session(self, state: dict) -> None:
        viewer_state = state.get("viewer") if isinstance(state, dict) else None
        if not bool(state.get("viewer_open", False)) or not isinstance(
            viewer_state,
            dict,
        ):
            return

        self._pending_viewer_session_state = viewer_state
        requested_tab = str(state.get("active_content_tab", "Viewer"))
        legacy_tab_names = {
            "Setup": "Viewer",
            "Calibration": "Diagnostics",
        }
        requested_tab = legacy_tab_names.get(requested_tab, requested_tab)
        self._pending_content_tab = (
            requested_tab
            if requested_tab in ("Viewer", "Diagnostics")
            else "Viewer"
        )
        self.run_viewer()

    def log(self, text: str) -> None:
        self.status_box.configure(state="normal")
        self.status_box.insert("end", text.rstrip() + "\n")
        self.status_box.see("end")
        self.status_box.configure(state="disabled")
        self.update_idletasks()

    @staticmethod
    def _add_to_combo_history(
        *,
        combo: ctk.CTkComboBox,
        history: list[str],
        path_str: str,
    ) -> None:
        if path_str in history:
            history.remove(path_str)
        history.insert(0, path_str)
        del history[20:]
        combo.configure(values=history)

    # Directory selection and SPEC metadata

    def get_spec_search_candidates(self) -> list[Path]:
        """
        Return ordered roots for automatic SPEC discovery.

        A directly selected SPEC file has priority and bypasses directory
        discovery. Otherwise an explicit SPEC Search Root has priority,
        followed by the roots used for scan-file discovery.
        """

        explicit_spec_file = parse_optional_path(
            self.spec_file_var.get()
        )

        if explicit_spec_file is not None:
            return [explicit_spec_file.expanduser().resolve()]

        explicit_spec_root = parse_optional_path(
            self.spec_search_root_var.get()
        )

        if explicit_spec_root is not None:
            return [
                explicit_spec_root.expanduser().resolve()
            ]

        return [
            root.expanduser().resolve()
            for root in get_scan_search_roots(
                self.search_root_var.get(),
                max_parent_levels=3,
            )
        ]



    def _choose_directory(
        self,
        *,
        title: str,
        variable: ctk.StringVar,
        combo: ctk.CTkComboBox | None = None,
        history: list[str] | None = None,
    ) -> None:
        directory = filedialog.askdirectory(title=title, parent=self)
        if not directory:
            return
        variable.set(directory)
        if combo is not None and history is not None:
            self._add_to_combo_history(
                combo=combo,
                history=history,
                path_str=directory,
            )

    def _on_spec_search_root_committed(
        self,
        _event=None,
    ) -> None:
        """Load metadata after an explicit SPEC root was entered."""

        if parse_optional_path(
            self.spec_search_root_var.get()
        ) is None:
            return

        self.after_idle(
            lambda: self.load_metadata(
                force_reload=True,
            )
        )

    def _on_spec_file_committed(self, _event=None) -> None:
        """Load metadata after a SPEC file path was entered manually."""

        if parse_optional_path(self.spec_file_var.get()) is None:
            return

        self.after_idle(
            lambda: self.load_metadata(force_reload=True)
        )

    def choose_search_root(self) -> None:
        self._choose_directory(
            title="Choose Scan Files Search Root",
            variable=self.search_root_var,
            combo=self.search_root_combo,
            history=self.app.search_root_history,
        )

    def choose_spec_file(self) -> None:
        """Choose one SPEC file and use it instead of directory discovery."""

        initial_dir: str | None = None
        initial_file: str | None = None

        current_file = parse_optional_path(self.spec_file_var.get())
        if current_file is not None:
            current_file = current_file.expanduser()
            if current_file.parent.is_dir():
                initial_dir = str(current_file.parent)
                initial_file = current_file.name
        else:
            spec_root = parse_optional_path(self.spec_search_root_var.get())
            if spec_root is not None and spec_root.expanduser().is_dir():
                initial_dir = str(spec_root.expanduser())

        selected_file = filedialog.askopenfilename(
            parent=self,
            title="Choose Spec File",
            filetypes=(
                ("SPEC files", "*.spec"),
                ("All files", "*.*"),
            ),
            initialdir=initial_dir,
            initialfile=initial_file,
        )
        if not selected_file:
            return

        self.spec_file_var.set(selected_file)
        self.load_metadata(force_reload=True)

    def choose_spec_search_root(self) -> None:
        previous_value = (
            self.spec_search_root_var.get().strip()
        )

        self._choose_directory(
            title="Choose Spec File Search Root",
            variable=self.spec_search_root_var,
            combo=self.spec_search_root_combo,
            history=self.app.spec_search_root_history,
        )

        new_value = self.spec_search_root_var.get().strip()

        if new_value and new_value != previous_value:
            self.load_metadata(
                force_reload=True,
            )

    def choose_spectra_dir(self) -> None:
        self._choose_directory(
            title="Choose spectra_dir",
            variable=self.spectra_dir_var,
        )

    def choose_histogram_dir(self) -> None:
        self._choose_directory(
            title="Choose histogram_dir",
            variable=self.histogram_dir_var,
        )

    @staticmethod
    def _widget_is_inside(widget, container) -> bool:
        current_widget = widget
        while current_widget is not None:
            if current_widget == container:
                return True
            current_widget = getattr(current_widget, "master", None)
        return False

    def _on_scan_input_focus_out(self, _event=None) -> None:
        self.after_idle(self._refresh_metadata_after_focus_change)

    def _refresh_metadata_after_focus_change(self) -> None:
        focused_widget = self.focus_get()
        if self._widget_is_inside(focused_widget, self.scans_entry):
            return
        if self._widget_is_inside(focused_widget, self.calibration_scans_entry):
            return
        self.refresh_metadata_scan_choices()

    def _reset_metadata(self) -> None:
        self.metadata_motor_dropdown.set_values([])
        self.metadata_value_var.set("—")
        self.expchamber_value_var.set(EXPCHAMBER_EMPTY_TEXT)

    def _collect_metadata_scan_numbers(
        self,
    ) -> np.ndarray:
        """Collect unique scans from both scan input fields."""

        scan_numbers: list[int] = []

        for text in (
            self.scans_entry.get().strip(),
            self.calibration_scans_entry.get().strip(),
        ):
            if not text:
                continue

            try:
                parsed_scans = parse_scans(text)
            except Exception:
                continue

            for scan in parsed_scans:
                scan_num = int(scan)

                if scan_num not in scan_numbers:
                    scan_numbers.append(scan_num)

        return np.asarray(
            scan_numbers,
            dtype=int,
        )

    def _ensure_spec_metadata(
        self,
        scans: np.ndarray,
    ) -> set[int]:
        """
        Ensure that SPEC metadata for the requested scans is available in memory.

        Returns the scan numbers for which no metadata could be found.
        """

        requested_scans = {
            int(scan)
            for scan in np.asarray(scans, dtype=int)
        }

        if not requested_scans:
            return set()

        missing_scans = requested_scans - set(
            self._metadata_cache
        )

        if not missing_scans:
            return set()

        viewer_module = mev_viewer

        if not hasattr(
            viewer_module,
            "get_motor_values_for_scans",
        ):
            raise AttributeError(
                "The viewer module does not contain "
                "'get_motor_values_for_scans'."
            )

        for candidate in self.get_spec_search_candidates():
            if not missing_scans:
                break

            if not candidate.exists():
                continue

            if candidate.is_file():
                if candidate.suffix.lower() != ".spec":
                    continue
            elif not candidate.is_dir():
                continue

            try:
                loaded_values = (
                    viewer_module.get_motor_values_for_scans(
                        np.asarray(
                            sorted(missing_scans),
                            dtype=int,
                        ),
                        spec_dir=candidate,
                        source_files=self._spec_file_cache,
                    )
                )
            except (
                FileNotFoundError,
                OSError,
                ValueError,
            ):
                continue

            if not loaded_values:
                continue

            for scan_num, motor_values in loaded_values.items():
                self._metadata_cache[int(scan_num)] = motor_values

            missing_scans.difference_update(
                int(scan_num)
                for scan_num in loaded_values
            )

        return missing_scans



    def load_metadata(
        self,
        *,
        force_reload: bool = True,
        refresh_scan_choices: bool = True,
    ) -> None:
        """
        Load metadata into memory.

        When SPEC Search Root is empty, search the same roots used
        for scan-file discovery. No modal error dialog is displayed.
        """

        if refresh_scan_choices:
            self.refresh_metadata_scan_choices(
                auto_load=False,
            )

        scan_numbers = self._collect_metadata_scan_numbers()

        if scan_numbers.size == 0:
            self._reset_metadata()
            self.metadata_value_var.set(
                "Enter scans before loading metadata."
            )
            return

        if force_reload:
            self._metadata_cache.clear()
            self._spec_file_cache.clear()

        scans_to_load = np.asarray(
            [
                int(scan)
                for scan in scan_numbers
                if int(scan) not in self._metadata_cache
            ],
            dtype=int,
        )

        if scans_to_load.size == 0:
            self.update_metadata_motors()
            return

        self.load_metadata_button.configure_button(
            state="disabled",
            text="Loading...",
        )

        loading_token = self._begin_local_loading(
            self._setup_cards["metadata"],
            "Loading metadata…",
        )

        try:
            missing_scans = self._ensure_spec_metadata(
                scans_to_load
            )

            loaded_scans = [
                int(scan)
                for scan in scans_to_load
                if int(scan) in self._metadata_cache
            ]

            if loaded_scans:
                spec_files = {
                    self._spec_file_cache[scan]
                    for scan in loaded_scans
                    if scan in self._spec_file_cache
                }

                if spec_files:
                    self.log("Loaded SPEC metadata from:")

                    for file_path in sorted(
                        spec_files,
                        key=str,
                    ):
                        self.log(f"  {file_path}")

            if missing_scans:
                missing_text = ", ".join(
                    str(scan)
                    for scan in sorted(missing_scans)
                )

                self.log(
                    "No SPEC metadata found for scans: "
                    f"{missing_text}"
                )

            if not self._metadata_cache:
                self.metadata_motor_dropdown.set_values([])
                self.metadata_value_var.set(
                    "No matching SPEC metadata found."
                )
                self.expchamber_value_var.set(
                    EXPCHAMBER_EMPTY_TEXT
                )
                return

            self.update_metadata_motors()

        except Exception as exc:
            self.metadata_motor_dropdown.set_values([])
            self.metadata_value_var.set(
                "Metadata could not be loaded."
            )
            self.expchamber_value_var.set(
                EXPCHAMBER_EMPTY_TEXT
            )
            self.log(
                f"Metadata loading error: {exc}"
            )

        finally:
            self.load_metadata_button.configure_button(
                state="normal",
                text="Load Metadata",
            )
            self._end_local_loading(loading_token)

    def refresh_metadata_scan_choices(
        self,
        *,
        auto_load: bool = True,
    ) -> None:
        scan_values: list[str] = []
        has_incomplete_input = False

        for text in (
            self.scans_entry.get().strip(),
            self.calibration_scans_entry.get().strip(),
        ):
            if not text:
                continue
            try:
                parsed_scans = parse_scans(text)
            except Exception:
                has_incomplete_input = True
                continue

            for scan in parsed_scans:
                scan_text = str(int(scan))
                if scan_text not in scan_values:
                    scan_values.append(scan_text)

        if not scan_values and has_incomplete_input:
            return

        current_scan = self.metadata_scan_var.get()
        preferred_scan = (
            current_scan
            if current_scan in scan_values
            else scan_values[0] if scan_values else None
        )
        self.metadata_scan_dropdown.set_values(
            scan_values,
            preferred_value=preferred_scan,
        )

        if not scan_values:
            self._reset_metadata()
            return

        explicit_spec_source = (
            parse_optional_path(self.spec_file_var.get())
            or parse_optional_path(self.spec_search_root_var.get())
        )

        if auto_load and explicit_spec_source is not None:
            self.load_metadata(
                force_reload=False,
                refresh_scan_choices=False,
            )
            return

        # Never start automatic parent-directory discovery merely
        # because a scan input field lost focus.
        self.update_metadata_motors()

    def update_metadata_motors(self) -> None:
        """Update the motor dropdown exclusively from the cache."""

        scan_text = self.metadata_scan_var.get().strip()

        if not scan_text:
            self._reset_metadata()
            return

        scan_num = int(scan_text)
        motor_values = self._metadata_cache.get(scan_num)

        if motor_values is None:
            self.metadata_motor_dropdown.set_values([])
            self.metadata_value_var.set(
                "Click 'Load Metadata' to load this scan."
            )
            self.expchamber_value_var.set(
                EXPCHAMBER_EMPTY_TEXT
            )
            return

        motor_names = sorted(motor_values)

        if not motor_names:
            self.metadata_motor_dropdown.set_values([])
            self.metadata_value_var.set(
                "No motors found for this scan."
            )
            self.expchamber_value_var.set(
                EXPCHAMBER_EMPTY_TEXT
            )
            return

        preferred_motor = (
            self.metadata_motor_var.get()
            if self.metadata_motor_var.get() in motor_names
            else (
                "pgm_en_setpoint"
                if "pgm_en_setpoint" in motor_names
                else (
                    "pgm_en"
                    if "pgm_en" in motor_names
                    else motor_names[0]
                )
            )
        )

        self.metadata_motor_dropdown.set_values(
            motor_names,
            preferred_value=preferred_motor,
        )

        self.update_metadata_motor_value()
        self.update_expchamber_values()

    def update_metadata_motor_value(self) -> None:
        """Display one motor value from the metadata cache."""

        scan_text = self.metadata_scan_var.get().strip()
        motor_name = self.metadata_motor_var.get().strip()

        if not scan_text or not motor_name:
            self.metadata_value_var.set("—")
            return

        motor_values = self._metadata_cache.get(
            int(scan_text)
        )

        if motor_values is None:
            self.metadata_value_var.set(
                "Metadata not loaded."
            )
            return

        value = motor_values.get(motor_name)

        self.metadata_value_var.set(
            "No value found."
            if value is None
            else str(value)
        )

    def update_expchamber_values(self) -> None:
        """Display Expchamber values from the metadata cache."""

        scan_text = self.metadata_scan_var.get().strip()

        if not scan_text:
            self.expchamber_value_var.set(
                EXPCHAMBER_EMPTY_TEXT
            )
            return

        motor_values = self._metadata_cache.get(
            int(scan_text)
        )

        if motor_values is None:
            self.expchamber_value_var.set(
                EXPCHAMBER_EMPTY_TEXT
            )
            return

        motor_formats = {
            "x": ("expchamber_x", ".3f"),
            "y": ("expchamber_y", ".3f"),
            "z": ("expchamber_z", ".3f"),
            "r": ("expchamber_r", ".1f"),
            "offset": ("expchamber_r_offset", ".1f"),
        }

        values: dict[str, str] = {}

        for key, (
            motor_name,
            number_format,
        ) in motor_formats.items():
            value = motor_values.get(motor_name)

            if value is None:
                values[key] = "—"
                continue

            try:
                values[key] = format(
                    float(value),
                    number_format,
                )
            except (TypeError, ValueError):
                values[key] = str(value)

        self.expchamber_value_var.set(
            "Expchamber:   "
            f"x = {values['x']}    "
            f"y = {values['y']}    "
            f"z = {values['z']}    "
            f"r = {values['r']} "
            f"(offset = {values['offset']})"
        )

    # Calibration diagnostics and execution

    def open_calibration_results(self) -> None:
        self.content_tabs.set("Diagnostics")
        self.calibration_plot_tabs._reflow_buttons()


    def _apply_calibration_plot_typography(
        self,
        figure: Figure,
    ) -> None:
        zoom_percent = self.app.zoom_percent

        title_size = scaled_plot_font_size(
            UI["font_calibration_plot_title"],
            zoom_percent,
            minimum=8,
        )

        label_size = scaled_plot_font_size(
            UI["font_calibration_plot_label"],
            zoom_percent,
            minimum=8,
        )

        tick_size = scaled_plot_font_size(
            UI["font_calibration_plot_tick"],
            zoom_percent,
            minimum=7,
        )

        legend_size = scaled_plot_font_size(
            UI["font_calibration_plot_legend"],
            zoom_percent,
            minimum=7,
        )

        figure.set_facecolor("#111111")

        for axis in figure.axes:
            axis.title.set_fontsize(title_size)
            axis.xaxis.label.set_fontsize(label_size)
            axis.yaxis.label.set_fontsize(label_size)

            axis.tick_params(
                axis="both",
                labelsize=tick_size,
            )

            for annotation in axis.texts:
                annotation.set_fontsize(tick_size)

            legend = axis.get_legend()
            if legend is not None:
                for text_item in legend.get_texts():
                    text_item.set_fontsize(legend_size)

        for legend in figure.legends:
            for text_item in legend.get_texts():
                text_item.set_fontsize(legend_size)

        try:
            if figure.legends:
                figure.tight_layout(
                    rect=(0.0, 0.0, 1.0, 0.92),
                    pad=1.1,
                )
            else:
                figure.tight_layout(pad=1.2)
        except (RuntimeError, ValueError):
            pass

    def refresh_calibration_plot_layout(self) -> None:
        if hasattr(self, "calibration_plot_tabs"):
            self.calibration_plot_tabs.refresh_text_layout()
        for figure in self._calibration_figures.values():
            self._apply_calibration_plot_typography(figure)
            canvas = figure.canvas
            if canvas is not None:
                canvas.draw_idle()

    def _clear_calibration_figures(self) -> None:
        for host in self._calibration_plot_hosts.values():
            for child in host.winfo_children():
                child.destroy()
        for figure in self._calibration_figures.values():
            try:
                plt.close(figure)
            except Exception:
                pass
        self._calibration_figures.clear()
        self._calibration_canvases.clear()
        self._calibration_toolbars.clear()
        self._sync_calibration_plot_button_indicator()

    def _sync_calibration_plot_button_indicator(self) -> None:
        """Highlight the calibration button while plots are available."""

        if not hasattr(self, "open_calibration_button"):
            return

        plots_available = bool(self._calibration_figures)
        self.open_calibration_button.configure_button(
            border_color=(
                VIEWER_ACTIVE_BORDER_COLOR
                if plots_available
                else CALIBRATION_GRADIENT_STYLE["border_color"]
            ),
            border_width=(
                1.0
                if plots_available
                else CALIBRATION_GRADIENT_STYLE["border_width"]
            ),
            border_opacity=(
                1.0
                if plots_available
                else CALIBRATION_GRADIENT_STYLE["border_opacity"]
            ),
        )

    def _embed_calibration_figures(
        self,
        figures: dict[str, Figure],
        *,
        open_results: bool = True,
    ) -> None:
        self._clear_calibration_figures()
        expected_plots = {
            "parameters": "Line-shape parameters",
            "line_fits": "Calibration lines",
            "energy_calibration": "Energy calibration",
            "energy_deviation": "Energy deviation",
        }

        for plot_key, tab_name in expected_plots.items():
            host = self._calibration_plot_hosts[plot_key]
            figure = figures.get(plot_key)
            if not isinstance(figure, Figure):
                make_label(
                    host,
                    f"No {tab_name.lower()} plot was generated.",
                    size=UI["font_placeholder"],
                    bold=True,
                    text_color=("gray45", "gray60"),
                ).grid(row=0, column=0, sticky="nsew", padx=20, pady=20)
                continue

            self._apply_calibration_plot_typography(figure)
            plot_card = ctk.CTkFrame(
                host,
                fg_color="#111111",
                corner_radius=12,
                border_width=1,
                border_color=("gray65", "gray25"),
            )
            plot_card.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
            plot_card.grid_columnconfigure(0, weight=1)
            plot_card.grid_rowconfigure(0, weight=1)

            if plot_key == "line_fits":
                scroll_shell = ctk.CTkFrame(
                    plot_card,
                    fg_color="#111111",
                    corner_radius=0,
                )
                scroll_shell.grid(
                    row=0,
                    column=0,
                    sticky="nsew",
                    padx=7,
                    pady=(7, 0),
                )
                scroll_shell.grid_columnconfigure(0, weight=1)
                scroll_shell.grid_rowconfigure(0, weight=1)

                plot_scroll_canvas = tk.Canvas(
                    scroll_shell,
                    background="#111111",
                    highlightthickness=0,
                    borderwidth=0,
                )
                plot_scroll_canvas.grid(
                    row=0,
                    column=0,
                    sticky="nsew",
                )

                visible_line_axes = sum(
                    1 for axis in figure.axes if axis.get_visible()
                )
                line_fit_rows = max(1, (visible_line_axes + 3) // 4)
                line_fit_scrollbar = ctk.CTkScrollbar(
                    scroll_shell,
                    orientation="vertical",
                    command=plot_scroll_canvas.yview,
                    width=13,
                    fg_color="#202020",
                    button_color="#7652D6",
                    button_hover_color="#8D73DF",
                )
                plot_scroll_canvas.configure(
                    yscrollcommand=line_fit_scrollbar.set
                )
                if line_fit_rows > 3:
                    line_fit_scrollbar.grid(
                        row=0,
                        column=1,
                        sticky="ns",
                        padx=(5, 0),
                    )

                canvas_parent = tk.Frame(
                    plot_scroll_canvas,
                    background="#111111",
                    highlightthickness=0,
                    borderwidth=0,
                )
                canvas_parent.pack_propagate(False)
                scroll_window_id = plot_scroll_canvas.create_window(
                    0,
                    0,
                    window=canvas_parent,
                    anchor="nw",
                )
                canvas = FigureCanvasTkAgg(figure, master=canvas_parent)
                canvas_widget = canvas.get_tk_widget()
                canvas_widget.configure(
                    background="#111111",
                    highlightthickness=0,
                    borderwidth=0,
                )
                canvas_widget.pack(fill="both", expand=True)

                def resize_line_fit_figure(
                    event,
                    *,
                    row_count=line_fit_rows,
                    scroll_canvas=plot_scroll_canvas,
                    scroll_window=scroll_window_id,
                    inner_frame=canvas_parent,
                    figure_canvas=canvas,
                    figure_object=figure,
                ) -> None:
                    if event.width <= 2 or event.height <= 2:
                        return
                    visible_rows = min(row_count, 3)
                    full_height = max(
                        event.height,
                        round(event.height * row_count / visible_rows),
                    )
                    full_width = max(1, event.width)
                    scroll_canvas.itemconfigure(
                        scroll_window,
                        width=full_width,
                        height=full_height,
                    )
                    inner_frame.configure(
                        width=full_width,
                        height=full_height,
                    )
                    figure_object.set_size_inches(
                        full_width / figure_object.dpi,
                        full_height / figure_object.dpi,
                        forward=False,
                    )
                    scroll_canvas.configure(
                        scrollregion=(0, 0, full_width, full_height)
                    )
                    figure_canvas.draw_idle()

                plot_scroll_canvas.bind(
                    "<Configure>",
                    resize_line_fit_figure,
                    add="+",
                )

                def scroll_line_fits(event, scroll_canvas=plot_scroll_canvas):
                    event_number = getattr(event, "num", None)
                    if event_number == 4:
                        scroll_canvas.yview_scroll(-3, "units")
                        return "break"
                    if event_number == 5:
                        scroll_canvas.yview_scroll(3, "units")
                        return "break"
                    delta = int(getattr(event, "delta", 0))
                    if delta == 0:
                        return None
                    scroll_canvas.yview_scroll(
                        -3 if delta > 0 else 3,
                        "units",
                    )
                    return "break"

                for event_name in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                    canvas_widget.bind(
                        event_name,
                        scroll_line_fits,
                        add="+",
                    )
            else:
                canvas = FigureCanvasTkAgg(figure, master=plot_card)
                canvas_widget = canvas.get_tk_widget()
                canvas_widget.configure(
                    background="#111111",
                    highlightthickness=0,
                    borderwidth=0,
                )
                canvas_widget.grid(
                    row=0,
                    column=0,
                    sticky="nsew",
                    padx=7,
                    pady=(7, 0),
                )

                def resize_embedded_figure(
                    event,
                    *,
                    figure_canvas=canvas,
                    figure_object=figure,
                ) -> None:
                    if event.width <= 2 or event.height <= 2:
                        return
                    figure_object.set_size_inches(
                        event.width / figure_object.dpi,
                        event.height / figure_object.dpi,
                        forward=False,
                    )
                    figure_canvas.draw_idle()

                canvas_widget.bind(
                    "<Configure>",
                    resize_embedded_figure,
                    add="+",
                )

                def sync_embedded_figure_size(
                    *,
                    figure_object=figure,
                ) -> None:
                    canvas_widget.update_idletasks()
                    width = canvas_widget.winfo_width()
                    height = canvas_widget.winfo_height()
                    if width <= 2 or height <= 2:
                        return
                    figure_object.set_size_inches(
                        width / figure_object.dpi,
                        height / figure_object.dpi,
                        forward=False,
                    )
                    canvas.draw_idle()

                canvas_widget.after_idle(sync_embedded_figure_size)

            native_toolbar_frame = tk.Frame(
                plot_card,
                background="#111111",
                highlightthickness=0,
                borderwidth=0,
            )
            native_toolbar = NavigationToolbar2Tk(
                canvas,
                native_toolbar_frame,
                pack_toolbar=False,
            )
            native_toolbar.update()

            modern_toolbar = ctk.CTkFrame(
                plot_card,
                fg_color="transparent",
                corner_radius=0,
            )
            modern_toolbar.grid(
                row=1,
                column=0,
                sticky="ew",
                padx=10,
                pady=(5, 7),
            )
            modern_toolbar.grid_columnconfigure(6, weight=1)

            toolbar_buttons: dict[str, ctk.CTkButton] = {}

            def update_modes(
                toolbar=native_toolbar,
                buttons=toolbar_buttons,
            ) -> None:
                mode_name = str(getattr(toolbar, "mode", "")).lower()
                for mode in ("pan", "zoom"):
                    if mode not in buttons:
                        continue
                    active = mode in mode_name
                    buttons[mode].configure(
                        fg_color="#3B247A" if active else "#292929",
                        border_color="#9B7AE8" if active else "#4A4A4A",
                    )

            def run_action(
                action: str,
                toolbar=native_toolbar,
                update=update_modes,
            ) -> None:
                getattr(toolbar, action)()
                update()

            for toolbar_column, (action, text_value) in enumerate(
                (
                    ("home", "Home"),
                    ("back", "Back"),
                    ("forward", "Next"),
                    ("pan", "Pan"),
                    ("zoom", "Zoom"),
                    ("save_figure", "Export plot"),
                )
            ):
                toolbar_button = ctk.CTkButton(
                    modern_toolbar,
                    text=text_value,
                    width=48 if action != "save_figure" else 82,
                    height=29,
                    corner_radius=8,
                    fg_color="#292929",
                    hover_color="#3A3A3A",
                    border_width=1,
                    border_color="#4A4A4A",
                    font=ui_font(UI["font_calibration_toolbar"], bold=True),
                    command=(
                        lambda selected_action=action, selected_runner=run_action:
                        selected_runner(selected_action)
                    ),
                )
                toolbar_button.grid(
                    row=0,
                    column=toolbar_column,
                    padx=(0, 5),
                )
                toolbar_buttons[action] = toolbar_button

            coordinate_label = ctk.CTkLabel(
                modern_toolbar,
                text="",
                anchor="e",
                text_color=("gray35", "gray65"),
                fg_color="transparent",
                font=ui_font(UI["font_calibration_toolbar"]),
            )
            coordinate_label.grid(
                row=0,
                column=6,
                sticky="ew",
                padx=(10, 3),
            )

            def update_coordinates(event, label=coordinate_label) -> None:
                if event.inaxes is None or event.xdata is None or event.ydata is None:
                    label.configure(text="")
                    return
                label.configure(
                    text=f"x = {event.xdata:.5g}    y = {event.ydata:.5g}"
                )

            canvas.mpl_connect("motion_notify_event", update_coordinates)
            canvas.draw()
            self._calibration_figures[plot_key] = figure
            self._calibration_canvases[plot_key] = canvas
            self._calibration_toolbars[plot_key] = native_toolbar

        self._sync_calibration_plot_button_indicator()
        self.calibration_plot_tabs.set("Line-shape parameters")
        if open_results:
            self.open_calibration_results()
        self.update_idletasks()
        self.refresh_calibration_plot_layout()

    def run_calibration_check(self) -> None:
        self._pending_calibration_for_viewer = None
        self._sync_apply_calibration_button_state()

        try:
            calibration_text = self.calibration_scans_entry.get().strip()
            if not calibration_text:
                raise ValueError("Please enter Calibration Scans first.")

            calibration_scans = parse_scans(calibration_text)
            search_roots = self._get_valid_search_roots()
            options = self._get_calibration_options()

            viewer_module = mev_viewer
            if not hasattr(viewer_module, "compute_energy_calibration_2"):
                raise AttributeError(
                    "The viewer module does not contain "
                    "'compute_energy_calibration_2'."
                )

            kwargs = self._build_calibration_kwargs(
                calibration_scans,
                search_roots,
                options,
                show_calibration_plots=False,
                show_test_plot=False,
            )
            kwargs["return_details"] = True
            kwargs["return_figures"] = True
        except Exception as exc:
            self.app.show_styled_message(
                title="Calibration input error",
                message=str(exc),
                kind="error",
            )
            return

        self.calibration_check_button.configure_button(
            state="disabled",
            text="Calibration is running...",
        )
        self.log(
            f"Running calibration check: polynomial order "
            f"{options.fit_poly_order}, {options.model_name}."
        )
        self.log(f"Peak selection buffer: {options.peak_buffer} px.")
        if options.params_poly_orders is not None:
            self.log(
                "Parameter polynomial orders (sigma1, sigma2, R, delta): "
                f"{options.params_poly_orders}"
            )
        self.log("Calibration search roots:")
        for root in search_roots:
            self.log(f"  {root}")

        loading_token = self._begin_local_loading(
            self._setup_cards["calibration"],
            "Computing calibration and diagnostics…",
        )
        self.after(
            50,
            lambda: self._open_calibration_check(kwargs, loading_token),
        )

    def _open_calibration_check(
        self,
        kwargs: dict,
        loading_token: dict[str, object] | None = None,
    ) -> None:
        try:
            calibration_output = (
                mev_viewer.compute_energy_calibration_2(**kwargs)
            )
            calibration_details, calibration_figures = (
                _unpack_calibration_output(calibration_output)
            )
            self._calibration_cache = calibration_details
            self._pending_calibration_for_viewer = copy.deepcopy(
                calibration_details
            )
            self._sync_apply_calibration_button_state()
            coefficients_obj = calibration_details.get("energy_coefficients")

            if not isinstance(coefficients_obj, (list, tuple)):
                raise TypeError(
                    "Invalid energy_coefficients in calibration details."
                )

            coefficients: list[float] = []

            for value in coefficients_obj:
                if not isinstance(value, (int, float, np.number)):
                    raise TypeError(
                        "Invalid value in energy_coefficients."
                    )
                coefficients.append(float(value))


            calibration_sigma = calibration_details.get(
                "calibration_sigma_eV"
            )
            mean_fwhm = calibration_details.get(
                "mean_fwhm_eV"
            )

            if not isinstance(calibration_sigma, (int, float, np.number)):
                raise TypeError(
                    "Invalid calibration_sigma_eV in calibration details."
                )

            if not isinstance(mean_fwhm, (int, float, np.number)):
                raise TypeError(
                    "Invalid mean_fwhm_eV in calibration details."
                )


            self.log(
                "Calibration coefficients (a3, a2, a1, a0): "
                + ", ".join(
                    f"{value:.8g}"
                    for value in coefficients
                )
            )

            self.log(
                "Calibration quality: "
                f"sigma = {float(calibration_sigma):.5g} eV, "
                f"mean FWHM = {float(mean_fwhm):.5g} eV."
            )
            self._embed_calibration_figures(calibration_figures)
        except Exception as exc:
            self.app.show_styled_message(
                title="Calibration error",
                message=str(exc),
                kind="error",
            )
            self.log(f"Calibration error: {exc}")
        finally:
            if loading_token is not None:
                self._end_local_loading(loading_token)
            self.calibration_check_button.configure_button(
                state="normal",
                text="Run Diagnostics",
            )

    def _compute_energy_calibration(
        self,
        options: ViewerOptions,
    ) -> tuple[object | None, float | None]:
        if options.calibration_scans is None:
            return None, None

        viewer_module = mev_viewer

        if not hasattr(
            viewer_module,
            "compute_energy_calibration_2",
        ):
            raise AttributeError(
                "The viewer module does not contain "
                "'compute_energy_calibration_2'."
            )

        cached_calibration = self._calibration_cache_matches(
            self._calibration_cache,
            options.calibration_scans,
            options.calibration,
        )

        cached_incident_energy: float | None = None

        if (
            cached_calibration
            and self._calibration_cache is not None
        ):
            incident_cache = self._calibration_cache.get(
                "incident_energy"
            )

            if isinstance(incident_cache, dict):
                try:
                    cached_scans = [
                        int(scan)
                        for scan in incident_cache["scans"]
                    ]

                    if cached_scans == [
                        int(scan)
                        for scan in options.scans
                    ]:
                        cached_incident_energy = float(
                            incident_cache["mean_eV"]
                        )

                except (
                    KeyError,
                    TypeError,
                    ValueError,
                ):
                    cached_incident_energy = None

        if (
            cached_calibration
            and cached_incident_energy is not None
        ):
            assert self._calibration_cache is not None

            coefficients = self._cached_energy_coefficients(
                self._calibration_cache
            )

            self.log(
                "Using the existing matching calibration; "
                "the calibration fit is not repeated."
            )
            self.log(
                f"Incident energy E_in: "
                f"{cached_incident_energy:.8g}"
            )

            return coefficients, cached_incident_energy

        if cached_calibration:
            assert self._calibration_cache is not None

            calibration_result = (
                self._cached_energy_coefficients(
                    self._calibration_cache
                )
            )

            self.log(
                "Using the existing matching calibration; "
                "the calibration fit is not repeated."
            )

        else:
            calibration_kwargs = (
                self._build_calibration_kwargs(
                    options.calibration_scans,
                    options.search_roots,
                    options.calibration,
                    show_calibration_plots=False,
                    show_test_plot=False,
                )
            )

            calibration_kwargs["return_details"] = True
            calibration_kwargs["return_figures"] = True

            with plt.rc_context():
                calibration_output = (
                    viewer_module.compute_energy_calibration_2(
                        **calibration_kwargs
                    )
                )

            calibration_details, calibration_figures = (
                _unpack_calibration_output(calibration_output)
            )

            self._calibration_cache = calibration_details

            calibration_result = (
                self._cached_energy_coefficients(
                    self._calibration_cache
                )
            )

            self._embed_calibration_figures(
                calibration_figures,
                open_results=False,
            )

            self.log(
                "Calibration plots are available in the Diagnostics view."
            )

        # Make sure metadata for the measured scans is cached.
        self._ensure_spec_metadata(
            options.scans
        )

        missing_scans: list[int] = []
        incident_energy_values: list[float] = []

        for scan in options.scans:
            scan_num = int(scan)

            motor_values = self._metadata_cache.get(
                scan_num
            )

            if (
                motor_values is None
                or "pgm_en" not in motor_values
            ):
                missing_scans.append(scan_num)
                continue

            try:
                incident_energy_values.append(
                    float(motor_values["pgm_en"])
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Motor value pgm_en for scan "
                    f"{scan_num} is not numeric: "
                    f"{motor_values['pgm_en']!r}"
                ) from exc

        if missing_scans:
            raise ValueError(
                "Could not determine incident energy for scans: "
                f"{missing_scans}. Missing motor value: pgm_en"
            )

        incident_energies = np.asarray(
            incident_energy_values,
            dtype=float,
        )

        incident_energy = float(
            np.mean(incident_energies)
        )

        if self._calibration_cache is not None:
            self._calibration_cache["incident_energy"] = {
                "scans": [
                    int(scan)
                    for scan in options.scans
                ],
                "values_eV": [
                    float(value)
                    for value in incident_energies
                ],
                "mean_eV": incident_energy,
            }

        self.log(
            f"Incident energy E_in: {incident_energy:.8g}"
        )
        self.log("Energy calibration points:")

        return calibration_result, incident_energy

    # Embedded viewer lifecycle

    def _viewer_session_state_for_start(
        self,
        pending_state: dict | None,
    ) -> dict | None:
        """Apply global preferences only when this Dataset opted in."""

        if not self.apply_global_preferences_var.get():
            return pending_state
        return self.app.merge_global_viewer_settings(pending_state)

    def run_viewer(self) -> None:
        if self.viewer_figure is not None:
            confirmed = self.app.ask_styled_confirmation(
                title="Restart Viewer",
                message="Do you really want to restart the Viewer?",
                detail=(
                    "The current Viewer settings and plot state will be "
                    "overwritten."
                ),
                destructive=True,
            )
            if not confirmed:
                return

        pending_viewer_state = self._pending_viewer_session_state
        content_tab_after_open = self._pending_content_tab
        self._pending_viewer_session_state = None
        self._pending_content_tab = "Viewer"

        loading_token: dict[str, object] | None = None

        try:
            options = self._collect_viewer_options()

            has_matching_calibration = (
                options.calibration_scans is not None
                and self._calibration_cache_matches(
                    self._calibration_cache,
                    options.calibration_scans,
                    options.calibration,
                )
            )

            if options.calibration_scans is None:
                loading_message = "Loading scan files…"
            elif has_matching_calibration:
                loading_message = "Preparing Viewer…"
            else:
                loading_message = "Computing calibration…"

            self.content_tabs.set("Viewer")
            loading_token = self._begin_local_loading(
                self.content_tabs.tab("Viewer"),
                loading_message,
            )

            self._ensure_scan_file_cache(
                options.scans,
                options.search_roots,
            )

            energy_calibration, incident_energy = (
                self._compute_energy_calibration(
                    options
                )
            )
            kwargs = {
                "scans": options.scans,
                "scans_dir": None,
                "search_roots": options.search_roots,
                "scan_files_by_scan": self._scan_file_cache,
                "spectra_dir": options.spectra_dir,
                "histogram_dir": options.histogram_dir,
                "energy_calibration": energy_calibration,
                "incident_energy": incident_energy,
                "window_title": options.dataset_name,
                "export_metadata": {
                    "dataset_name": options.dataset_name,
                    "source_scans": [int(scan) for scan in options.scans],
                    "calibration_scans": (
                        [int(scan) for scan in options.calibration_scans]
                        if options.calibration_scans is not None
                        else []
                    ),
                    "scan_search_roots": [
                        str(root) for root in options.search_roots
                    ],
                    "spec_search_root": self.spec_search_root_var.get(),
                    "spec_file": self.spec_file_var.get(),
                    "spec_metadata_by_scan": self._metadata_cache,
                    "calibration": self._calibration_cache,
                },
                "viewer_font_sizes": {
                    key: int(value)
                    for key, value in UI.items()
                    if key.startswith("font_viewer_")
                },
                "viewer_font_factory": lambda size, *, bold=False: ui_font(
                    size,
                    family=self.app.gui_font_family,
                    bold=bold,
                ),
            }
            effective_viewer_state = self._viewer_session_state_for_start(
                pending_viewer_state
            )
            if effective_viewer_state is not None:
                kwargs["session_state"] = effective_viewer_state
        except Exception as exc:
            if loading_token is not None:
                self._end_local_loading(loading_token)
            self.app.show_styled_message(
                title="Input error",
                message=str(exc),
                kind="error",
            )
            return

        self.run_button.configure_button(state="disabled", text="Viewer is running...")
        self.log(f"Starting viewer for scans: {options.scans.tolist()}")
        self.log(f"Dataset name: {options.dataset_name}")
        self.log(f"Scan Files Search Root: {options.search_roots}")
        self.log(
            f"Calibration polynomial order: "
            f"{options.calibration.fit_poly_order}"
        )
        self.log(f"Calibration peak model: {options.calibration.model_name}")
        if options.spectra_dir is not None:
            self.log(f"Spectra Dir.: {options.spectra_dir}")
        if options.histogram_dir is not None:
            self.log(f"Histogram Dir.: {options.histogram_dir}")

        assert loading_token is not None
        self._update_local_loading(
            loading_token,
            "Loading scan files and building the Viewer…",
        )
        # Tk-based Matplotlib rendering must remain in the main thread.
        self.after(
            50,
            lambda: self._open_viewer(
                kwargs,
                content_tab_after_open,
                loading_token,
            ),
        )

    def _open_viewer(
        self,
        kwargs: dict,
        content_tab_after_open: str = "Viewer",
        loading_token: dict[str, object] | None = None,
    ) -> None:
        try:
            if self.viewer_figure is not None:
                try:
                    plt.close(self.viewer_figure)
                except Exception:
                    pass
            placeholder_figure = getattr(
                self,
                "viewer_placeholder_figure",
                None,
            )
            if placeholder_figure is not None:
                try:
                    plt.close(placeholder_figure)
                except Exception:
                    pass
            for child in self.viewer_frame.winfo_children():
                child.destroy()

            self.viewer_figure = None
            self.viewer_axes = None
            self.viewer_placeholder_figure = None
            self.viewer_placeholder_canvas = None
            self.content_tabs.set("Viewer")
            self.update_idletasks()

            self.log(f"Launcher file: {Path(__file__).resolve()}")
            self.viewer_figure, self.viewer_axes = mev_viewer.view_spectra(
                **kwargs,
                tk_parent=self.viewer_frame,
            )
            self.refresh_open_viewer_plot_typography()
            self._record_open_viewer_calibration()
            self._sync_open_viewer_save_directories()
            self.content_tabs.set(content_tab_after_open)
            self.log("Viewer embedded in the current dataset tab.")
        except Exception as exc:
            self._viewer_applied_calibration = None
            self._pending_calibration_for_viewer = None
            if not self.viewer_frame.winfo_children():
                self._build_viewer_placeholder()
            self._sync_apply_calibration_button_state()
            self.app.show_styled_message(
                title="Viewer startup error",
                message=str(exc),
                kind="error",
            )
            self.log(f"Error: {exc}")
        finally:
            self.run_button.configure_button(state="normal", text="Start Viewer")
            self._sync_viewer_tab_indicator()
            if loading_token is not None:
                self._end_local_loading(loading_token)

    def update_open_viewer_title(self, title: str) -> None:
        if self.viewer_figure is None:
            return

        keepalive = getattr(self.viewer_figure, "_view_spectra_keepalive", None)
        if not isinstance(keepalive, dict):
            return

        set_viewer_title = keepalive.get("set_viewer_title")
        if not callable(set_viewer_title):
            return

        try:
            set_viewer_title(title)
        except tk.TclError:
            return

    def refresh_open_viewer_plot_typography(self) -> None:
        if (
            self.viewer_figure is None
            or self.viewer_axes is None
        ):
            return

        zoom_percent = self.app.zoom_percent

        label_size = scaled_plot_font_size(
            UI["font_viewer_plot_label"],
            zoom_percent,
            minimum=8,
        )

        tick_size = scaled_plot_font_size(
            UI["font_viewer_plot_tick"],
            zoom_percent,
            minimum=7,
        )

        for axis in np.asarray(
            self.viewer_axes,
            dtype=object,
        ).flat:
            axis.xaxis.label.set_fontsize(label_size)
            axis.yaxis.label.set_fontsize(label_size)

            axis.tick_params(
                axis="both",
                labelsize=tick_size,
            )

        keepalive = getattr(
            self.viewer_figure,
            "_view_spectra_keepalive",
            None,
        )

        if isinstance(keepalive, dict):
            for key in (
                "energy_axis",
                "loss_axis",
            ):
                secondary_axis = keepalive.get(key)

                if secondary_axis is None:
                    continue

                secondary_axis.xaxis.label.set_fontsize(
                    label_size
                )
                secondary_axis.tick_params(
                    axis="x",
                    labelsize=tick_size,
                )

        self.viewer_figure.canvas.draw_idle()

    def refresh_open_viewer_section_layout(self) -> None:
        """Restore collapsed viewer sections after CustomTkinter rescales them."""

        if self.viewer_figure is None:
            return

        keepalive = getattr(self.viewer_figure, "_view_spectra_keepalive", None)
        if not isinstance(keepalive, dict):
            return

        refresh_section_layout = keepalive.get("refresh_section_layout")
        if not callable(refresh_section_layout):
            return

        try:
            refresh_section_layout()
        except tk.TclError:
            return

    def _sync_viewer_tab_indicator(self) -> None:
        """Show an orange Viewer-tab border while a viewer is available."""

        self.content_tabs.set_tab_border_highlight(
            "Viewer",
            self.viewer_figure is not None,
            color=VIEWER_ACTIVE_BORDER_COLOR,
            width=1.0,
            opacity=1.0,
        )

# =============================================================================
# Main application window
# =============================================================================


class SpectraLauncher(ctk.CTk):
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        configure_platform_gui_scaling(UI["user_scale"])
        super().__init__()
        if platform.system() == "Windows":
            try:
                self.iconbitmap(resource_path("icon8.ico"))
            except (tk.TclError, OSError):
                pass
        else:
            try:
                icon_image = tk.PhotoImage(file=resource_path("icon8.png"))
                self.iconphoto(True, icon_image)
                self._icon_image = icon_image
            except (tk.TclError, OSError):
                pass

        self.zoom_percent = 100
        self.text_size_percent = 100
        self.settings_percent_values = [
            f"{value} %" for value in range(50, 151, 10)
        ]
        self._settings_dialog: ctk.CTkToplevel | None = None
        self._global_viewer_settings_dialog: ctk.CTkToplevel | None = None
        self._global_viewer_settings_configured = False
        self._global_viewer_settings_enabled: set[tuple[str, str]] = set()
        self._global_viewer_settings_values: dict[str, object] = {
            "version": 1,
            "settings": {},
            "mode": {},
            "controls": {},
        }
        for spec in GLOBAL_VIEWER_SETTING_SPECS:
            section = str(spec["section"])
            values = self._global_viewer_settings_values[section]
            assert isinstance(values, dict)
            values[str(spec["key"])] = copy.deepcopy(spec["default"])
        self._loading_requests: list[tuple[object, str]] = []
        self._after_loading_callbacks: list[Callable[[], None]] = []
        self._resize_finish_job: str | None = None
        self._resize_in_progress = False
        self._last_root_size: tuple[int, int] | None = None
        self._custom_resize_state: dict[str, int | str] | None = None

        # Let the native geometry code run normally, but keep intermediate
        # window states invisible until the complete GUI has been laid out.
        self._startup_transparency_supported = False
        try:
            self.attributes("-alpha", 0.0)
            self._startup_transparency_supported = True
        except tk.TclError:
            pass

        self.gui_font_family = get_cross_platform_gui_font_family(self)
        ctk.ThemeManager.theme["CTkFont"]["family"] = self.gui_font_family
        configure_native_tk_fonts(self, self.gui_font_family)

        self.readonly_value_font = ui_font(
            UI["font_viewer_value"],
            family=self.gui_font_family,
        )
        self.readonly_value_bold_font = ui_font(
            UI["font_viewer_value"],
            family=self.gui_font_family,
            bold=True,
        )

        print(f"GUI font: {self.gui_font_family}")
        if self.gui_font_family != "Courier New":
            print(
                "\033[31mWarning\033[0m: The GUI font is not 'Courier New'.\n"
                "Some text may not align properly.\n"
                "Consider installing mscorefonts using: \033[35msudo apt install ttf-mscorefonts-installer \033[0m \n"
                "Clear matplotlib cache if the font is installed but not detected using:\n"
                "\033[35mrm -f ~/.cache/matplotlib/fontlist-v*.json \033[0m "
            )

        # Fallback color, visible briefly before the gradient is drawn.
        self.configure(fg_color="#141414")

        self.session_name = UNTITLED_SESSION_NAME
        self._update_window_title()
        self.set_responsive_initial_geometry()
        self._build_window_gradient_background()

        self.search_root_history: list[str] = []
        self.spec_search_root_history: list[str] = []

        self.tab_counter = 0
        self.tabs: dict[str, SpectraTab] = {}
        self.first_tab_name: str | None = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self._build_top_bar()
        self._build_tabs()
        self.add_plot_tab()
        self._build_resize_overlay()
        self._build_resize_handles()
        self._build_loading_overlay()
        self.protocol("WM_DELETE_WINDOW", self.close_application)
        self.after_idle(self._show_initialized_window)

    def _show_initialized_window(self) -> None:
        """Reveal the fully laid-out GUI without changing its geometry."""
        self.update_idletasks()
        self._redraw_window_gradient()
        self.update_idletasks()

        if self._startup_transparency_supported:
            try:
                self.attributes("-alpha", 1.0)
            except tk.TclError:
                pass
        self.lift()

    @staticmethod
    def _session_name_from_path(path: str | Path) -> str:
        session_name = Path(path).stem.strip()
        return session_name or UNTITLED_SESSION_NAME

    def _set_session_name(self, session_name: str) -> None:
        normalized_name = str(session_name).strip() or UNTITLED_SESSION_NAME
        self.session_name = normalized_name
        self._update_window_title()

    def _update_window_title(self) -> None:
        self.title(f"{APPLICATION_TITLE} - {self.session_name}")

    def _build_resize_overlay(self) -> None:
        """Create a calm placeholder shown during interactive window resizing."""

        self.resize_overlay = ctk.CTkFrame(
            self,
            fg_color="#121419",
            corner_radius=0,
        )
        self.resize_overlay_label = make_label(
            self.resize_overlay,
            "Resizing…",
            size=UI["font_status"],
            bold=True,
            text_color=("gray30", "gray76"),
        )
        self.resize_overlay_label.place(relx=0.5, rely=0.5, anchor="center")

    def _build_resize_handles(self) -> None:
        """Add generous inner resize targets without replacing native borders."""

        handle_specs = {
            "n": ("sb_v_double_arrow", {"x": 14, "y": 0, "relwidth": 1, "width": -28, "height": 9}),
            "s": ("sb_v_double_arrow", {"x": 14, "rely": 1, "y": -9, "relwidth": 1, "width": -28, "height": 9}),
            "w": ("sb_h_double_arrow", {"x": 0, "y": 14, "width": 9, "relheight": 1, "height": -28}),
            "e": ("sb_h_double_arrow", {"relx": 1, "x": -9, "y": 14, "width": 9, "relheight": 1, "height": -28}),
            "nw": ("size_nw_se", {"x": 0, "y": 0, "width": 14, "height": 14}),
            "ne": ("size_ne_sw", {"relx": 1, "x": -14, "y": 0, "width": 14, "height": 14}),
            "sw": ("size_ne_sw", {"x": 0, "rely": 1, "y": -14, "width": 14, "height": 14}),
            "se": ("size_nw_se", {"relx": 1, "rely": 1, "x": -14, "y": -14, "width": 14, "height": 14}),
        }
        self._resize_handles: list[tk.Frame] = []
        for edge, (cursor_name, placement) in handle_specs.items():
            try:
                handle = tk.Frame(
                    self,
                    bd=0,
                    highlightthickness=0,
                    bg="#141414",
                    cursor=cursor_name,
                )
            except tk.TclError:
                handle = tk.Frame(
                    self,
                    bd=0,
                    highlightthickness=0,
                    bg="#141414",
                )
            handle.place(x=placement.get("x", 0),
                y=placement.get("y", 0),
                relx=placement.get("relx", 0.0),
                rely=placement.get("rely", 0.0),
                width=placement.get("width", 14),
                height=placement.get("height", 14),)
            handle.bind(
                "<ButtonPress-1>",
                lambda event, selected_edge=edge: self._begin_custom_resize(
                    event,
                    selected_edge,
                ),
            )
            handle.bind("<B1-Motion>", self._drag_custom_resize)
            handle.bind("<ButtonRelease-1>", self._end_custom_resize)
            self._resize_handles.append(handle)

    def _begin_custom_resize(self, event, edge: str) -> None:
        self._custom_resize_state = {
            "edge": edge,
            "pointer_x": int(event.x_root),
            "pointer_y": int(event.y_root),
            "width": int(self.winfo_width()),
            "height": int(self.winfo_height()),
            "x": int(self.winfo_x()),
            "y": int(self.winfo_y()),
        }
        self._begin_interactive_resize()

    def _drag_custom_resize(self, event) -> None:
        state = self._custom_resize_state
        if not isinstance(state, dict):
            return

        edge = str(state["edge"])
        delta_x = int(event.x_root) - int(state["pointer_x"])
        delta_y = int(event.y_root) - int(state["pointer_y"])
        width = int(state["width"])
        height = int(state["height"])
        x_position = int(state["x"])
        y_position = int(state["y"])
        minimum_width, minimum_height = self.wm_minsize()

        if "e" in edge:
            width = max(minimum_width, width + delta_x)
        if "s" in edge:
            height = max(minimum_height, height + delta_y)
        if "w" in edge:
            proposed_width = max(minimum_width, width - delta_x)
            x_position += width - proposed_width
            width = proposed_width
        if "n" in edge:
            proposed_height = max(minimum_height, height - delta_y)
            y_position += height - proposed_height
            height = proposed_height

        tk.Tk.geometry(
            self,
            f"{width}x{height}+{x_position}+{y_position}",
        )

    def _end_custom_resize(self, _event=None) -> None:
        self._custom_resize_state = None
        self._schedule_resize_finish(delay_ms=40)

    def _begin_interactive_resize(self) -> None:
        if not hasattr(self, "resize_overlay") or self._loading_requests:
            return

        if not self._resize_in_progress:
            self._resize_in_progress = True
            if hasattr(self, "top_bar"):
                self.top_bar.grid_remove()
            if hasattr(self, "tabview"):
                self.tabview.grid_remove()
            self.resize_overlay.place(
                x=0,
                y=0,
                relwidth=1,
                relheight=1,
            )
            self.resize_overlay.lift()

        self._schedule_resize_finish()

    def _schedule_resize_finish(self, *, delay_ms: int = 240) -> None:
        if self._resize_finish_job is not None:
            self.after_cancel(self._resize_finish_job)
        self._resize_finish_job = self.after(
            delay_ms,
            self._finish_interactive_resize,
        )

    def _finish_interactive_resize(self) -> None:
        self._resize_finish_job = None
        if not self._resize_in_progress:
            return

        if hasattr(self, "top_bar"):
            self.top_bar.grid()
        if hasattr(self, "tabview"):
            self.tabview.grid()
        self.update_idletasks()
        self._redraw_window_gradient()
        self.update_idletasks()
        self.resize_overlay.place_forget()
        self._resize_in_progress = False
        for handle in getattr(self, "_resize_handles", []):
            handle.lift()

    def _on_root_configure(self, event=None) -> None:
        if event is not None and event.widget is not self:
            return

        current_size = (int(self.winfo_width()), int(self.winfo_height()))
        previous_size = self._last_root_size
        self._last_root_size = current_size
        if previous_size is None or current_size == previous_size:
            return
        self._begin_interactive_resize()

    # Loading overlay

    def _build_loading_overlay(self) -> None:
        """Create the full-window loading view used by blocking GUI operations."""

        self.loading_overlay = ctk.CTkFrame(
            self,
            fg_color="#121419",
            corner_radius=0,
        )
        self.loading_overlay.grid_columnconfigure(0, weight=1)
        self.loading_overlay.grid_rowconfigure(0, weight=1)
        self.loading_overlay.grid_rowconfigure(3, weight=1)

        self.loading_title_label = make_label(
            self.loading_overlay,
            "Loading…",
            size=UI["font_title"],
            family=self.gui_font_family,
            bold=True,
            text_color="#F4F6FA",
        )
        self.loading_title_label.grid(
            row=1,
            column=0,
            padx=30,
            pady=(30, 4),
        )

        self.loading_message_label = make_label(
            self.loading_overlay,
            "Please wait.",
            size=UI["font_normal"],
            family=self.gui_font_family,
            text_color="#AEB7C6",
            wraplength=360,
            justify="center",
        )
        self.loading_message_label.grid(
            row=2,
            column=0,
            padx=30,
            pady=(0, 30),
        )

        self._refresh_loading_overlay_layout()
        self.loading_overlay.place_forget()

    def _refresh_loading_overlay_layout(self) -> None:
        """Keep loading labels tall enough for the selected text size."""

        text_scale = self.text_size_percent / 100
        self.loading_title_label.configure(
            height=max(
                44,
                round(UI["font_title"] * 1.8 * text_scale),
            )
        )
        self.loading_message_label.configure(
            height=max(
                34,
                round(UI["font_normal"] * 1.8 * text_scale),
            )
        )

    def begin_loading(self, message: str) -> object:
        """Show the loading overlay and return a token for this operation."""

        token = object()
        overlay_was_hidden = not self._loading_requests
        self._loading_requests.append((token, str(message)))
        self.loading_message_label.configure(text=str(message))
        if overlay_was_hidden:
            self.loading_overlay.place(x=0, y=0, relwidth=1, relheight=1)
        self.loading_overlay.lift()
        self.update_idletasks()
        return token

    def update_loading(self, token: object, message: str) -> None:
        """Update one active loading operation without changing its lifetime."""

        for index, (active_token, _active_message) in enumerate(
            self._loading_requests
        ):
            if active_token is token:
                self._loading_requests[index] = (token, str(message))
                break
        else:
            return

        if self._loading_requests[-1][0] is token:
            self.loading_message_label.configure(text=str(message))
            self.loading_overlay.lift()
            self.update_idletasks()

    def end_loading(self, token: object) -> None:
        """Finish one loading operation and hide the overlay when all are done."""

        self._loading_requests = [
            request
            for request in self._loading_requests
            if request[0] is not token
        ]

        if self._loading_requests:
            self.loading_message_label.configure(
                text=self._loading_requests[-1][1]
            )
            self.loading_overlay.lift()
        else:
            self.loading_overlay.place_forget()
            callbacks = self._after_loading_callbacks
            self._after_loading_callbacks = []
            for callback in callbacks:
                self.after_idle(callback)
        self.update_idletasks()

    def run_after_loading(self, callback: Callable[[], None]) -> None:
        """Run a GUI callback once every active loading operation has ended."""

        if self._loading_requests:
            self._after_loading_callbacks.append(callback)
        else:
            self.after_idle(callback)

    # Cross-platform window geometry

    def set_responsive_initial_geometry(self) -> None:
        self.deiconify()
        self.update_idletasks()
        self.update()

        system = platform.system()

        def is_wsl() -> bool:
            return (
                "WSL_DISTRO_NAME" in os.environ
                or "WSL_INTEROP" in os.environ
                or "microsoft" in platform.release().lower()
            )

        def calculate_client_width(
            work_width: int,
            horizontal_nonclient: int,
        ) -> tuple[int, int]:
            max_client_width = max(
                1,
                work_width - horizontal_nonclient,
            )

            minimum_width = min(
                1200,
                max_client_width,
            )

            preferred_width = min(
                int(work_width * 0.88),
                1900,
                max_client_width,
            )

            client_width = max(
                minimum_width,
                preferred_width,
            )

            return client_width, max_client_width

        def set_normal_state() -> None:
            try:
                self.attributes("-zoomed", False)
            except tk.TclError:
                pass

            try:
                tk.Wm.state(self, "normal")
            except tk.TclError:
                pass

            self.update_idletasks()
            self.update()

        def set_zoomed_state() -> bool:
            try:
                self.attributes("-zoomed", True)
                self.update_idletasks()
                self.update()
                return True
            except tk.TclError:
                pass

            try:
                tk.Wm.state(self, "zoomed")
                self.update_idletasks()
                self.update()
                return True
            except tk.TclError:
                return False

        # =========================================================
        # Windows
        # =========================================================
        if system == "Windows":
            from ctypes import wintypes

            user32 = ctypes.windll.user32  # type: ignore[attr-defined]

            GA_ROOT = 2
            MONITOR_DEFAULTTONEAREST = 2

            SWP_NOZORDER = 0x0004
            SWP_NOACTIVATE = 0x0010
            SWP_FRAMECHANGED = 0x0020

            class MONITORINFO(ctypes.Structure):
                _fields_ = [
                    ("cbSize", wintypes.DWORD),
                    ("rcMonitor", wintypes.RECT),
                    ("rcWork", wintypes.RECT),
                    ("dwFlags", wintypes.DWORD),
                ]

            user32.GetAncestor.argtypes = [
                wintypes.HWND,
                wintypes.UINT,
            ]
            user32.GetAncestor.restype = wintypes.HWND

            user32.MonitorFromWindow.argtypes = [
                wintypes.HWND,
                wintypes.DWORD,
            ]
            user32.MonitorFromWindow.restype = wintypes.HANDLE

            hwnd = user32.GetAncestor(
                self.winfo_id(),
                GA_ROOT,
            )

            if not hwnd:
                hwnd = self.winfo_id()

            monitor = user32.MonitorFromWindow(
                hwnd,
                MONITOR_DEFAULTTONEAREST,
            )

            monitor_info = MONITORINFO()
            monitor_info.cbSize = ctypes.sizeof(MONITORINFO)

            if not user32.GetMonitorInfoW(
                monitor,
                ctypes.byref(monitor_info),
            ):
                raise ctypes.WinError()  # type: ignore[attr-defined]

            work_area = monitor_info.rcWork

            work_x = work_area.left
            work_y = work_area.top
            work_width = work_area.right - work_area.left
            work_height = work_area.bottom - work_area.top

            window_rect = wintypes.RECT()
            client_rect = wintypes.RECT()

            if not user32.GetWindowRect(
                hwnd,
                ctypes.byref(window_rect),
            ):
                raise ctypes.WinError()  # type: ignore[attr-defined]

            if not user32.GetClientRect(
                hwnd,
                ctypes.byref(client_rect),
            ):
                raise ctypes.WinError()  # type: ignore[attr-defined]

            client_top_left = wintypes.POINT(
                client_rect.left,
                client_rect.top,
            )

            client_bottom_right = wintypes.POINT(
                client_rect.right,
                client_rect.bottom,
            )

            user32.ClientToScreen(
                hwnd,
                ctypes.byref(client_top_left),
            )

            user32.ClientToScreen(
                hwnd,
                ctypes.byref(client_bottom_right),
            )

            left_nonclient = max(
                0,
                client_top_left.x - window_rect.left,
            )

            top_nonclient = max(
                0,
                client_top_left.y - window_rect.top,
            )

            right_nonclient = max(
                0,
                window_rect.right - client_bottom_right.x,
            )

            bottom_nonclient = max(
                0,
                window_rect.bottom - client_bottom_right.y,
            )

            horizontal_nonclient = (
                left_nonclient
                + right_nonclient
            )

            vertical_nonclient = (
                top_nonclient
                + bottom_nonclient
            )

            window_width, max_client_width = calculate_client_width(
                work_width,
                horizontal_nonclient,
            )

            window_height = max(
                1,
                work_height - vertical_nonclient,
            )

            tk.Wm.minsize(
                self,
                min(1150, max_client_width),
                min(650, window_height),
            )

            outer_width = (
                window_width
                + horizontal_nonclient
            )

            outer_height = (
                window_height
                + vertical_nonclient
            )

            if not user32.SetWindowPos(
                hwnd,
                None,
                work_x,
                work_y,
                outer_width,
                outer_height,
                SWP_NOZORDER
                | SWP_NOACTIVATE
                | SWP_FRAMECHANGED,
            ):
                raise ctypes.WinError()  # type: ignore[attr-defined]

            self.update_idletasks()

            return

        # =========================================================
        # WSLg
        # =========================================================
        if is_wsl():
            set_normal_state()


            reported_width = self.winfo_screenwidth()
            reported_height = self.winfo_screenheight()

            screen_dimensions_are_plausible = (
                800 <= reported_width <= 3440
                and 600 <= reported_height <= 2160
            )

            taskbar_height = 48
            window_nonclient_height = 40

            if screen_dimensions_are_plausible:
                window_width = min(
                    int(1720*.8),
                    reported_width,
                )

                window_height = max(
                    1,
                    reported_height
                    - taskbar_height
                    - window_nonclient_height,
                )
            else:
                window_width = 1080
                window_height = 1200

            x_position = 0
            y_position = 0

            tk.Wm.minsize(
                self,
                min(720, window_width),
                min(960, window_height),
            )

            tk.Wm.geometry(
                self,
                (
                    f"{window_width}x{window_height}"
                    f"+{x_position}+{y_position}"
                ),
            )

            self.update_idletasks()
            self.update()
            self.lift()

            print(
                "WSLg reported screen: "
                f"{reported_width}x{reported_height}"
            )
            print(
                "Reserved vertical space: "
                f"{taskbar_height + window_nonclient_height}px"
            )
            print(
                "Client geometry: "
                f"{window_width}x{window_height}"
                f"+{x_position}+{y_position}"
            )

            return

        # =========================================================
        # Native Linux / macOS
        # =========================================================
        set_normal_state()

        normal_geometry = self.wm_geometry()
        normal_width = self.winfo_width()
        normal_height = self.winfo_height()

        work_x = self.winfo_vrootx()
        work_y = self.winfo_vrooty()
        work_width = self.winfo_vrootwidth()
        work_height = self.winfo_vrootheight()

        # Fallback for window managers that do not provide a
        # meaningful virtual-root size.
        if work_width < 600 or work_height < 400:
            work_x = 0
            work_y = 0
            work_width = self.winfo_screenwidth()
            work_height = self.winfo_screenheight()

        work_area_from_maximized_window = False

        if set_zoomed_state():
            maximized_width = self.winfo_width()
            maximized_height = self.winfo_height()

            maximized_x = self.winfo_x()
            maximized_y = self.winfo_y()

            maximized_left_border = max(
                0,
                self.winfo_rootx() - maximized_x,
            )

            maximized_top_border = max(
                0,
                self.winfo_rooty() - maximized_y,
            )

            # Tk does not expose the complete frame geometry on all
            # Linux window managers. Side and bottom borders are
            # therefore estimated from the measured left border.
            maximized_right_border = maximized_left_border
            maximized_bottom_border = maximized_left_border

            maximized_geometry_is_plausible = (
                maximized_width >= normal_width
                and maximized_height >= normal_height
                and maximized_width >= 600
                and maximized_height >= 400
            )

            if maximized_geometry_is_plausible:
                work_x = maximized_x
                work_y = maximized_y

                work_width = (
                    maximized_width
                    + maximized_left_border
                    + maximized_right_border
                )

                work_height = (
                    maximized_height
                    + maximized_top_border
                    + maximized_bottom_border
                )

                work_area_from_maximized_window = True

            set_normal_state()

            try:
                tk.Wm.geometry(
                    self,
                    normal_geometry,
                )
            except tk.TclError:
                pass

            self.update_idletasks()
            self.update()

        normal_x = self.winfo_x()
        normal_y = self.winfo_y()

        left_nonclient = max(
            0,
            self.winfo_rootx() - normal_x,
        )

        top_nonclient = max(
            0,
            self.winfo_rooty() - normal_y,
        )

        right_nonclient = left_nonclient
        bottom_nonclient = left_nonclient

        horizontal_nonclient = (
            left_nonclient
            + right_nonclient
        )

        vertical_nonclient = (
            top_nonclient
            + bottom_nonclient
        )

        window_width, max_client_width = calculate_client_width(
            work_width,
            horizontal_nonclient,
        )

        window_height = max(
            1,
            work_height - vertical_nonclient,
        )

        tk.Wm.minsize(
            self,
            min(1150, max_client_width),
            min(650, window_height),
        )

        tk.Wm.geometry(
            self,
            (
                f"{window_width}x{window_height}"
                f"+{work_x}+{work_y}"
            ),
        )

        self.update_idletasks()
        self.update()
        self.lift()

        print(
            "Geometry source: "
            + (
                "maximized Linux/macOS work area"
                if work_area_from_maximized_window
                else "Tk virtual-root fallback"
            )
        )

        print(
            "Work area: "
            f"{work_width}x{work_height}"
            f"+{work_x}+{work_y}"
        )

        print(
            "Client geometry: "
            f"{window_width}x{window_height}"
            f"+{work_x}+{work_y}"
        )

    # Window artwork

    def _build_window_gradient_background(
        self,
    ) -> None:
        """Create a resizable gradient behind the complete GUI."""

        self._window_gradient_photo = None
        self._window_gradient_image_id = None

        self._window_gradient_canvas = tk.Canvas(
            self,
            bd=0,
            highlightthickness=0,
            relief="flat",
            bg="#141414",
        )
        self._window_gradient_canvas.place(
            x=0,
            y=0,
            relwidth=1,
            relheight=1,
        )

        self.bind(
            "<Configure>",
            self._on_root_configure,
            add="+",
        )

        self.after_idle(
            self._redraw_window_gradient
        )


    def _create_window_gradient_image(
        self,
        width: int,
        height: int,
    ) -> Image.Image:
        """Generate a subtle, softly offset gnuplot gradient."""

        # Render at a limited working resolution. A smooth gradient
        # does not require full-resolution calculation.
        render_width = min(width, 1000)
        render_height = min(height, 700)

        x = np.linspace(
            -1.0,
            1.0,
            render_width,
        )
        y = np.linspace(
            -1.0,
            1.0,
            render_height,
        )
        xx, yy = np.meshgrid(x, y)

        # Move the glow slightly toward the upper-left area.
        radius = np.sqrt(
            ((xx + 0.25) / 1.25) ** 2
            + ((yy - 0.45) / 1.05) ** 2
        )

        falloff = 1 - np.clip(
            radius,
            0.0,
            1.0,
        )

        # Only use a very dark part of the gnuplot colormap.
        start_grad = 0
        end_grad = 0.4
        gradient = (
            start_grad
            + (end_grad - start_grad) * falloff
        )


        rgba = (
            matplotlib.colormaps["gist_gray"](
                gradient
            )
            * 255
        ).astype(np.uint8)

        image = Image.fromarray(
            rgba,
            mode="RGBA",
        )

        if (
            render_width != width
            or render_height != height
        ):
            image = image.resize(
                (width, height),
                Image.Resampling.BICUBIC,
            )

        return image


    def _redraw_window_gradient(self) -> None:
        """Redraw the background at the current window size."""

        if not self.winfo_exists():
            return

        width = self.winfo_width()
        height = self.winfo_height()

        if width < 10 or height < 10:
            return

        image = self._create_window_gradient_image(
            width,
            height,
        )

        self._window_gradient_photo = (
            ImageTk.PhotoImage(image)
        )

        if self._window_gradient_image_id is None:
            self._window_gradient_image_id = (
                self._window_gradient_canvas.create_image(
                    0,
                    0,
                    anchor="nw",
                    image=self._window_gradient_photo,
                )
            )
        else:
            self._window_gradient_canvas.itemconfigure(
                self._window_gradient_image_id,
                image=self._window_gradient_photo,
            )

    def _create_title_image(self):
        dark_style()
        text_path = TextPath((0, 0), APPLICATION_TITLE, size=1, prop=FontProperties(family=self.gui_font_family, weight="bold"))
        bbox = text_path.get_extents()
        main_transform = Affine2D().translate(-bbox.x0, -bbox.y0)
        text_bbox = main_transform.transform_bbox(bbox)

        offset = 0.01 * text_bbox.height
        padding = 0.01 * text_bbox.height
        white_transform = (
            Affine2D()
            .translate(-bbox.x0, -bbox.y0)
            .translate(-offset, offset)
        )
        black_transform = (
            Affine2D()
            .translate(-bbox.x0, -bbox.y0)
            .translate(offset, -offset)
        )

        x_min = text_bbox.x0 - offset - padding
        x_max = text_bbox.x1 + offset + padding
        y_min = text_bbox.y0 - offset - padding
        y_max = text_bbox.y1 + offset + padding
        aspect_ratio = (x_max - x_min) / (y_max - y_min)

        figure_width = 10
        fig, ax = plt.subplots(
            figsize=(figure_width, figure_width / aspect_ratio),
            dpi=200,
        )
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

        ax.add_patch(
            PathPatch(
                text_path,
                transform=black_transform + ax.transData,
                facecolor="black",
                edgecolor="none",
                zorder=1,
            )
        )
        ax.add_patch(
            PathPatch(
                text_path,
                transform=white_transform + ax.transData,
                facecolor="white",
                edgecolor="none",
                zorder=2,
            )
        )

        clip_patch = PathPatch(
            text_path,
            transform=main_transform + ax.transData,
            facecolor="none",
            edgecolor="none",
        )
        ax.add_patch(clip_patch)

        x = np.linspace(-1, 1, 2000)
        y = np.linspace(-1, 1, 300)
        xx, yy = np.meshgrid(x, y)
        radius = np.sqrt(xx**2 + (yy / 2.5) ** 2)
        gradient = 0.35 * (1 - np.clip(radius, 0, 1))

        image = ax.imshow(
            gradient,
            extent=(text_bbox.x0, text_bbox.x1, text_bbox.y0, text_bbox.y1),
            cmap="gnuplot",
            vmin=0,
            vmax=1,
            origin="lower",
            aspect="auto",
            interpolation="bicubic",
            zorder=3,
        )
        image.set_clip_path(clip_patch)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.axis("off")

        buffer = BytesIO()
        fig.savefig(
            buffer,
            format="png",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)
        buffer.seek(0)
        pil_image = Image.open(buffer).convert("RGBA").copy()
        buffer.close()

        width = UI["title_image_width"]
        return ctk.CTkImage(
            light_image=pil_image,
            dark_image=pil_image,
            size=(width, int(width / aspect_ratio)),
        )

    # UI settings and session persistence

    def apply_ui_settings(
        self,
        zoom_percent: int,
        text_size_percent: int,
    ) -> None:
        valid_values = range(50, 151, 10)
        if zoom_percent not in valid_values or text_size_percent not in valid_values:
            raise ValueError("Settings percentages must be between 50 and 150.")

        loading_token = self.begin_loading("Applying interface settings…")
        try:
            self.zoom_percent = zoom_percent
            self.text_size_percent = text_size_percent
            ctk.set_widget_scaling(
                USER_SCALE_FACTOR * zoom_percent / 100
            )
            set_text_size_multiplier(text_size_percent / 100)
            self._refresh_native_menu_font()
            self._refresh_loading_overlay_layout()

            self.tabview.refresh_text_layout()
            for tab in self.tabs.values():
                tab.content_tabs.refresh_text_layout()
                tab.refresh_open_viewer_section_layout()
                tab.refresh_open_viewer_plot_typography()
                tab.refresh_calibration_plot_layout()
            self.update_idletasks()

            self.update()
            self.tabview._reflow_buttons()
            for tab in self.tabs.values():
                tab.content_tabs._reflow_buttons()
                tab.calibration_plot_tabs._reflow_buttons()
                tab.refresh_open_viewer_section_layout()
                tab.refresh_open_viewer_plot_typography()
                tab.refresh_calibration_plot_layout()
            self.update_idletasks()
            self._redraw_window_gradient()
            self.update_idletasks()
        finally:
            self.end_loading(loading_token)

    @staticmethod
    def _json_default(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot serialize {type(value).__name__} to JSON")

    def _build_session_document(
        self,
        *,
        session_name: str | None = None,
    ) -> dict:
        document_session_name = (
            str(session_name).strip()
            if session_name is not None
            else self.session_name
        ) or UNTITLED_SESSION_NAME
        return {
            "schema": "mev-viewer-session",
            "version": 1,
            "application": {
                "session_name": document_session_name,
                "zoom_percent": int(self.zoom_percent),
                "text_size_percent": int(self.text_size_percent),
                "selected_dataset": self.tabview.get(),
                "search_root_history": list(self.search_root_history),
                "spec_search_root_history": list(self.spec_search_root_history),
                "global_viewer_settings_configured": bool(
                    self._global_viewer_settings_configured
                ),
                "global_viewer_settings_values": copy.deepcopy(
                    self._global_viewer_settings_values
                ),
                "global_viewer_settings_enabled": [
                    [section, key]
                    for section, key in sorted(self._global_viewer_settings_enabled)
                ],
            },
            "datasets": [
                {
                    "name": name,
                    **tab.export_session_state(),
                }
                for name, tab in self.tabs.items()
            ],
        }

    @staticmethod
    def _validate_session_document(document: object) -> dict:
        if not isinstance(document, dict):
            raise ValueError("The selected file does not contain a session object.")
        if document.get("schema") != "mev-viewer-session":
            raise ValueError("The selected file is not a meV Viewer session.")
        version = document.get("version")
        if version != 1:
            raise ValueError(f"Unsupported session version: {version!r}")
        datasets = document.get("datasets")
        if not isinstance(datasets, list) or not datasets:
            raise ValueError("The session does not contain any datasets.")
        dataset_names: set[str] = set()
        for index, dataset in enumerate(datasets, start=1):
            if not isinstance(dataset, dict):
                raise ValueError(f"Dataset {index} is not a valid object.")
            name = dataset.get("name")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Dataset {index} has no valid name.")
            if name in dataset_names:
                raise ValueError(f"Duplicate dataset name: {name!r}")
            dataset_names.add(name)
        return document

    def _resolve_portable_session_roots(self, document: dict) -> dict | None:
        """Map missing scan/SPEC roots before replacing the current session."""

        portable_document = json.loads(
            json.dumps(document, ensure_ascii=False)
        )
        missing_roots: dict[str, list[str]] = {}
        missing_spec_files: dict[str, list[str]] = {}

        root_specs = (
            ("search_root", "Scan Files Search Root"),
            ("spec_search_root", "Spec File Search Root"),
        )
        for dataset in portable_document["datasets"]:
            setup = dataset.get("setup", {})
            if not isinstance(setup, dict):
                continue

            dataset_name = str(dataset["name"])
            for key, label in root_specs:
                if key == "spec_search_root" and str(
                    setup.get("spec_file", "")
                ).strip():
                    # A direct SPEC file makes the saved search root irrelevant.
                    continue
                original_value = str(setup.get(key, "")).strip()
                if not original_value:
                    continue

                original_path = Path(original_value).expanduser()
                if original_path.exists() and original_path.is_dir():
                    continue

                missing_roots.setdefault(original_value, []).append(
                    f"{dataset_name} — {label}"
                )

            original_spec_file = str(setup.get("spec_file", "")).strip()
            if original_spec_file:
                spec_path = Path(original_spec_file).expanduser()
                if not (spec_path.exists() and spec_path.is_file()):
                    missing_spec_files.setdefault(
                        original_spec_file,
                        [],
                    ).append(f"{dataset_name} - Spec File")

        if not missing_roots and not missing_spec_files:
            return portable_document

        missing_paths = [
            *missing_roots.items(),
            *missing_spec_files.items(),
        ]
        missing_summary = "\n\n".join(
            f"{old_path}\n  " + "\n  ".join(contexts)
            for old_path, contexts in missing_paths
        )
        self.show_styled_message(
            title="Session paths not found",
            message=(
                "Some saved folders or SPEC files do not exist on this "
                "machine. Please select replacements."
            ),
            detail=missing_summary,
            kind="warning",
        )

        root_replacements: dict[str, str] = {}
        for old_path, contexts in missing_roots.items():
            context_text = ", ".join(contexts)
            replacement = filedialog.askdirectory(
                parent=self,
                title=f"Select replacement for {context_text}",
                mustexist=True,
            )
            if not replacement:
                self.show_styled_message(
                    title="Session load cancelled",
                    message=(
                        "No session data was changed because path mapping "
                        "was cancelled."
                    ),
                    kind="info",
                )
                return None

            replacement_path = Path(replacement).expanduser().resolve()
            if not replacement_path.is_dir():
                raise ValueError(
                    f"Selected replacement is not a folder:\n{replacement_path}"
                )
            root_replacements[old_path] = str(replacement_path)

        spec_file_replacements: dict[str, str] = {}
        for old_path, contexts in missing_spec_files.items():
            context_text = ", ".join(contexts)
            replacement = filedialog.askopenfilename(
                parent=self,
                title=f"Select replacement for {context_text}",
                filetypes=(
                    ("SPEC files", "*.spec"),
                    ("All files", "*.*"),
                ),
            )
            if not replacement:
                self.show_styled_message(
                    title="Session load cancelled",
                    message=(
                        "No session data was changed because path mapping "
                        "was cancelled."
                    ),
                    kind="info",
                )
                return None

            replacement_path = Path(replacement).expanduser().resolve()
            if not replacement_path.is_file():
                raise ValueError(
                    f"Selected replacement is not a file:\n{replacement_path}"
                )
            spec_file_replacements[old_path] = str(replacement_path)

        for dataset in portable_document["datasets"]:
            setup = dataset.get("setup", {})
            if not isinstance(setup, dict):
                continue
            for key, _label in root_specs:
                original_value = str(setup.get(key, "")).strip()
                if original_value in root_replacements:
                    setup[key] = root_replacements[original_value]

            original_spec_file = str(setup.get("spec_file", "")).strip()
            if original_spec_file in spec_file_replacements:
                setup["spec_file"] = spec_file_replacements[original_spec_file]

        application_state = portable_document.get("application", {})
        if isinstance(application_state, dict):
            for history_key in (
                "search_root_history",
                "spec_search_root_history",
            ):
                history = application_state.get(history_key)
                if isinstance(history, list):
                    application_state[history_key] = [
                        root_replacements.get(str(value), str(value))
                        for value in history
                    ]

        return portable_document

    def _save_current_session(
        self,
        *,
        show_confirmation: bool = True,
    ) -> bool:
        output_path = filedialog.asksaveasfilename(
            parent=self,
            title="Save Session",
            defaultextension=".mevsession",
            initialfile=f"{self.session_name}.mevsession",
            filetypes=(
                ("meV Viewer Session", "*.mevsession"),
                ("JSON text file", "*.json"),
                ("Text file", "*.txt"),
                ("All files", "*.*"),
            ),
        )
        if not output_path:
            return False

        saved_session_name = self._session_name_from_path(output_path)
        loading_token = self.begin_loading("Saving session…")
        try:
            document = self._build_session_document(
                session_name=saved_session_name,
            )
            Path(output_path).write_text(
                json.dumps(
                    document,
                    indent=2,
                    ensure_ascii=False,
                    default=self._json_default,
                )
                + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            self.end_loading(loading_token)
            self.show_styled_message(
                title="Session save error",
                message=str(exc),
                kind="error",
            )
            return False

        self.end_loading(loading_token)
        self._set_session_name(saved_session_name)
        if show_confirmation:
            self.show_styled_message(
                title="Session saved",
                message=f"Session saved to:\n{output_path}",
                kind="info",
            )
        return True

    def save_current_session(self) -> None:
        """Menu callback for saving the current session."""

        self._save_current_session()

    def load_current_session(self) -> None:
        input_path = filedialog.askopenfilename(
            parent=self,
            title="Load Session",
            filetypes=(
                ("meV Viewer Session", "*.mevsession"),
                ("JSON text file", "*.json"),
                ("Text file", "*.txt"),
                ("All files", "*.*"),
            ),
        )
        if not input_path:
            return

        loading_token = self.begin_loading("Reading session file…")
        try:
            raw_document = json.loads(
                Path(input_path).read_text(encoding="utf-8")
            )
            document = self._validate_session_document(raw_document)
        except Exception as exc:
            self.end_loading(loading_token)
            self.show_styled_message(
                title="Session load error",
                message=str(exc),
                kind="error",
            )
            return
        self.end_loading(loading_token)

        confirmed = self.ask_styled_confirmation(
            title="Load Session",
            message="Replace the currently open datasets with this session?",
            detail=(
                "Unsaved changes in the current tabs will be discarded."
            ),
            destructive=True,
        )
        if not confirmed:
            return

        loading_token = self.begin_loading("Checking session paths…")
        try:
            portable_document = self._resolve_portable_session_roots(document)
        except Exception as exc:
            self.end_loading(loading_token)
            self.show_styled_message(
                title="Session path error",
                message=str(exc),
                kind="error",
            )
            return
        self.end_loading(loading_token)
        if portable_document is None:
            return

        loading_token = self.begin_loading("Restoring session…")
        try:
            self._restore_session_document(portable_document)
            self._set_session_name(
                self._session_name_from_path(input_path)
            )
        except Exception as exc:
            self.end_loading(loading_token)
            self.show_styled_message(
                title="Session restore error",
                message=str(exc),
                kind="error",
            )
            return
        self.end_loading(loading_token)

        def show_session_loaded() -> None:
            self.show_styled_message(
                title="Session loaded",
                message=f"Session loaded from:\n{input_path}",
                kind="info",
            )

        self.run_after_loading(show_session_loaded)

    # Top bar and dataset tabs

    def _show_file_menu(self) -> None:
        x_position = self.file_button.winfo_rootx()
        y_position = self.file_button.winfo_rooty() + self.file_button.winfo_height()
        try:
            self.file_menu.tk_popup(x_position, y_position)
        finally:
            self.file_menu.grab_release()

    def _refresh_native_menu_font(self) -> None:
        """Scale the native Tk menu like the surrounding CustomTkinter UI."""

        file_menu_font = getattr(self, "file_menu_font", None)
        if file_menu_font is None:
            return

        combined_scale = (
            USER_SCALE_FACTOR * self.zoom_percent
            * self.text_size_percent
            / 10_000
        )
        file_menu_font.configure(
            family=self.gui_font_family,
            # Negative Tk font sizes are pixels. This matches the way
            # CustomTkinter renders its fonts and avoids a second DPI scaling.
            size=-max(1, round(UI["font_normal"] * combined_scale)),
        )

    def _selected_viewer_state(self) -> dict | None:
        selected_tab = self.tabs.get(self.tabview.get())
        if selected_tab is None or selected_tab.viewer_figure is None:
            return None
        keepalive = getattr(
            selected_tab.viewer_figure,
            "_view_spectra_keepalive",
            None,
        )
        if not isinstance(keepalive, dict):
            return None
        exporter = keepalive.get("export_session_state")
        if not callable(exporter):
            return None
        state = exporter()
        return state if isinstance(state, dict) else None

    def merge_global_viewer_settings(self, state: dict | None) -> dict | None:
        """Overlay enabled global values on a Viewer session state."""

        if not self._global_viewer_settings_configured:
            return state

        merged: dict[str, object] = (
            copy.deepcopy(state)
            if isinstance(state, dict)
            else {"version": 1}
        )
        for section, key in self._global_viewer_settings_enabled:
            section_values = self._global_viewer_settings_values.get(section, {})
            if not isinstance(section_values, dict) or key not in section_values:
                continue
            target_section = merged.setdefault(section, {})
            if isinstance(target_section, dict):
                target_section[key] = copy.deepcopy(section_values[key])
        return merged

    def apply_global_viewer_settings(self) -> int:
        """Apply enabled global values to every currently open Viewer."""

        updated_viewers = 0
        for tab in self.tabs.values():
            if tab.viewer_figure is None:
                continue
            keepalive = getattr(
                tab.viewer_figure,
                "_view_spectra_keepalive",
                None,
            )
            if not isinstance(keepalive, dict):
                continue
            exporter = keepalive.get("export_session_state")
            restorer = keepalive.get("restore_session_state")
            if not callable(exporter) or not callable(restorer):
                continue
            exported_state = exporter()
            if not isinstance(exported_state, dict):
                continue
            restorer(self.merge_global_viewer_settings(exported_state))
            updated_viewers += 1
        return updated_viewers

    @staticmethod
    def _validate_global_viewer_value(key: str, value: object) -> object:
        if not isinstance(value, (str, int, float)):
            raise ValueError(f"Unsupported value type for {key}: {type(value).__name__}")
        if key == "tilt" and not 0 <= float(value) <= 0.045:
            raise ValueError("Tilt must be between 0 and 0.045.")
        if key in {
            "lower_percentile",
            "upper_percentile",
            "local_filter_bottom_limit",
            "local_filter_upper_limit",
        } and not 0 <= float(value) <= 100:
            raise ValueError("Percentages must be between 0 and 100.")
        if key in {"colormap_start", "colormap_end"} and not 0 <= float(value) <= 1:
            raise ValueError("Colormap limits must be between 0 and 1.")
        if key in {"display_bin_x", "display_bin_y", "spectrum_bin"} and int(
            value
        ) not in {1, 2, 4, 8, 16, 32, 64}:
            raise ValueError("Binning must be one of 1, 2, 4, 8, 16, 32 or 64.")
        if key == "tilt_speedup" and not 1 <= int(value) <= 10:
            raise ValueError("Tilt speedup must be between 1 and 10.")
        if key in {"median_filter_window", "local_filter_window"}:
            return mev_viewer.normalize_median_window(int(value))
        return value

    def open_global_viewer_settings_dialog(self) -> None:
        if (
            self._global_viewer_settings_dialog is not None
            and self._global_viewer_settings_dialog.winfo_exists()
        ):
            self._global_viewer_settings_dialog.lift()
            self._global_viewer_settings_dialog.focus_force()
            return

        initial_state = copy.deepcopy(self._global_viewer_settings_values)
        if not self._global_viewer_settings_configured:
            current_state = self._selected_viewer_state()
            if current_state is not None:
                for section in ("settings", "mode", "controls"):
                    current_values = current_state.get(section, {})
                    target_values = initial_state.get(section, {})
                    if isinstance(current_values, dict) and isinstance(target_values, dict):
                        target_values.update(current_values)

        dialog = ctk.CTkToplevel(self)
        dialog.withdraw()
        dialog.title("Global Viewer Settings")
        dialog.transient(self)
        dialog.grab_set()
        dialog.geometry("760x760")
        dialog.minsize(650, 520)
        dialog.grid_rowconfigure(0, weight=1)
        dialog.grid_columnconfigure(0, weight=1)
        self._global_viewer_settings_dialog = dialog

        content = ctk.CTkScrollableFrame(dialog, fg_color="#171717")
        content.grid(row=0, column=0, sticky="nsew", padx=14, pady=(14, 8))
        content.grid_columnconfigure(1, weight=1)
        content.grid_columnconfigure(2, weight=1)

        make_label(
            content,
            "Global Viewer Settings",
            size=UI["font_section"],
            bold=True,
            anchor="w",
        ).grid(row=0, column=0, columnspan=3, sticky="ew", padx=8, pady=(4, 4))
        make_label(
            content,
            "Choose which values are applied to open Viewers. New Viewers use them only when “Apply global preferences” is enabled in Setup.",
            size=UI["font_small"],
            text_color=("gray35", "gray65"),
            anchor="w",
        ).grid(row=1, column=0, columnspan=3, sticky="ew", padx=8, pady=(0, 10))

        field_vars: dict[
            tuple[str, str],
            tuple[ctk.BooleanVar, tk.Variable, str],
        ] = {}
        select_all_var = ctk.BooleanVar(value=False)

        def set_all_selections() -> None:
            selected = bool(select_all_var.get())
            for enabled_var, _value_var, _kind in field_vars.values():
                enabled_var.set(selected)

        def sync_select_all_state() -> None:
            select_all_var.set(
                bool(field_vars)
                and all(
                    enabled_var.get()
                    for enabled_var, _value_var, _kind in field_vars.values()
                )
            )

        ctk.CTkCheckBox(
            content,
            text="Select all settings",
            variable=select_all_var,
            command=set_all_selections,
            onvalue=True,
            offvalue=False,
            font=ui_font(UI["font_normal"], family=self.gui_font_family),
        ).grid(
            row=2,
            column=0,
            columnspan=3,
            sticky="w",
            padx=8,
            pady=(0, 10),
        )

        for column, heading in enumerate(("Apply", "Setting", "Value")):
            make_label(
                content,
                heading,
                size=UI["font_small"],
                bold=True,
                text_color=("gray35", "gray65"),
                anchor="w",
            ).grid(
                row=3,
                column=column,
                sticky="ew",
                padx=(8, 4) if column == 0 else (0, 8),
                pady=(0, 4),
            )

        row = 4
        previous_group = None

        for spec in GLOBAL_VIEWER_SETTING_SPECS:
            group = str(spec["group"])
            section = str(spec["section"])
            key = str(spec["key"])
            kind = str(spec["kind"])
            if group != previous_group:
                make_label(
                    content,
                    group,
                    size=UI["font_normal"],
                    bold=True,
                    anchor="w",
                ).grid(
                    row=row,
                    column=0,
                    columnspan=3,
                    sticky="ew",
                    padx=8,
                    pady=(12 if previous_group is not None else 2, 5),
                )
                row += 1
                previous_group = group

            enabled_var = ctk.BooleanVar(
                value=(section, key) in self._global_viewer_settings_enabled
            )
            enabled_checkbox = ctk.CTkCheckBox(
                content,
                text="",
                variable=enabled_var,
                command=sync_select_all_state,
                width=26,
                checkbox_width=20,
                checkbox_height=20,
            )
            enabled_checkbox.grid(row=row, column=0, sticky="w", padx=(8, 4), pady=4)
            make_label(content, str(spec["label"]), anchor="w").grid(
                row=row,
                column=1,
                sticky="w",
                padx=(0, 12),
                pady=4,
            )

            section_values = initial_state.get(section, {})
            value = (
                section_values.get(key, spec["default"])
                if isinstance(section_values, dict)
                else spec["default"]
            )
            if kind == "bool":
                value_var: tk.Variable = ctk.BooleanVar(value=bool(value))
                value_widget = ctk.CTkCheckBox(
                    content,
                    text="Enabled",
                    variable=value_var,
                    onvalue=True,
                    offvalue=False,
                )
            elif kind == "choice":
                value_var = ctk.StringVar(value=str(value))
                options = [str(item) for item in spec.get("options", [])]
                if str(value) not in options:
                    options.append(str(value))
                value_widget = make_combobox(
                    content,
                    variable=value_var,
                    values=options,
                    family=self.gui_font_family,
                    font_size=UI["font_normal"],
                    dropdown_font_size=UI["font_normal"],
                )
            else:
                value_var = ctk.StringVar(value=str(value))
                value_widget = ctk.CTkEntry(
                    content,
                    textvariable=value_var,
                    font=ui_font(UI["font_input"], family=self.gui_font_family),
                )
            value_widget.grid(row=row, column=2, sticky="ew", padx=(0, 8), pady=4)
            field_vars[(section, key)] = (enabled_var, value_var, kind)
            row += 1

        sync_select_all_state()

        error_var = ctk.StringVar(value="")
        make_label(
            content,
            "",
            textvariable=error_var,
            size=UI["font_small"],
            text_color="#FF806F",
            anchor="w",
        ).grid(row=row, column=0, columnspan=3, sticky="ew", padx=8, pady=(8, 4))

        buttons = ctk.CTkFrame(dialog, fg_color="transparent")
        buttons.grid(row=1, column=0, pady=(4, 14))

        def close_dialog() -> None:
            try:
                dialog.grab_release()
            except tk.TclError:
                pass
            dialog.destroy()
            self._global_viewer_settings_dialog = None

        def apply_settings() -> None:
            new_values = copy.deepcopy(self._global_viewer_settings_values)
            new_enabled: set[tuple[str, str]] = set()
            try:
                for spec in GLOBAL_VIEWER_SETTING_SPECS:
                    section = str(spec["section"])
                    key = str(spec["key"])
                    enabled_var, value_var, kind = field_vars[(section, key)]
                    if enabled_var.get():
                        new_enabled.add((section, key))
                    raw_value = value_var.get()
                    if kind == "bool":
                        parsed_value: object = bool(raw_value)
                    elif kind == "int":
                        parsed_value = int(str(raw_value).strip())
                    elif kind == "float":
                        parsed_value = float(str(raw_value).strip())
                    else:
                        parsed_value = str(raw_value)
                    parsed_value = self._validate_global_viewer_value(key, parsed_value)
                    section_values = new_values[section]
                    assert isinstance(section_values, dict)
                    section_values[key] = parsed_value
                settings_values = new_values["settings"]
                controls_values = new_values["controls"]
                assert isinstance(settings_values, dict)
                assert isinstance(controls_values, dict)
                if float(settings_values["lower_percentile"]) >= float(
                    settings_values["upper_percentile"]
                ):
                    raise ValueError("Lower percentile must be below upper percentile.")
                if int(settings_values["local_filter_bottom_limit"]) >= int(
                    settings_values["local_filter_upper_limit"]
                ):
                    raise ValueError("Local lower percentile must be below the upper limit.")
                if float(controls_values["colormap_start"]) >= float(
                    controls_values["colormap_end"]
                ):
                    raise ValueError("Colormap start must be below colormap end.")
                if float(controls_values["display_vmin"]) >= float(
                    controls_values["display_vmax"]
                ):
                    raise ValueError("Display minimum must be below display maximum.")
            except (TypeError, ValueError) as exc:
                error_var.set(f"Invalid value: {exc}")
                return

            self._global_viewer_settings_values = new_values
            self._global_viewer_settings_enabled = new_enabled
            self._global_viewer_settings_configured = True
            updated = self.apply_global_viewer_settings()
            close_dialog()
            self.show_styled_message(
                title="Global Viewer Settings",
                message=(
                    f"Settings applied to {updated} open Viewer"
                    f"{'s' if updated != 1 else ''}. Future Viewers will use them as well."
                ),
                kind="info",
            )

        make_gradient_button(
            buttons,
            text="Cancel",
            width=130,
            command=close_dialog,
            **SUBTLE_GRADIENT_STYLE,
        ).grid(row=0, column=0, padx=(0, 6))
        make_gradient_button(
            buttons,
            text="Apply to All",
            width=160,
            command=apply_settings,
            bold=True,
            **SECONDARY_GRADIENT_STYLE,
        ).grid(row=0, column=1, padx=(6, 0))

        dialog.protocol("WM_DELETE_WINDOW", close_dialog)
        dialog.bind("<Escape>", lambda _event: close_dialog())
        dialog.deiconify()
        dialog.lift()
        dialog.focus_force()

    def open_settings_dialog(self) -> None:
        if (
            self._settings_dialog is not None
            and self._settings_dialog.winfo_exists()
        ):
            self._settings_dialog.lift()
            self._settings_dialog.focus_force()
            return

        dialog = ctk.CTkToplevel(self)
        dialog.withdraw()
        dialog.title("Settings")
        dialog.transient(self)
        dialog.grab_set()
        dialog.resizable(True, True)
        dialog.minsize(500,350)
        dialog.grid_columnconfigure(1, weight=1)
        self._settings_dialog = dialog

        zoom_var = ctk.StringVar(value=f"{self.zoom_percent} %")
        text_size_var = ctk.StringVar(value=f"{self.text_size_percent} %")

        make_label(
            dialog,
            "Interface Settings",
            size=UI["font_section"],
            bold=True,
            anchor="w",
        ).grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=24,
            pady=(20, 14),
        )

        make_label(dialog, "Zoom", anchor="w").grid(
            row=1,
            column=0,
            sticky="w",
            padx=(24, 18),
            pady=7,
        )
        make_combobox(
            dialog,
            variable=zoom_var,
            values=self.settings_percent_values,
            family=self.gui_font_family,
            font_size=UI["font_normal"],
            dropdown_font_size=UI["font_normal"],
        ).grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(0, 24),
            pady=7,
        )

        make_label(dialog, "Text size", anchor="w").grid(
            row=2,
            column=0,
            sticky="w",
            padx=(24, 18),
            pady=7,
        )
        make_combobox(
            dialog,
            variable=text_size_var,
            values=self.settings_percent_values,
            family=self.gui_font_family,
            font_size=UI["font_normal"],
            dropdown_font_size=UI["font_normal"],
        ).grid(
            row=2,
            column=1,
            sticky="ew",
            padx=(0, 24),
            pady=7,
        )

        make_label(
            dialog,
            "100 % uses the current interface defaults.",
            size=UI["font_small"],
            text_color=("gray35", "gray65"),
            anchor="w",
        ).grid(
            row=3,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=24,
            pady=(8, 14),
        )

        button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        button_frame.grid(
            row=4,
            column=0,
            columnspan=2,
            pady=(0, 20),
        )

        def close_dialog() -> None:
            try:
                dialog.grab_release()
            except tk.TclError:
                pass
            dialog.destroy()
            self._settings_dialog = None

        def apply_settings() -> None:
            selected_zoom = int(zoom_var.get().replace("%", "").strip())
            selected_text_size = int(
                text_size_var.get().replace("%", "").strip()
            )
            close_dialog()
            self.apply_ui_settings(selected_zoom, selected_text_size)

        make_gradient_button(
            button_frame,
            text="Cancel",
            width=120,
            command=close_dialog,
            **SUBTLE_GRADIENT_STYLE,
        ).grid(row=0, column=0, padx=(0, 6))
        make_gradient_button(
            button_frame,
            text="Apply",
            width=120,
            command=apply_settings,
            bold=True,
            **SECONDARY_GRADIENT_STYLE,
        ).grid(row=0, column=1, padx=(6, 0))

        dialog.protocol("WM_DELETE_WINDOW", close_dialog)
        dialog.bind("<Escape>", lambda _event: close_dialog())
        dialog.bind("<Return>", lambda _event: apply_settings())
        dialog.update_idletasks()

        width = dialog.winfo_reqwidth()
        height = dialog.winfo_reqheight()
        x = self.winfo_rootx() + (self.winfo_width() - width) // 2
        y = self.winfo_rooty() + (self.winfo_height() - height) // 2
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        dialog.deiconify()
        dialog.lift()
        dialog.focus_force()

    def _build_top_bar(self) -> None:
        self.top_bar = ctk.CTkFrame(self, fg_color="transparent")
        self.top_bar.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=UI["outer_padding"],
            pady=(16, 8),
        )
        self.top_bar.grid_columnconfigure(0, weight=1)

        self.file_menu_font = tkfont.Font(
            root=self,
            family=self.gui_font_family,
            size=UI["font_normal"],
        )
        self._refresh_native_menu_font()

        self.file_menu = tk.Menu(
            self,
            tearoff=False,
            bg="#1B1B1F",
            fg="#F4F4F6",
            activebackground="#4800C7",
            activeforeground="#FFFFFF",
            bd=1,
            relief="solid",
            font=self.file_menu_font,
        )
        self.file_menu.add_command(
            label="Settings",
            command=self.open_settings_dialog,
            font=ui_font(UI["font_normal"])
        )
        self.file_menu.add_command(
            label="Global Viewer Settings",
            command=self.open_global_viewer_settings_dialog,
            font=ui_font(UI["font_normal"]),
        )
        self.file_menu.add_separator()
        self.file_menu.add_command(
            label="Save Session",
            command=self.save_current_session,
            font=ui_font(UI["font_normal"])
        )
        self.file_menu.add_command(
            label="Load Session",
            command=self.load_current_session,
            font=ui_font(UI["font_normal"])
        )

        self.file_button = ctk.CTkButton(
            self.top_bar,
            text="File  ▾",
            command=self._show_file_menu,
            width=96,
            height=28,
            fg_color="transparent",
            hover=False,
            border_width=0,
            corner_radius=0,
            text_color=("gray25", "gray75"),
            font=ui_font(UI["font_normal"]),
            anchor="w",
        )
        self.file_button.grid(
            row=0,
            column=1,
            sticky="e",
            padx=(8, 0),
            pady=10,
        )

        self.title_image = self._create_title_image()
        self.title_label = ctk.CTkLabel(
            self.top_bar,
            text="",
            image=self.title_image,
            fg_color="transparent",
        )
        self.title_label.grid(row=0, column=0, sticky="w", padx=12, pady=10)

    def _build_tabs(self) -> None:
        self.tabview = ButtonTabView(
            self,
            button_width=160,
            max_button_width_factor=None,
            button_text_padding=0,
            button_height=UI["button_height"],
            button_gap=6,
            button_corner_radius=10,
            button_border_width=1,
            overflow_width=110,
            overflow_text="More...",
            wrap_buttons=True,
            reorderable=True,
            reorder_command=self._on_dataset_tabs_reordered,
            add_command=self.add_plot_tab,
            close_command=self.remove_dataset_tab,
            close_button_width=34,
            close_font=ui_font(UI["font_tab"] + 4, bold=True),
            rename_command=self.rename_selected_tab,
            rename_button_width=30,
            rename_font=ui_font(UI["font_tab"] + 3, bold=True),

            # Colors for the overflow menu.
            selected_color=("#6236D9", "#4800C7"),
            selected_hover_color=("#7A46E8", "#6624E8"),
            selected_border_color=BUTTON_BORDER_COLOR,
            selected_text_color="white",
            unselected_color=("gray90", "gray14"),
            unselected_hover_color=("#CEC2E9", "#382653"),
            unselected_border_color=("gray75", "gray30"),
            unselected_text_color=("black", "white"),

            # Gradient styles for the visible Dataset tabs.
            selected_gradient_style={
                **TAB_SELECTED_GRADIENT_STYLE,
            },
            unselected_gradient_style={
                **TAB_UNSELECTED_GRADIENT_STYLE,
            },

            font=ui_font(
                UI["font_dataset_tab"],
                bold=True,
            ),
        )
        self.tabview.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=UI["outer_padding"],
            pady=(0, 16),
        )

    def _on_dataset_tabs_reordered(self, ordered_names: list[str]) -> None:
        """Keep application iteration and session saving in visual tab order."""

        self.tabs = {
            name: self.tabs[name]
            for name in ordered_names
            if name in self.tabs
        }

    def get_tab_name_for_tab(self, tab: SpectraTab) -> str:
        return next(
            (name for name, tab_object in self.tabs.items() if tab_object is tab),
            self.tabview.get(),
        )

    def rename_selected_tab(self, old_name: str | None = None) -> None:
        if old_name is None:
            old_name = self.tabview.get()
        if old_name not in self.tabs:
            return

        new_name = ask_dataset_name(
            self,
            title="Rename Dataset",
            prompt="Enter a new name for this dataset:",
            initialvalue=old_name,
            existing_names=set(self.tabs),
            old_name=old_name,
        )
        if new_name is None:
            return

        tab = self.tabs.pop(old_name)
        self.tabview.rename(old_name, new_name)
        if self.first_tab_name == old_name:
            self.first_tab_name = new_name
        self.tabs[new_name] = tab
        self._on_dataset_tabs_reordered(self.tabview.ordered_names())
        self.tabview.set(new_name)
        tab.update_open_viewer_title(new_name)

    def remove_dataset_tab(self, selected_tab: str | None = None) -> None:
        if len(self.tabs) <= 1:
            return

        if selected_tab is None:
            selected_tab = self.tabview.get()
        if selected_tab not in self.tabs:
            return

        confirmed = self.ask_styled_confirmation(
            title="Close dataset",
            message=f'Do you really want to close "{selected_tab}"?',
            detail=(
                "The current viewer and all unsaved settings of this tab "
                "will be discarded."
            ),
            destructive=True,
        )
        if not confirmed:
            return

        self.tabview.delete(selected_tab)
        self.tabs.pop(selected_tab, None)
        if self.first_tab_name == selected_tab:
            self.first_tab_name = next(iter(self.tabs), None)

    def _inherit_search_roots(self, tab_name: str, tab: SpectraTab) -> None:
        if (
            self.first_tab_name is None
            or tab_name == self.first_tab_name
            or self.first_tab_name not in self.tabs
        ):
            return

        first_tab = self.tabs[self.first_tab_name]
        inherit_specs = (
            (
                first_tab.search_root_var,
                tab.search_root_var,
                tab.search_root_combo,
                self.search_root_history,
            ),
            (
                first_tab.spec_search_root_var,
                tab.spec_search_root_var,
                tab.spec_search_root_combo,
                self.spec_search_root_history,
            ),
        )

        for source_var, target_var, combo, history in inherit_specs:
            value = source_var.get().strip()
            if not value:
                continue
            target_var.set(value)
            tab._add_to_combo_history(
                combo=combo,
                history=history,
                path_str=value,
            )

        spec_file = first_tab.spec_file_var.get().strip()
        if spec_file:
            tab.spec_file_var.set(spec_file)

    def _create_dataset_tab(
        self,
        tab_name: str,
        *,
        inherit_search_roots: bool,
    ) -> SpectraTab:
        is_initial_tab = not self.tabs
        self.tabview.add(tab_name, defer_reflow=not is_initial_tab)

        tab = SpectraTab(self.tabview.tab(tab_name), app=self)
        tab.pack(fill="both", expand=True)
        if inherit_search_roots:
            self._inherit_search_roots(tab_name, tab)

        # Complete all pending geometry and CustomTkinter drawing work while
        # the previously selected dataset still covers the new tab.
        if not is_initial_tab:
            tab.update_idletasks()

        self.tabs[tab_name] = tab
        if self.first_tab_name is None:
            self.first_tab_name = tab_name
        self.tabview.set(tab_name)
        return tab

    def add_plot_tab(self) -> None:
        while True:
            self.tab_counter += 1
            tab_name = f"Dataset {self.tab_counter}"
            if tab_name not in self.tabs:
                break

        self._create_dataset_tab(
            tab_name,
            inherit_search_roots=True,
        )

    def _restore_session_document(self, document: dict) -> None:
        datasets = document["datasets"]
        application_state = document.get("application", {})
        if not isinstance(application_state, dict):
            application_state = {}

        for name in list(self.tabs):
            self.tabview.delete(name)
        self.tabs.clear()
        self.first_tab_name = None
        self.tab_counter = 0

        search_history = application_state.get("search_root_history", [])
        spec_history = application_state.get("spec_search_root_history", [])
        self.search_root_history = (
            [str(value) for value in search_history]
            if isinstance(search_history, list)
            else []
        )
        self.spec_search_root_history = (
            [str(value) for value in spec_history]
            if isinstance(spec_history, list)
            else []
        )

        configured = application_state.get(
            "global_viewer_settings_configured",
            False,
        )
        saved_global_values = application_state.get(
            "global_viewer_settings_values",
            {},
        )
        if isinstance(saved_global_values, dict):
            for section in ("settings", "mode", "controls"):
                saved_section = saved_global_values.get(section, {})
                current_section = self._global_viewer_settings_values.get(section, {})
                if isinstance(saved_section, dict) and isinstance(current_section, dict):
                    current_section.update(copy.deepcopy(saved_section))
        saved_enabled = application_state.get(
            "global_viewer_settings_enabled",
            [],
        )
        if isinstance(saved_enabled, list):
            restored_enabled = {
                (str(item[0]), str(item[1]))
                for item in saved_enabled
                if isinstance(item, list) and len(item) == 2
            }
            if restored_enabled or configured:
                self._global_viewer_settings_enabled = restored_enabled
        self._global_viewer_settings_configured = bool(configured)

        restored_tabs: list[tuple[SpectraTab, dict]] = []
        for dataset_state in datasets:
            tab_name = str(dataset_state["name"])
            tab = self._create_dataset_tab(
                tab_name,
                inherit_search_roots=False,
            )
            tab.restore_setup_session_state(dataset_state)
            tab.search_root_combo.configure(values=self.search_root_history)
            tab.spec_search_root_combo.configure(values=self.spec_search_root_history)
            restored_tabs.append((tab, dataset_state))

        self.tab_counter = max(len(self.tabs), self.tab_counter)

        requested_zoom = int(
            application_state.get("zoom_percent", self.zoom_percent)
        )
        requested_text_size = int(
            application_state.get("text_size_percent", self.text_size_percent)
        )
        if requested_zoom not in range(50, 151, 10):
            requested_zoom = 100
        if requested_text_size not in range(50, 151, 10):
            requested_text_size = 100
        self.apply_ui_settings(requested_zoom, requested_text_size)

        selected_dataset = str(
            application_state.get("selected_dataset", next(iter(self.tabs)))
        )
        if selected_dataset not in self.tabs:
            selected_dataset = next(iter(self.tabs))
        self.tabview.set(selected_dataset)

        for tab, dataset_state in restored_tabs:
            tab.restore_session_metadata()
            tab.restore_calibration_plots_from_session()

        for tab, dataset_state in restored_tabs:
            tab.start_viewer_from_session(dataset_state)

        self.tabview.set(selected_dataset)

    # Dialogs and shutdown

    def _show_styled_dialog(
        self,
        *,
        title: str,
        heading: str,
        message: str,
        detail: str = "",
        kind: Literal["info", "warning", "error", "question"] = "info",
        actions: (
            tuple[
                tuple[str, str, Literal["subtle", "primary", "danger"]],
                ...,
            ]
            | None
        ) = None,
        cancel_result: str = "ok",
    ) -> str:
        """Show one consistently styled, text-size-aware modal dialog."""

        if actions is None:
                actions = (("OK", "ok", "primary"),)

        dialog = ctk.CTkToplevel(self)
        dialog.withdraw()
        dialog.title(title)
        dialog.transient(self)
        dialog.resizable(False, False)
        dialog.configure(fg_color="#16181D")
        dialog.grid_columnconfigure(0, weight=1)

        result = cancel_result
        text_scale = self.text_size_percent / 100
        title_height = max(
            46,
            round(UI["font_section"] * 1.8 * text_scale),
        )
        message_line_count = max(
            message.count("\n") + 1,
            (len(message) + 69) // 70,
        )
        message_height = max(
            46,
            round(
                UI["font_normal"]
                * 1.9
                * message_line_count
                * text_scale
            ),
        )
        detail_line_count = max(
            detail.count("\n") + 1,
            (len(detail) + 79) // 80,
        )
        detail_height = max(
            44,
            round(
                UI["font_small"]
                * 1.8
                * detail_line_count
                * text_scale
            ),
        )
        button_height = max(
            UI["button_height"],
            round(UI["font_normal"] * 1.8 * text_scale),
        )

        def finish(action: str) -> None:
            nonlocal result
            result = action
            try:
                dialog.grab_release()
            except tk.TclError:
                pass
            dialog.destroy()

        make_label(
            dialog,
            heading,
            size=UI["font_section"],
            family=self.gui_font_family,
            bold=True,
            text_color={
                "info": "#A996FF",
                "warning": "#F3C969",
                "error": "#FF7F8F",
                "question": "#F4F6FA",
            }[kind],
            height=title_height,
        ).grid(
            row=0,
            column=0,
            padx=32,
            pady=(28, 10),
        )

        make_label(
            dialog,
            message,
            size=UI["font_normal"],
            family=self.gui_font_family,
            text_color="#E2E6ED",
            justify="center",
            wraplength=680,
            height=message_height,
        ).grid(
            row=1,
            column=0,
            padx=32,
            pady=(0, 8 if detail else 24),
        )

        button_row = 2
        if detail:
            make_label(
                dialog,
                detail,
                size=UI["font_small"],
                family=self.gui_font_family,
                text_color="#AEB7C6",
                justify="center",
                wraplength=680,
                height=detail_height,
            ).grid(
                row=2,
                column=0,
                padx=32,
                pady=(0, 24),
            )
            button_row = 3

        button_frame = ctk.CTkFrame(dialog, fg_color="transparent")
        button_frame.grid(
            row=button_row,
            column=0,
            padx=28,
            pady=(0, 28),
        )

        style_options = {
            "subtle": SUBTLE_GRADIENT_STYLE,
            "primary": PRIMARY_GRADIENT_STYLE,
            "danger": DANGER_GRADIENT_STYLE,
        }
        for column, (label, action, style_name) in enumerate(actions):
            button_width = max(
                110,
                round((42 + len(label) * 9) * text_scale),
            )
            make_gradient_button(
                button_frame,
                text=label,
                width=button_width,
                height=button_height,
                command=lambda value=action: finish(value),
                bold=style_name == "primary",
                **style_options[style_name],
            ).grid(
                row=0,
                column=column,
                padx=(0 if column == 0 else 6, 0 if column == len(actions) - 1 else 6),
            )

        dialog.protocol("WM_DELETE_WINDOW", lambda: finish(cancel_result))
        dialog.bind("<Escape>", lambda _event: finish(cancel_result))
        dialog.update_idletasks()

        width = max(720, dialog.winfo_reqwidth())
        height = max(260, dialog.winfo_reqheight())
        x = self.winfo_rootx() + (self.winfo_width() - width) // 2
        y = self.winfo_rooty() + (self.winfo_height() - height) // 2
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        dialog.deiconify()
        dialog.lift()
        dialog.grab_set()
        dialog.focus_force()
        self.wait_window(dialog)
        return result

    def show_styled_message(
        self,
        *,
        title: str,
        message: str,
        detail: str = "",
        kind: Literal["info", "warning", "error"] = "info",
    ) -> None:
        """Show a styled informational dialog with one OK button."""

        button_style: Literal["primary", "danger"] = (
            "danger" if kind == "error" else "primary"
        )
        self._show_styled_dialog(
            title=title,
            heading=title,
            message=message,
            detail=detail,
            kind=kind,
            actions=(("OK", "ok", button_style),),
            cancel_result="ok",
        )

    def ask_styled_confirmation(
        self,
        *,
        title: str,
        message: str,
        detail: str = "",
        destructive: bool = False,
    ) -> bool:
        """Show a styled Yes/No confirmation dialog."""

        yes_style: Literal["primary", "danger"] = (
            "danger" if destructive else "primary"
        )
        result = self._show_styled_dialog(
            title=title,
            heading=title,
            message=message,
            detail=detail,
            kind="question",
            actions=(
                ("No", "no", "subtle"),
                ("Yes", "yes", yes_style),
            ),
            cancel_result="no",
        )
        return result == "yes"

    def _ask_close_action(
        self,
    ) -> Literal["cancel", "close", "save_and_close"]:
        """Ask whether to cancel, close directly, or save before closing."""

        result = self._show_styled_dialog(
            title="Close application",
            heading="Close application?",
            message="All open dataset tabs and unsaved settings will be discarded.",
            kind="question",
            actions=(
                ("Cancel", "cancel", "subtle"),
                ("Close", "close", "danger"),
                ("Save and Close", "save_and_close", "primary"),
            ),
            cancel_result="cancel",
        )
        if result == "close":
            return "close"
        if result == "save_and_close":
            return "save_and_close"
        return "cancel"

    def close_application(self) -> None:
        close_action = self._ask_close_action()
        if close_action == "cancel":
            return
        if close_action == "save_and_close" and not self._save_current_session(
            show_confirmation=False
        ):
            return

        for tab in self.tabs.values():
            figure = getattr(tab, "viewer_figure", None)
            if figure is not None:
                try:
                    plt.close(figure)
                except Exception:
                    pass
        self.destroy()


# =============================================================================
# Entry point
# =============================================================================

def set_windows_app_id() -> None:
    if platform.system() == "Windows":
        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID( #type: ignore
                "mevrixs.toolkit"
            )
        except Exception:
            pass

if __name__ == "__main__":
    set_windows_app_id()
    app = SpectraLauncher()
    app.mainloop()
