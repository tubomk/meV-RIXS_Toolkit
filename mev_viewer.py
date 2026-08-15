# Interactive loading, processing, and plotting of detector spectra.

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, cast, TypedDict

import copy
import json
import os
import re

import numpy as np
import matplotlib.pyplot as plt
import customtkinter as ctk
from matplotlib import rcParams
from matplotlib.backend_bases import Event, MouseEvent
from matplotlib.colors import LinearSegmentedColormap
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.figure import Figure

from lmfit import Model, Parameters, minimize

try:
    from numba import njit, prange, set_num_threads
    # Leave two logical CPUs available for the GUI and operating system.
    numba_thread_count = max(1, (os.cpu_count() or 1) - 2)
    set_num_threads(numba_thread_count)
    print(
        "\033[32mNumba detected, using "
        f"{numba_thread_count} threads for percentile filtering.\033[0m"
    )
except ImportError:  # Median filter still imports, but will be slow without numba.
    print("\033[33mNumba not detected, local percentile filtering will be slow. Install numba for (much) faster filtering.\033[0m")
    def njit(*decorator_args, **decorator_kwargs):
        if (
            decorator_args
            and callable(decorator_args[0])
            and len(decorator_args) == 1
            and not decorator_kwargs
        ):
            return decorator_args[0]

        def decorate(func):
            return func

        return decorate

    prange = range


# =============================================================================
# Styling helpers
# =============================================================================


def _build_cross_platform_font_rcparams() -> dict:
    import matplotlib.font_manager as font_manager

    preferred_monospace_fonts = [
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

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    monospace_fallbacks = [
        font_name
        for font_name in preferred_monospace_fonts
        if font_name in available_fonts
    ]

    if not monospace_fallbacks:
        monospace_fallbacks = ["DejaVu Sans Mono"]

    selected_font = monospace_fallbacks[0]

    font_rcparams = {
        "font.family": "monospace",
        "font.monospace": monospace_fallbacks,
        "mathtext.fontset": "custom",
        "mathtext.rm": selected_font,
        "mathtext.it": f"{selected_font}:italic",
        "mathtext.bf": f"{selected_font}:bold",
        "mathtext.tt": selected_font,
        "mathtext.fallback": "stix",
    }

    return font_rcparams





def _font_scaling_rcparams(
    figsize: tuple[float, float] = (9, 6),
    base_fontsize: float = 12.0,
    reference_figsize: tuple[float, float] = (4, 3),
) -> dict:
    width, height = figsize
    ref_width, ref_height = reference_figsize
    scale = ((width / ref_width) + (height / ref_height)) / 2.0

    label_size = base_fontsize * scale
    tick_size = label_size * 0.9
    title_size = label_size * 1.15
    legend_size = label_size * 0.7

    return {
        "font.size": label_size,
        "axes.titlesize": title_size,
        "axes.labelsize": label_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": legend_size,
    }


def dark_style(
    figsize: tuple[float, float] = (9, 6),
    base_fontsize: float = 12.0,
) -> None:
    rcparams = {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.transparent": False,
        "axes.facecolor": (1, 1, 1, 0.15),
        "axes.labelcolor": "white",
        "xtick.color": "white",
        "ytick.color": "white",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "text.color": "white",
        "grid.color": "white",
        "grid.alpha": 0.3,
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "axes.edgecolor": "white",
        "legend.framealpha": 0.05,
        "legend.edgecolor": "white",
    }
    rcparams.update(_build_cross_platform_font_rcparams())

    rcparams["figure.facecolor"] = "#0d1117"
    plt.rcParams.update(rcparams)  # type: ignore[arg-type]


def light_style(
    grid_color: str = "black",
    figsize: tuple[float, float] = (9, 6),
    base_fontsize: float = 12.0,
) -> None:
    rcparams = {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.transparent": False,
        "axes.facecolor": "none",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "text.color": "black",
        "grid.color": grid_color,
        "grid.alpha": 0.18,
        "axes.grid": True,
        "grid.linewidth": 0.5,
        "axes.edgecolor": "black",
        "legend.framealpha": 0.12,
        "legend.edgecolor": "black",
        "legend.facecolor": "none",
    }
    rcparams.update(_build_cross_platform_font_rcparams())
    rcparams.update(_font_scaling_rcparams(figsize=figsize, base_fontsize=base_fontsize))

    rcparams["figure.facecolor"] = "white"
    plt.rcParams.update(rcparams)  # type: ignore[arg-type]


# =============================================================================
# Embedded CustomTkinter viewer controls
# =============================================================================

VIEWER_BG = "#1a1a1a"
VIEWER_PANEL = "#121214"
VIEWER_CARD = "#1a1a1a"
VIEWER_BORDER = "#566173"
VIEWER_CONTROL = "#2A2F3A"
VIEWER_CONTROL_HOVER = "#353C49"
VIEWER_ACCENT = "#4F8CFF"
VIEWER_ACCENT_HOVER = "#6DA1FF"
VIEWER_SLIDER_PROGRESS = "#7652D6"
VIEWER_DROPDOWN_HOVER = "#8D73DF"
VIEWER_SUCCESS = "#45D483"
VIEWER_TEXT = "#F4F6FA"
VIEWER_MUTED_TEXT = "#9DA7B8"


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


VIEWER_FONT_DEFAULTS = {
    "font_viewer_title": 25,
    "font_viewer_subtitle": 14,
    "font_viewer_card_title": 19,
    "font_viewer_card_note": 14,
    "font_viewer_plot_axes_label": 16,
    "font_viewer_tick_label": 14,
    "font_viewer_subsection": 16,
    "font_viewer_control_label": 16,
    "font_viewer_value": 15,
    "font_viewer_button": 15,
    "font_viewer_toolbar": 14,
    "font_viewer_status": 15,
    "font_viewer_coordinates": 15,
    "font_viewer_step": 19,
}


@dataclass(frozen=True)
class ViewerFontSet:
    title: ctk.CTkFont
    subtitle: ctk.CTkFont
    card_title: ctk.CTkFont
    card_note: ctk.CTkFont
    subsection: ctk.CTkFont
    control_label: ctk.CTkFont
    value: ctk.CTkFont
    button: ctk.CTkFont
    toolbar: ctk.CTkFont
    status: ctk.CTkFont
    coordinates: ctk.CTkFont
    step: ctk.CTkFont


def build_viewer_fonts(
    sizes: dict[str, int] | None = None,
    factory: Callable[..., ctk.CTkFont] | None = None,
) -> ViewerFontSet:
    """Build shared fonts, optionally through the launcher's live registry."""

    resolved_sizes = dict(VIEWER_FONT_DEFAULTS)
    if sizes is not None:
        for key in VIEWER_FONT_DEFAULTS:
            if key in sizes:
                resolved_sizes[key] = max(1, int(sizes[key]))

    def create(key: str, *, bold: bool = False) -> ctk.CTkFont:
        size = resolved_sizes[key]
        if factory is not None:
            return factory(size, bold=bold)
        return ctk.CTkFont(
            size=size,
            weight="bold" if bold else "normal",
        )

    return ViewerFontSet(
        title=create("font_viewer_title", bold=True),
        subtitle=create("font_viewer_subtitle"),
        card_title=create("font_viewer_card_title", bold=True),
        card_note=create("font_viewer_card_note"),
        subsection=create("font_viewer_subsection", bold=True),
        control_label=create("font_viewer_control_label", bold=True),
        value=create("font_viewer_value"),
        button=create("font_viewer_button", bold=True),
        toolbar=create("font_viewer_toolbar", bold=True),
        status=create("font_viewer_status"),
        coordinates=create("font_viewer_coordinates"),
        step=create("font_viewer_step", bold=True),
    )


def resolve_viewer_fonts(
    master,
    fonts: ViewerFontSet | None = None,
) -> ViewerFontSet:
    if fonts is not None:
        return fonts

    current = master
    while current is not None:
        inherited = getattr(current, "_viewer_fonts", None)
        if isinstance(inherited, ViewerFontSet):
            return inherited
        current = getattr(current, "master", None)

    return build_viewer_fonts()


class _ControlTextProxy:
    """Small compatibility layer for former Matplotlib text artists."""

    def __init__(self, widget) -> None:
        self.widget = widget

    def set_text(self, text: object) -> None:
        if isinstance(self.widget, ctk.CTkEntry):
            self.widget.delete(0, "end")
            self.widget.insert(0, str(text))
        else:
            self.widget.configure(text=str(text))

    def get_text(self) -> str:
        if isinstance(self.widget, ctk.CTkEntry):
            return str(self.widget.get())
        return str(self.widget.cget("text"))

    def set_color(self, color: str) -> None:
        try:
            self.widget.configure(text_color=color)
        except (TypeError, ValueError, tk.TclError):
            pass

    def set_fontsize(self, _size: float) -> None:
        return

    def set_horizontalalignment(self, _alignment: str) -> None:
        return

    def set_position(self, _position) -> None:
        return


class _ControlAxesProxy:
    """Expose the tiny subset of Axes API used by the viewer callbacks."""

    def __init__(self, control) -> None:
        self.control = control

    def set_visible(self, visible: bool) -> None:
        self.control.set_visible(bool(visible))

    def get_visible(self) -> bool:
        return bool(self.control.visible)

    def set_xlim(self, lower: float, upper: float) -> None:
        if hasattr(self.control, "apply_range"):
            self.control.apply_range(float(lower), float(upper))

    def tick_params(self, **_kwargs) -> None:
        return


class ModernSliderControl(ctk.CTkFrame):
    """Compact slider with label, editable value, and step buttons."""

    def __init__(
        self,
        master,
        label: str,
        valmin: float,
        valmax: float,
        *,
        valinit: float,
        valstep: float | list[int] | list[float] | None,
        fonts: ViewerFontSet | None = None,
        slider_width: int = 158,
        value_formatter: Callable[[float], str] | None = None,
    ) -> None:
        super().__init__(master, fg_color="transparent", corner_radius=0)
        fonts = resolve_viewer_fonts(master, fonts)

        self.valmin = float(valmin)
        self.valmax = float(valmax)
        self.valstep = valstep
        self.val = float(valinit)
        self._value_formatter = value_formatter
        self.eventson = True
        self.visible = True
        self._active = True
        self._callbacks: dict[int, Callable[[float], None]] = {}
        self._next_callback_id = 1

        self.grid_columnconfigure(0, weight=1)

        self._label_widget = ctk.CTkLabel(
            self,
            text=label,
            anchor="w",
            justify="left",
            text_color=VIEWER_TEXT,
            font=fonts.control_label,
        )
        self._label_widget.grid(row=0, column=0, columnspan=3, sticky="ew")
        bind_responsive_label_wrap(
            self._label_widget,
            self,
            horizontal_padding=100,
        )

        self._value_var = tk.StringVar(value=self._format_value(self.val))
        self._entry = ctk.CTkEntry(
            self,
            width=76,
            height=29,
            textvariable=self._value_var,
            justify="center",
            fg_color=VIEWER_CONTROL,
            border_color=VIEWER_BORDER,
            border_width=1,
            corner_radius=7,
            text_color=VIEWER_TEXT,
            font=fonts.value,
        )
        self._entry.grid(row=0, column=3, columnspan=2, sticky="e", pady=(0, 4))
        self._entry.bind("<Return>", self._commit_entry)
        self._entry.bind("<KP_Enter>", self._commit_entry)
        self._entry.bind("<FocusOut>", self._commit_entry)

        self._slider = ctk.CTkSlider(
            self,
            width=slider_width,
            height=16,
            corner_radius=8,
            border_width=0,
            fg_color="#343A46",
            progress_color=VIEWER_SLIDER_PROGRESS,
            button_color="#E6ECF7",
            button_hover_color="white",
            command=self._slider_changed,
        )
        self._slider.grid(row=1, column=0, sticky="w", padx=(0, 7), pady=(1, 3))

        self.minus_button = ctk.CTkButton(
            self,
            text="−",
            width=30,
            height=30,
            corner_radius=8,
            fg_color=VIEWER_CONTROL,
            hover_color=VIEWER_CONTROL_HOVER,
            border_width=1,
            border_color=VIEWER_BORDER,
            font=fonts.step,
            command=lambda: self.step(-1),
        )
        self.minus_button.grid(row=1, column=3, padx=(0, 5), pady=(1, 3))

        self.plus_button = ctk.CTkButton(
            self,
            text="+",
            width=30,
            height=30,
            corner_radius=8,
            fg_color=VIEWER_CONTROL,
            hover_color=VIEWER_CONTROL_HOVER,
            border_width=1,
            border_color=VIEWER_BORDER,
            font=fonts.step,
            command=lambda: self.step(1),
        )
        self.plus_button.grid(row=1, column=4, pady=(1, 3))

        self.label = _ControlTextProxy(self._label_widget)
        self.valtext = _ControlTextProxy(self._entry)
        self.ax = _ControlAxesProxy(self)
        self.apply_range(self.valmin, self.valmax)
        self.set_val(self.val, emit=False)

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, active: bool) -> None:
        self._active = bool(active)
        state = "normal" if self._active else "disabled"
        for widget in (self._slider, self._entry, self.minus_button, self.plus_button):
            widget.configure(state=state)

    def _allowed_values(self) -> np.ndarray | None:
        if isinstance(self.valstep, (list, tuple, np.ndarray)):
            allowed = np.asarray(self.valstep, dtype=float)
            return allowed[(allowed >= self.valmin) & (allowed <= self.valmax)]
        return None

    def _scalar_step(self) -> float | None:
        if self.valstep is None or isinstance(
            self.valstep,
            (list, tuple, np.ndarray),
        ):
            return None
        return abs(float(self.valstep))

    def _format_value(self, value: float) -> str:
        if self._value_formatter is not None:
            return self._value_formatter(value)
        if isinstance(self.valstep, (list, tuple, np.ndarray)):
            return f"{value:g}"
        scalar_step = self._scalar_step()
        if scalar_step is None:
            return f"{value:g}"
        if scalar_step >= 1:
            return f"{value:.0f}"
        decimals = int(min(max(np.ceil(-np.log10(scalar_step)), 0), 8))
        return f"{value:.{decimals}f}".rstrip("0").rstrip(".")

    def _snap(self, value: float) -> float:
        value = float(np.clip(value, self.valmin, self.valmax))
        allowed = self._allowed_values()
        if allowed is not None and allowed.size:
            return float(allowed[int(np.argmin(np.abs(allowed - value)))])
        scalar_step = self._scalar_step()
        if scalar_step is not None and scalar_step > 0:
            value = (
                self.valmin
                + round((value - self.valmin) / scalar_step) * scalar_step
            )
        return float(np.clip(value, self.valmin, self.valmax))

    def _value_to_slider(self, value: float) -> float:
        allowed = self._allowed_values()
        if allowed is not None and allowed.size:
            return float(int(np.argmin(np.abs(allowed - value))))
        return float(value)

    def _slider_to_value(self, slider_value: float) -> float:
        allowed = self._allowed_values()
        if allowed is not None and allowed.size:
            index = int(np.clip(round(slider_value), 0, allowed.size - 1))
            return float(allowed[index])
        return float(slider_value)

    def apply_range(self, lower: float, upper: float) -> None:
        self.valmin = float(lower)
        self.valmax = max(float(upper), self.valmin)
        allowed = self._allowed_values()
        if allowed is not None and allowed.size:
            slider_min = 0.0
            slider_max = float(max(allowed.size - 1, 1))
            steps = max(allowed.size - 1, 1)
        else:
            slider_min = self.valmin
            slider_max = self.valmax if self.valmax > self.valmin else self.valmin + 1.0
            scalar_step = self._scalar_step()
            if scalar_step is None or scalar_step <= 0:
                steps = 500
            else:
                steps = int(
                    np.clip(
                        round((self.valmax - self.valmin) / scalar_step),
                        1,
                        2000,
                    )
                )
        self._slider.configure(
            from_=slider_min,
            to=slider_max,
            number_of_steps=steps,
        )
        self.set_val(self.val, emit=False)

    def _slider_changed(self, slider_value: float) -> None:
        if self._active:
            self.set_val(self._slider_to_value(float(slider_value)), emit=True)

    def _commit_entry(self, _event=None) -> None:
        try:
            value = float(self._value_var.get().strip().replace(",", "."))
        except ValueError:
            self._value_var.set(self._format_value(self.val))
            return
        self.set_val(value, emit=True)

    def step(self, direction: int) -> None:
        if not self._active:
            return
        allowed = self._allowed_values()
        if allowed is not None and allowed.size:
            index = int(np.argmin(np.abs(allowed - self.val)))
            new_value = float(allowed[int(np.clip(index + direction, 0, allowed.size - 1))])
        else:
            scalar_step = self._scalar_step()
            step = (
                scalar_step
                if scalar_step is not None
                else max((self.valmax - self.valmin) / 100.0, 1e-12)
            )
            new_value = self.val + direction * step
        self.set_val(new_value, emit=True)

    def set_val(self, value: float, *, emit: bool = True) -> None:
        new_value = self._snap(float(value))
        changed = not np.isclose(new_value, self.val, rtol=0.0, atol=1e-12)
        self.val = new_value
        self._slider.set(self._value_to_slider(self.val))
        self._value_var.set(self._format_value(self.val))
        if emit and changed and self.eventson:
            for callback in tuple(self._callbacks.values()):
                callback(self.val)

    def on_changed(self, callback: Callable[[float], None]) -> int:
        callback_id = self._next_callback_id
        self._next_callback_id += 1
        self._callbacks[callback_id] = callback
        return callback_id

    def set_visible(self, visible: bool) -> None:
        self.visible = bool(visible)
        if self.visible:
            self.grid()
        else:
            self.grid_remove()


class ModernButtonControl(ctk.CTkButton):
    """CustomTkinter action/toggle button with Matplotlib-like callbacks."""

    def __init__(
        self,
        master,
        text: str,
        *,
        font: ctk.CTkFont | None = None,
        width: int = 110,
    ) -> None:
        if font is None:
            font = resolve_viewer_fonts(master).button
        super().__init__(
            master,
            text=text,
            width=width,
            height=34,
            corner_radius=9,
            fg_color=VIEWER_CONTROL,
            hover_color=VIEWER_CONTROL_HOVER,
            border_width=1,
            border_color=VIEWER_BORDER,
            text_color=VIEWER_TEXT,
            font=font,
            command=self._clicked,
        )
        self.visible = True
        self._active = True
        self._callbacks: dict[int, Callable[[object], None]] = {}
        self._next_callback_id = 1
        self.label = _ControlTextProxy(self)
        self.ax = _ControlAxesProxy(self)

    @property
    def active(self) -> bool:
        return self._active

    @active.setter
    def active(self, active: bool) -> None:
        self._active = bool(active)
        self.configure(state="normal" if self._active else "disabled")

    def _clicked(self) -> None:
        if self._active:
            for callback in tuple(self._callbacks.values()):
                callback(None)

    def on_clicked(self, callback: Callable[[object], None]) -> int:
        callback_id = self._next_callback_id
        self._next_callback_id += 1
        self._callbacks[callback_id] = callback
        return callback_id

    def set_enabled_style(self, enabled: bool) -> None:
        if enabled:
            self.configure(
                fg_color="#174C36",
                hover_color="#1F6B49",
                border_color=VIEWER_SUCCESS,
            )
        else:
            self.configure(
                fg_color=VIEWER_CONTROL,
                hover_color=VIEWER_CONTROL_HOVER,
                border_color=VIEWER_BORDER,
            )

    def set_visible(self, visible: bool) -> None:
        self.visible = bool(visible)
        if self.visible:
            self.grid()
        else:
            self.grid_remove()


class ModernChoiceControl(ctk.CTkOptionMenu):
    """Compact selector exposing the former RadioButtons callback name."""

    def __init__(
        self,
        master,
        values: list[str],
        value: str,
        *,
        font: ctk.CTkFont | None = None,
        dropdown_font: ctk.CTkFont | None = None,
        width: int = 210,
    ) -> None:
        self._callbacks: dict[int, Callable[[str], None]] = {}
        self._next_callback_id = 1
        inherited_fonts = resolve_viewer_fonts(master)
        if font is None:
            font = inherited_fonts.button
        if dropdown_font is None:
            dropdown_font = inherited_fonts.value
        super().__init__(
            master,
            values=values,
            width=width,
            height=34,
            corner_radius=9,
            fg_color=VIEWER_CONTROL,
            button_color=VIEWER_SLIDER_PROGRESS,
            button_hover_color=VIEWER_DROPDOWN_HOVER,
            dropdown_fg_color=VIEWER_CARD,
            dropdown_hover_color=VIEWER_CONTROL_HOVER,
            text_color=VIEWER_TEXT,
            font=font,
            dropdown_font=dropdown_font,
            command=self._selected,
        )
        self.set(value)
        self.labels: list[object] = []

    def _selected(self, value: str) -> None:
        for callback in tuple(self._callbacks.values()):
            callback(value)

    def on_clicked(self, callback: Callable[[str], None]) -> int:
        callback_id = self._next_callback_id
        self._next_callback_id += 1
        self._callbacks[callback_id] = callback
        return callback_id


class ModernStatusLabel(ctk.CTkLabel):
    def set_text(self, text: object) -> None:
        self.configure(text=str(text))
        refresh_wrap = getattr(self, "_responsive_wrap_update", None)
        if callable(refresh_wrap):
            refresh_wrap()

    def get_text(self) -> str:
        return str(self.cget("text"))


class ModernStepControl(TypedDict):
    widget: ModernSliderControl
    values: list[int]
    get_value: Callable[[], float]
    set_index: Callable[[int], None]
    callback_ids: tuple[int]


@dataclass(frozen=True)
class ViewerSection:
    """Typed access to a sidebar section and its content area."""

    container: ctk.CTkFrame
    body: ctk.CTkFrame
    set_expanded: Callable[[bool], None]
    is_expanded: Callable[[], bool]


def make_viewer_card(
    master,
    title: str,
    subtitle: str = "",
    fonts: ViewerFontSet | None = None,
    initially_expanded: bool = True,
) -> ViewerSection:
    """Create one consistently styled section in the viewer sidebar."""

    fonts = resolve_viewer_fonts(master, fonts)

    card = ctk.CTkFrame(
        master,
        fg_color=VIEWER_CARD,
        border_color=VIEWER_BORDER,
        border_width=2,
        corner_radius=13,
    )
    card.grid_columnconfigure(0, weight=1)

    header_button = ctk.CTkButton(
        card,
        text=f"▾  {title.upper()}",
        anchor="w",
        height=30,
        corner_radius=8,
        fg_color="transparent",
        hover_color=VIEWER_CONTROL_HOVER,
        text_color=VIEWER_TEXT,
        font=fonts.card_title,
    )
    header_button.grid(row=0, column=0, sticky="ew", padx=7, pady=(6, 0))

    next_row = 1
    subtitle_label: ctk.CTkLabel | None = None
    if subtitle:
        subtitle_label = ctk.CTkLabel(
            card,
            text=subtitle,
            anchor="w",
            justify="left",
            text_color=VIEWER_MUTED_TEXT,
            font=fonts.card_note,
        )
        subtitle_label.grid(
            row=1,
            column=0,
            sticky="ew",
            padx=13,
            pady=(1, 5),
        )
        bind_responsive_label_wrap(
            subtitle_label,
            card,
            horizontal_padding=26,
        )
        next_row = 2

    body = ctk.CTkFrame(card, fg_color="transparent", corner_radius=0)
    body.grid(row=next_row, column=0, sticky="ew", padx=13, pady=(7, 11))
    body.grid_columnconfigure(0, weight=1)

    expanded = True

    def set_expanded(value: bool) -> None:
        nonlocal expanded
        expanded = bool(value)
        header_button.configure(
            text=f"{'▾' if expanded else '▸'}  {title.upper()}"
        )
        header_button.grid_configure(pady=(6, 0) if expanded else (6, 6))

        if expanded:
            if subtitle_label is not None:
                subtitle_label.grid()
            body.grid()
        else:
            if subtitle_label is not None:
                subtitle_label.grid_remove()
            body.grid_remove()

    def toggle_expanded() -> None:
        set_expanded(not expanded)

    def is_expanded() -> bool:
        return expanded

    header_button.configure(command=toggle_expanded)
    set_expanded(initially_expanded)
    return ViewerSection(
        container=card,
        body=body,
        set_expanded=set_expanded,
        is_expanded=is_expanded,
    )


def make_viewer_subsection(
    master,
    title: str,
    fonts: ViewerFontSet | None = None,
    initially_expanded: bool = True,
) -> ViewerSection:
    """Create a compact nested group inside a viewer card."""

    fonts = resolve_viewer_fonts(master, fonts)

    group = ctk.CTkFrame(
        master,
        fg_color="#191D24",
        border_color="#2A303B",
        border_width=1,
        corner_radius=10,
    )
    group.grid_columnconfigure(0, weight=1)

    header_button = ctk.CTkButton(
        group,
        text=f"▾  {title.upper()}",
        anchor="w",
        height=27,
        corner_radius=7,
        fg_color="transparent",
        hover_color=VIEWER_CONTROL_HOVER,
        text_color="#C7D0DE",
        font=fonts.subsection,
    )
    header_button.grid(row=0, column=0, sticky="ew", padx=5, pady=(4, 0))

    body = ctk.CTkFrame(group, fg_color="transparent", corner_radius=0)
    body.grid(row=1, column=0, sticky="ew", padx=10, pady=(2, 9))
    body.grid_columnconfigure(0, weight=1)

    expanded = True

    def set_expanded(value: bool) -> None:
        nonlocal expanded
        expanded = bool(value)
        header_button.configure(
            text=f"{'▾' if expanded else '▸'}  {title.upper()}"
        )
        header_button.grid_configure(pady=(4, 0) if expanded else (4, 4))
        if expanded:
            body.grid()
        else:
            body.grid_remove()

    def toggle_expanded() -> None:
        set_expanded(not expanded)

    def is_expanded() -> bool:
        return expanded

    header_button.configure(command=toggle_expanded)
    set_expanded(initially_expanded)
    return ViewerSection(
        container=group,
        body=body,
        set_expanded=set_expanded,
        is_expanded=is_expanded,
    )


# =============================================================================
# Small fitting models
# =============================================================================


def gauss(x, amp, mu, sigma, offset):
    return amp * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + offset


def poly_model(x, a0, a1, a2, a4):
    return a0 + a1 * x + a2 * x**2 + a4 * x**4


# =============================================================================
# Data helpers
# =============================================================================


def get_scan_file_dtype(dtype_ending: str):
    if dtype_ending == ".u16":
        # Kept as in your original script. Change to np.uint16 if your data really is unsigned.
        return np.int16
    if dtype_ending == ".bin":
        return np.float32
    raise ValueError(
        f"Unsupported scan extension '{dtype_ending}'. Supported: .bin, .u16"
    )


# =============================================================================
# SPEC metadata helpers
# =============================================================================

def _resolve_spec_dir(
    spec_dir: str | Path | None = None,
) -> Path:
    """Resolve an explicitly provided SPEC search path."""

    if spec_dir is None:
        raise ValueError(
            "No SPEC search root was provided."
        )

    return Path(spec_dir).expanduser().resolve()

def _find_spec_files(spec_dir: str | Path | None = None) -> list[Path]:
    """
    Find SPEC files below a search root.

    If spec_dir is a folder, search recursively below that folder.
    If spec_dir is a .spec file, use that file directly.
    A SPEC search path must be provided explicitly.
    """

    spec_dir = _resolve_spec_dir(spec_dir)

    if spec_dir.is_file():
        if spec_dir.suffix.lower() == ".spec":
            return [spec_dir]

        raise FileNotFoundError(
            f"SPEC path is a file but not a .spec file:\n{spec_dir}"
        )

    if not spec_dir.exists():
        raise FileNotFoundError(
            f"SPEC search root does not exist:\n{spec_dir}"
        )

    spec_files = sorted(
        path
        for path in spec_dir.rglob("*")
        if path.is_file() and path.suffix.lower() == ".spec"
    )

    if not spec_files:
        raise FileNotFoundError(
            "No .spec files found below SPEC search root:\n"
            f"{spec_dir}"
        )

    return spec_files

def _read_spec_text(file_path: Path) -> str:
    """
    Read a SPEC file robustly.

    SPEC files are usually plain text, but the encoding may vary between systems.
    """

    try:
        return file_path.read_text(encoding="utf-8", errors="replace")
    except UnicodeDecodeError:
        return file_path.read_text(encoding="latin-1", errors="replace")


def _normalize_scan_numbers(*args) -> set[int]:
    """
    Normalize scan input into a set of integer scan numbers.

    Accepted examples:
        get_motor_value(1001)
        get_motor_value([1001, 1002])
        get_motor_value(np.array([1001, 1002]))
        get_motor_value((1001, 1005))             # inclusive range
        get_motor_value([(1001, 1005), (1010, 1012)])
    """

    scan_nums: set[int] = set()

    for arg in args:
        if arg is None:
            continue

        if isinstance(arg, np.ndarray):
            scan_nums.update(int(x) for x in arg.ravel())
            continue

        if isinstance(arg, range):
            scan_nums.update(int(x) for x in arg)
            continue

        if (
            isinstance(arg, tuple)
            and len(arg) == 2
            and all(isinstance(x, (int, np.integer)) for x in arg)
        ):
            start, end = int(arg[0]), int(arg[1])
            step = 1 if end >= start else -1
            scan_nums.update(range(start, end + step, step))
            continue

        if isinstance(arg, (list, tuple)):
            if len(arg) == 0:
                continue

            if all(
                isinstance(item, (list, tuple))
                and len(item) == 2
                for item in arg
            ):
                for start, end in arg:
                    start = int(start)
                    end = int(end)
                    step = 1 if end >= start else -1
                    scan_nums.update(range(start, end + step, step))
            else:
                scan_nums.update(int(x) for x in arg)

            continue

        scan_nums.add(int(arg))

    return scan_nums


def _extract_motor_names_from_spec_header(header_text: str) -> list[str]:
    """
    Extract motor names from SPEC #O lines.

    SPEC stores motor names in header lines like:
        #O0 motor1 motor2 motor3
        #O1 motor4 motor5 ...
    """

    motor_names: list[str] = []

    for line in header_text.splitlines():
        if re.match(r"#O\d+\b", line):
            motor_names.extend(line.split()[1:])

    return motor_names


def _get_scan_number_from_spec_block(scan_block: str) -> int | None:
    """
    Extract the scan number from one SPEC scan block.

    A block starts after '#S ', so the first token is the scan number.
    """

    lines = scan_block.splitlines()

    if not lines:
        return None

    first_line_parts = lines[0].split()

    if not first_line_parts:
        return None

    try:
        return int(first_line_parts[0])
    except ValueError:
        return None


def _extract_p_values_from_scan_block(scan_block: str) -> list[str]:
    """
    Extract all motor position values from #P lines inside one scan block.
    """

    p_values: list[str] = []

    for line in scan_block.splitlines():
        if re.match(r"#P\d+\b", line):
            p_values.extend(line.split()[1:])

    return p_values


def _coerce_spec_value(raw_value: str):
    """
    Convert a SPEC value to float when possible.

    Some files may use D notation instead of E notation.
    """

    try:
        return float(raw_value.replace("D", "E").replace("d", "e"))
    except ValueError:
        return raw_value


def get_available_motor_names(spec_dir: str | Path | None = None) -> list[str]:
    """
    Return all motor names found in all .spec file headers inside spec_dir.
    """

    spec_dir = _resolve_spec_dir(spec_dir)
    spec_files = _find_spec_files(spec_dir)

    motor_names: set[str] = set()

    for file_path in spec_files:
        content = _read_spec_text(file_path)
        header = content.split("#S ", 1)[0]
        motor_names.update(_extract_motor_names_from_spec_header(header))

    return sorted(motor_names)


def get_available_motor_names_for_scan(
    scan_num: int,
    spec_dir: str | Path | None = None,
) -> list[str]:
    """
    Return all available motor names for the SPEC file that contains scan_num.

    The motor names are taken from the header of the SPEC file containing
    the selected scan.
    """

    scan_num = int(scan_num)
    spec_dir = _resolve_spec_dir(spec_dir)
    spec_files = _find_spec_files(spec_dir)

    for file_path in spec_files:
        content = _read_spec_text(file_path)
        parts = content.split("#S ")

        header = parts[0]
        motor_names = _extract_motor_names_from_spec_header(header)

        if not motor_names:
            continue

        for scan_block in parts[1:]:
            current_scan = _get_scan_number_from_spec_block(scan_block)

            if current_scan == scan_num:
                return sorted(motor_names)

    return []


def get_motor_values_for_scans(
    *args,
    spec_dir: str | Path | None = None,
    source_files: dict[int, Path] | None = None,
) -> dict[int, dict[str, float | str]]:
    """
    Load all available motor names and values for multiple scans.

    Every relevant SPEC file is read only once.
    """

    scan_nums = _normalize_scan_numbers(*args)

    if not scan_nums:
        raise ValueError("No scan numbers given.")

    spec_files = _find_spec_files(spec_dir)
    result: dict[int, dict[str, float | str]] = {}

    for file_path in spec_files:
        remaining_scans = scan_nums - set(result)

        if not remaining_scans:
            break

        content = _read_spec_text(file_path)
        parts = content.split("#S ")

        header = parts[0]
        motor_names = _extract_motor_names_from_spec_header(
            header
        )

        if not motor_names:
            continue

        for scan_block in parts[1:]:
            scan_num = _get_scan_number_from_spec_block(
                scan_block
            )

            if (
                scan_num is None
                or scan_num not in remaining_scans
            ):
                continue

            raw_values = _extract_p_values_from_scan_block(
                scan_block
            )

            result[scan_num] = {
                motor_name: _coerce_spec_value(raw_value)
                for motor_name, raw_value in zip(
                    motor_names,
                    raw_values,
                )
            }
            if source_files is not None:
                source_files[scan_num] = file_path

    return result

def get_motor_value(
    *args,
    motor_name: str = "pgm_en",
    spec_dir: str | Path | None = None,
    sort: bool = False,
) -> dict[int, float | str]:
    """
    Extract one motor value for one or multiple scans.

    Returns:
        {
            scan_number: motor_value,
            ...
        }

    Example:
        get_motor_value(1001, motor_name="pgm_en", spec_dir="/path/to/SpecData")
        get_motor_value([1001, 1002, 1003], motor_name="pgm_en")
    """

    spec_dir = _resolve_spec_dir(spec_dir)
    scan_nums = _normalize_scan_numbers(*args)

    if not scan_nums:
        raise ValueError("No scan numbers given.")

    if not isinstance(motor_name, str) or not motor_name.strip():
        raise ValueError("motor_name must be a non-empty string.")

    motor_name = motor_name.strip()

    spec_files = _find_spec_files(spec_dir)

    if not spec_files:
        raise FileNotFoundError(f"No .spec files found under: {spec_dir}")

    valid_motor_names = get_available_motor_names(spec_dir)

    if motor_name not in valid_motor_names:
        if valid_motor_names:
            raise ValueError(
                f"Unknown motor_name '{motor_name}'. "
                f"Valid motor names are: {valid_motor_names}"
            )

        raise ValueError(
            f"Unknown motor_name '{motor_name}'. "
            f"No motor names found in .spec headers under '{spec_dir}'."
        )

    motor_dict: dict[int, float | str] = {}

    for file_path in spec_files:
        remaining = scan_nums - set(motor_dict)

        if not remaining:
            break

        content = _read_spec_text(file_path)
        parts = content.split("#S ")

        header = parts[0]
        motor_names = _extract_motor_names_from_spec_header(header)

        if motor_name not in motor_names:
            continue

        idx_motor = motor_names.index(motor_name)

        for scan_block in parts[1:]:
            scan_num = _get_scan_number_from_spec_block(scan_block)

            if scan_num is None or scan_num not in remaining:
                continue

            p_values = _extract_p_values_from_scan_block(scan_block)

            if idx_motor >= len(p_values):
                continue

            motor_dict[scan_num] = _coerce_spec_value(p_values[idx_motor])

    if sort:
        motor_dict = dict(sorted(motor_dict.items(), key=lambda item: item[1]))

    return motor_dict


def get_motor_value_for_scan(
    scan_num: int,
    motor_name: str,
    spec_dir: str | Path | None = None,
) -> float | str | None:
    """
    Convenience helper for the GUI.

    Returns only one value instead of a dictionary.
    """

    values = get_motor_value(
        int(scan_num),
        motor_name=motor_name,
        spec_dir=spec_dir,
    )

    return values.get(int(scan_num))

def _matches_scan_axis(path: Path, scan_num: int, axis: str) -> bool:
    name = path.stem.lower()
    return f"_{int(scan_num)}{axis.lower()}" in name


def _iter_files_recursive(root: Path):
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(
        root,
        topdown=True,
        onerror=lambda error: None,
        followlinks=False,
    ):
        dirnames[:] = [
            d
            for d in dirnames
            if d
            not in {
                ".git",
                "__pycache__",
                ".cache",
                ".local/share/Trash",
                "node_modules",
            }
        ]
        for filename in filenames:
            yield Path(dirpath) / filename


def _find_scan_pair_in_directory(scan_num: int, directory: Path) -> tuple[Path | None, Path | None]:
    directory = Path(directory)
    x_file = None
    y_file = None

    for path in directory.iterdir():
        if not path.is_file():
            continue
        if x_file is None and _matches_scan_axis(path, scan_num, "x"):
            x_file = path
        if y_file is None and _matches_scan_axis(path, scan_num, "y"):
            y_file = path
        if x_file is not None and y_file is not None:
            break

    return x_file, y_file


def _find_scan_pair_recursive(
    scan_num: int,
    search_roots: Iterable[Path],
) -> tuple[Path | None, Path | None]:
    x_file = None
    y_file = None

    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue

        for path in _iter_files_recursive(root):
            if x_file is None and _matches_scan_axis(path, scan_num, "x"):
                x_file = path
            if y_file is None and _matches_scan_axis(path, scan_num, "y"):
                y_file = path
            if x_file is not None and y_file is not None:
                return x_file, y_file

    return x_file, y_file


def _find_scan_files(
    scans: np.ndarray,
    scans_dir: Path | None = None,
    *,
    search_roots: Iterable[Path] | None = None,
    fallback_recursive_search: bool = True, print_log: bool = True,
) -> list[tuple[Path, Path]]:
    scans = np.asarray(scans, dtype=int)
    if scans.size == 0:
        raise ValueError("No scans given.")

    if scans_dir is not None:
        primary_dir = Path(scans_dir)
        search_roots_for_fallback = [primary_dir]
    else:
        if search_roots is None:
            search_roots = [Path("/")]

        first_scan = int(scans[0])
        # print(f"No scans_dir given. Searching recursively for scan files {first_scan}...")
        first_x, first_y = _find_scan_pair_recursive(first_scan, search_roots)
        if first_x is None or first_y is None:
            raise FileNotFoundError(f"Could not find x/y file pair for first scan {first_scan}.")

        primary_dir = first_x.parent
        if print_log:
            print(f"Found first scan {first_scan} in: {primary_dir}. Using this as primary directory.")
        search_roots_for_fallback = list(search_roots)

    scan_files: list[tuple[Path, Path]] = []
    missing_scans: list[int] = []

    for scan_num in scans:
        scan_num = int(scan_num)
        x_file, y_file = _find_scan_pair_in_directory(scan_num, primary_dir)

        if (x_file is None or y_file is None) and fallback_recursive_search:
            if print_log:
                print(f"Scan {scan_num} not complete in {primary_dir}. Searching recursively...")
            x_file, y_file = _find_scan_pair_recursive(scan_num, search_roots_for_fallback)

        if x_file is None or y_file is None:
            missing_scans.append(scan_num)
            continue

        scan_files.append((x_file, y_file))

    if missing_scans:
        print(f"Warning: missing x/y file pair for scans: {missing_scans}")

    if not scan_files:
        raise FileNotFoundError("No matching x/y scan-file pairs found.")

    return scan_files

def resolve_scan_file_map(
    scans: np.ndarray,
    *,
    search_roots: Iterable[Path],
    fallback_recursive_search: bool = True,
    print_log: bool = True,
) -> dict[int, tuple[Path, Path]]:
    scans = np.asarray(scans, dtype=int)

    scan_files = _find_scan_files(
        scans,
        scans_dir=None,
        search_roots=search_roots,
        fallback_recursive_search=fallback_recursive_search,
        print_log=print_log,
    )

    if len(scan_files) != scans.size:
        raise FileNotFoundError(
            "Not all requested scan files could be resolved."
        )

    return {
        int(scan): file_pair
        for scan, file_pair in zip(
            scans,
            scan_files,
            strict=True,
        )
    }


def _load_scan_arrays(scan_files: list[tuple[Path, Path]]) -> tuple[np.ndarray, np.ndarray]:
    x_arrays: list[np.ndarray] = []
    y_arrays: list[np.ndarray] = []

    for x_file, y_file in scan_files:
        x = np.fromfile(x_file, dtype=get_scan_file_dtype(x_file.suffix))
        y = np.fromfile(y_file, dtype=get_scan_file_dtype(y_file.suffix))

        if x.size != y.size:
            raise ValueError(
                f"Length mismatch for {x_file.name} and {y_file.name}: {x.size} != {y.size}"
            )

        mask = (x <= 4095) & (y <= 4095)
        x_arrays.append(x[mask])
        y_arrays.append(y[mask])

    return np.concatenate(x_arrays), np.concatenate(y_arrays)


def _build_calibration_spectrum(
    scan_num: int,
    scan_files: list[tuple[Path, Path]],
    percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
    # Calibration routines share the same histogram preparation. Keeping it in
    # one place prevents the two fitting APIs from drifting apart.
    x_array, y_array = _load_scan_arrays(scan_files)
    histogram, _x_edges, _y_edges = build_histogram_fast(
        x_array,
        y_array,
        tilt=0.0,
        x_min=0,
        x_max=4095,
    )

    histogram = histogram.astype(float, copy=True)
    nonzero = histogram[histogram > 0]
    if nonzero.size == 0:
        raise ValueError(
            f"Calibration scan {scan_num} produced an empty histogram."
        )

    upper_limit = float(np.percentile(nonzero, percentile))
    histogram[histogram > upper_limit] = 0.0
    spectrum = histogram.sum(axis=1)
    if float(np.sum(spectrum)) <= 0:
        raise ValueError(
            f"Calibration scan {scan_num} produced an empty 1D spectrum "
            "after percentile filtering."
        )

    pixel_axis = np.arange(spectrum.size, dtype=float)
    return pixel_axis, spectrum


# =============================================================================
# Energy calibration
# =============================================================================

def fit_gauss_offset_peak(
    pixel_axis: np.ndarray,
    spectrum: np.ndarray,
    *,
    scan_num: int | None = None,
    fit_half_width: int | None = None,
    min_sigma: float = 1.0,
    max_sigma: float = 300.0,
) -> dict:
    """
    Fit a Gaussian + constant offset to the strongest peak in a 1D spectrum.

    The fit is performed only in a range around the strongest peak.

    Returns:
        {
            "mu": fitted peak center,
            "amp": fitted amplitude,
            "sigma": fitted sigma,
            "offset": fitted constant offset,
            "fit_min": lower pixel of fit range,
            "fit_max": upper pixel of fit range,
            "success": lmfit success flag,
        }
    """

    if Model is None:
        raise ImportError(
            "lmfit is required for Gaussian calibration fitting. "
            "Install it with: pip install lmfit"
        )

    pixel_axis = np.asarray(pixel_axis, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)

    finite = np.isfinite(pixel_axis) & np.isfinite(spectrum)

    if not np.any(finite):
        raise ValueError("No finite values available for Gaussian peak fit.")

    x = pixel_axis[finite]
    y = spectrum[finite]

    if x.size < 5:
        raise ValueError("Not enough points for Gaussian peak fit.")

    if float(np.max(y)) <= 0:
        raise ValueError("Spectrum is empty or non-positive.")


    offset_guess = float(np.percentile(y, 10))

    signal = y - offset_guess
    signal[signal < 0] = 0.0

    top10 = sorted(signal, reverse=True)[:10]
    top10x = x[np.argsort(signal)[-10:]]

    mu_guess = int(np.mean(top10x))

    amp_guess = float(max(y[mu_guess] - offset_guess, np.max(y) - offset_guess, 1.0))

    # Estimate sigma from approximate FWHM.
    half_level = offset_guess + 0.5 * amp_guess

    left_index = mu_guess
    while left_index > 0 and y[left_index] > half_level:
        left_index -= 1

    right_index = mu_guess
    while right_index < y.size - 1 and y[right_index] > half_level:
        right_index += 1

    fwhm_guess = float(max(x[right_index] - x[left_index], 2.0))
    sigma_guess = fwhm_guess / 2.354820045

    sigma_guess = float(np.clip(sigma_guess, min_sigma, max_sigma))

    # Fit only a local range around the peak.
    if fit_half_width is None:
        fit_half_width = int(np.clip(6.0 * sigma_guess, 30, 400))

    fit_half_width = int(max(fit_half_width, 5))

    fit_min = mu_guess - fit_half_width
    fit_max = mu_guess + fit_half_width

    fit_mask = (x >= fit_min) & (x <= fit_max)

    x_fit = x[fit_mask]
    y_fit = y[fit_mask]

    if x_fit.size < 5:
        raise ValueError("Gaussian fit range contains too few points.")

    # Recompute offset/amplitude in the selected fit range.
    offset_guess = float(np.percentile(y_fit, 10))
    amp_guess = float(max(np.max(y_fit) - offset_guess, 1.0))

    model = Model(gauss)

    params = model.make_params(
        amp=amp_guess,
        mu=mu_guess,
        sigma=sigma_guess,
        offset=offset_guess,
    )

    params["amp"].min = 0.0
    params["mu"].min = float(x_fit[0])
    params["mu"].max = float(x_fit[-1])
    params["sigma"].min = float(min_sigma)
    params["sigma"].max = float(max(max_sigma, fit_half_width))
    params["offset"].min = 0.0

    result = model.fit(y_fit, params, x=x_fit)

    mu = float(result.params["mu"].value)
    amp = float(result.params["amp"].value)
    sigma = float(result.params["sigma"].value)
    offset = float(result.params["offset"].value)

    if not np.isfinite(mu):
        label = f" for scan {scan_num}" if scan_num is not None else ""
        raise RuntimeError(f"Gaussian peak fit failed{label}: non-finite mu.")

    return {
        "mu": mu,
        "amp": amp,
        "sigma": sigma,
        "offset": offset,
        "fit_min": float(x_fit[0]),
        "fit_max": float(x_fit[-1]),
        "success": bool(result.success),
    }

def bin_data(x, y, bin_size):
    num_bins = len(x) // bin_size
    x = x[: num_bins * bin_size]
    y = y[: num_bins * bin_size]

    edges = np.arange(0, num_bins * bin_size, bin_size)
    binned_counts = np.add.reduceat(y, edges)
    binned_x = x.reshape(num_bins, bin_size).mean(axis=1)
    binned_counts/=bin_size
    return binned_x, binned_counts

def compute_energy_calibration(
    calibration_scans,
    *,
    search_roots: Iterable[Path],
    spec_dir: str | Path | None = None,
    motor_name: str = "pgm_en",
    percentile: float = 99.985,
    fallback_recursive_search: bool = True,
) -> dict:


    scans = np.asarray(calibration_scans, dtype=int)

    if scans.size < 2:
        raise ValueError("At least two calibration scans are required.")

    if search_roots is None:
        raise ValueError("search_roots is required for calibration scans.")

    energy_by_scan = get_motor_value(
        scans,
        motor_name=motor_name,
        spec_dir=spec_dir,
    )

    points: list[dict] = []

    for scan_num in scans:
        scan_num = int(scan_num)

        if scan_num not in energy_by_scan:
            raise ValueError(
                f"No SPEC energy value found for calibration scan {scan_num} "
                f"using motor '{motor_name}'."
            )

        try:
            energy = float(energy_by_scan[scan_num])
        except Exception as exc:
            raise ValueError(
                f"Energy value for scan {scan_num} is not numeric: "
                f"{energy_by_scan[scan_num]!r}"
            ) from exc

        scan_files = _find_scan_files(
            np.asarray([scan_num], dtype=int),
            scans_dir=None,
            search_roots=search_roots,
            fallback_recursive_search=fallback_recursive_search,
        )

        pixel_axis, spectrum = _build_calibration_spectrum(
            scan_num,
            scan_files,
            percentile,
        )

        peak_fit = fit_gauss_offset_peak(
            pixel_axis,
            spectrum,
            scan_num=scan_num,
        )

        fitted_pixel = float(peak_fit["mu"])

        points.append(
            {
                "scan": scan_num,
                "pixel": fitted_pixel,
                "energy": energy,
                "amp": float(peak_fit["amp"]),
                "sigma": float(peak_fit["sigma"]),
                "offset": float(peak_fit["offset"]),
                "fit_min": float(peak_fit["fit_min"]),
                "fit_max": float(peak_fit["fit_max"]),
                "fit_success": bool(peak_fit["success"]),
            }
        )

    pixel_values = np.asarray([point["pixel"] for point in points], dtype=float)
    energy_values = np.asarray([point["energy"] for point in points], dtype=float)

    slope, intercept = np.polyfit(pixel_values, energy_values, 1)

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "points": points,
    }



com_scale_factor = 2048
def scaled_coordinate(com):
    return (com - com_scale_factor) / com_scale_factor


def poly_eval(z, coeffs):
    y = 0.0
    for i, c in enumerate(coeffs):
        y += c * z**i
    return y

def get_poly_param(params, name, z, poly_orders):
    order = poly_orders[name]
    coeffs = [params[f"{name}_c{i}"].value for i in range(order + 1)]
    return poly_eval(z, coeffs)

def double_gaussian(x, mu_bar, sigma1, sigma2, A, R, delta, offset=0):
    mu1 = mu_bar + delta
    mu2 = mu_bar - R * delta

    w1 = R / (1.0 + R)
    w2 = 1.0 / (1.0 + R)

    g1 = np.exp(-0.5 * ((x - mu1) / sigma1)**2) / (sigma1 * np.sqrt(2*np.pi))
    g2 = np.exp(-0.5 * ((x - mu2) / sigma2)**2) / (sigma2 * np.sqrt(2*np.pi))

    return A * (w1 * g1 + w2 * g2) + offset

def double_gaussian_components(x, mu_bar, sigma1, sigma2, A, R, delta, offset=0):
    mu1 = mu_bar + delta
    mu2 = mu_bar - R * delta

    w1 = R / (1.0 + R)
    w2 = 1.0 / (1.0 + R)

    g1 = np.exp(-0.5 * ((x - mu1) / sigma1)**2) / (sigma1 * np.sqrt(2*np.pi))
    g2 = np.exp(-0.5 * ((x - mu2) / sigma2)**2) / (sigma2 * np.sqrt(2*np.pi))

    return A * (w1 * g1), A*(w2 * g2)

def single_gaussian(x, mu, sigma, A, offset=0):
        profile = (
            np.exp(-0.5 * ((x - mu) / sigma) ** 2)
            / (sigma * np.sqrt(2 * np.pi))
        )
        return A * profile + offset

def get_coefficients(coeffs):
    return [0] * (4 - len(coeffs)) + list(coeffs)

EnergyCalibrationCoefficients = tuple[float, float, float, float]
EnergyCalibrationPayload = EnergyCalibrationCoefficients | dict[str, object]
EnergyCalibrationReturn = (
    EnergyCalibrationPayload
    | tuple[EnergyCalibrationPayload, dict[str, Figure]]
)

def compute_energy_calibration_2(
    calibration_scans,
    *,
    search_roots: Iterable[Path],
    spec_dir: str | Path | None = None,
    motor_name: str = "pgm_en",
    scan_files_by_scan: dict[int,tuple[Path, Path],] | None = None,
    motor_values_by_scan: dict[int,dict[str, float | str],] | None = None,
    percentile: float = 99.985,
    buffer: int = 0,
    fit_poly_order: int = 3,
    double_gaussian_model: bool = True,
    params_poly_orders=(2, 2, 2, 0),
    show_calibration_plots: bool = False,
    show_test_plot: bool = False,
    fallback_recursive_search: bool = True,
    return_details: bool = False,
    return_figures: bool = False,
) -> EnergyCalibrationReturn:
    """
    Determine the pixel-to-energy calibration.

    Parameters
    ----------
    fit_poly_order:
        Polynomial order of the final pixel-to-energy calibration (1, 2, or 3).
    double_gaussian_model:
        Use the global double-Gaussian model when True, otherwise use a single
        Gaussian with a detector-position-dependent width.
    buffer:
        Number of additional detector pixels included on both sides of the
        automatically detected calibration peak region.
    show_calibration_plots:
        Show the parameter, individual-fit, calibration-curve, and test plots.
    show_test_plot:
        Show only the final calibration test plot when
        show_calibration_plots is False.
    return_details:
        Return a JSON-serializable calibration package instead of only the four
        pixel-to-energy coefficients. The default preserves the legacy API.
    return_figures:
        Return the four calibration figures without opening Matplotlib windows.
    """

    if Model is None:
        raise ImportError(
            "lmfit is required for energy calibration. "
            "Install it with: pip install lmfit"
        )

    if fit_poly_order not in (1, 2, 3):
        raise ValueError("fit_poly_order must be 1, 2, or 3.")

    try:
        buffer = int(buffer)
    except (TypeError, ValueError) as exc:
        raise ValueError("buffer must be a non-negative integer.") from exc
    if buffer < 0:
        raise ValueError("buffer must be a non-negative integer.")

    scans = np.asarray(calibration_scans, dtype=int)

    if scans.size < 2:
        raise ValueError("At least two calibration scans are required.")

    if scans.size <= fit_poly_order:
        raise ValueError(
            f"At least {fit_poly_order + 1} calibration scans are required "
            f"for a polynomial fit of order {fit_poly_order}."
        )

    if search_roots is None:
        raise ValueError("search_roots is required for calibration scans.")

    if motor_values_by_scan is not None:
        energies = {
            int(scan): motor_values_by_scan[int(scan)][motor_name]
            for scan in scans
            if (
                int(scan) in motor_values_by_scan
                and motor_name in motor_values_by_scan[int(scan)]
            )
        }
    else:
        energies = get_motor_value(
            scans,
            motor_name=motor_name,
            spec_dir=spec_dir,
        )

    elastic_lines = []

    # Validate and convert all SPEC energies first.
    numeric_energies: dict[int, float] = {}

    for scan_num in scans:
        scan_num = int(scan_num)

        if scan_num not in energies:
            raise ValueError(
                f"No SPEC energy value found for calibration scan {scan_num} "
                f"using motor '{motor_name}'."
            )

        try:
            numeric_energies[scan_num] = float(energies[scan_num])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Energy value for scan {scan_num} is not numeric: "
                f"{energies[scan_num]!r}"
            ) from exc

    # Find all calibration scan files with one call.
    if scan_files_by_scan is None:
        scan_files = _find_scan_files(
            scans,
            scans_dir=None,
            search_roots=search_roots,
            fallback_recursive_search=fallback_recursive_search,
        )
    else:
        scan_files = [
            scan_files_by_scan[int(scan)]
            for scan in scans
        ]

    # Calibration requires a complete x/y pair for every requested scan.
    if len(scan_files) != scans.size:
        raise FileNotFoundError(
            "Not all calibration scan files were found. "
            f"Requested {scans.size} scans, but found "
            f"{len(scan_files)} complete x/y file pairs."
        )

    elastic_lines = []

    for scan_num, scan_file_pair in zip(
        scans,
        scan_files,
        strict=True,
    ):
        scan_num = int(scan_num)
        energy = numeric_energies[scan_num]

        pixel_axis, spectrum = _build_calibration_spectrum(
            scan_num,
            [scan_file_pair],
            percentile,
        )

        cut_value = np.percentile(spectrum, 90)
        mask = spectrum > cut_value
        changes = np.diff(mask.astype(int))

        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        if mask[0]:
            starts = np.r_[0, starts]

        if mask[-1]:
            ends = np.r_[ends, len(mask)]

        if starts.size == 0 or ends.size == 0:
            raise ValueError(
                f"No peak region found for calibration scan {scan_num}."
            )

        lengths = ends - starts
        longest_region = int(np.argmax(lengths))

        start = int(starts[longest_region])
        end = int(ends[longest_region])

        buffered_start = max(start - buffer, 0)
        buffered_end = min(end + buffer, spectrum.size)
        x_full = pixel_axis[buffered_start:buffered_end]
        y_full = spectrum[buffered_start:buffered_end]

        area = float(np.sum(y_full))

        if area <= 0:
            raise ValueError(
                f"Peak region for calibration scan {scan_num} is empty."
            )

        barycenter = float(
            np.sum(x_full * y_full) / area
        )

        elastic_lines.append(
            [
                x_full,
                y_full / area,
                barycenter,
                energy,
            ]
        )

    if double_gaussian_model:
        poly_orders = {
            "sigma1": params_poly_orders[0],
            "sigma2": params_poly_orders[1],
            "R": params_poly_orders[2],
            "delta": params_poly_orders[3],
        }
        initial_values = {
            "sigma1": 20.0,
            "sigma2": 20.0,
            "R": 2.0,
            "delta": 15.0,
        }
        bounds = {
            "sigma1": (1.0, 100.0),
            "sigma2": (1.0, 100.0),
            "R": (0.05, 20.0),
            "delta": (1.0, 100.0),
        }
    else:
        poly_orders = {"sigma": 2}
        initial_values = {"sigma": 20.0}
        bounds = {"sigma": (1.0, 100.0)}

    centers_of_mass = np.asarray(
        [line[2] for line in elastic_lines],
        dtype=float,
    )

    def residual_global(params, lines):
        residuals = []

        for i, (x, y, barycenter, _energy) in enumerate(lines):
            z = scaled_coordinate(barycenter)
            mu = params[f"mu_bar_{i}"].value
            A = params[f"A_{i}"].value
            offset = params[f"offset_{i}"].value

            if double_gaussian_model:
                sigma1 = get_poly_param(params, "sigma1", z, poly_orders)
                sigma2 = get_poly_param(params, "sigma2", z, poly_orders)
                R = get_poly_param(params, "R", z, poly_orders)
                delta = get_poly_param(params, "delta", z, poly_orders)

                y_model = double_gaussian(
                    x,
                    mu_bar=mu,
                    sigma1=sigma1,
                    sigma2=sigma2,
                    A=A,
                    R=R,
                    delta=delta,
                    offset=offset,
                )
            else:
                sigma = get_poly_param(params, "sigma", z, poly_orders)
                y_model = single_gaussian(
                    x,
                    mu=mu,
                    sigma=sigma,
                    A=A,
                    offset=offset,
                )

            residuals.append(y_model - y)

        z_grid = np.linspace(
            scaled_coordinate(0),
            scaled_coordinate(4096),
            200,
        )
        penalty = []
        penalty_strength = 1e3
        epsilon = 1e-8

        for name in poly_orders:
            values = np.asarray(
                [
                    get_poly_param(params, name, z, poly_orders)
                    for z in z_grid
                ]
            )
            violation = np.minimum(values - epsilon, 0.0)
            penalty.append(penalty_strength * violation)

        return np.concatenate(residuals + penalty)

    params = Parameters()  # type: ignore

    for name, order in poly_orders.items():
        minimum, maximum = bounds[name]
        params.add(
            f"{name}_c0",
            value=initial_values[name],
            min=minimum,
            max=maximum,
        )

        for coefficient_index in range(1, order + 1):
            params.add(
                f"{name}_c{coefficient_index}",
                value=0.5,
            )

    for i, (x, y, barycenter, _energy) in enumerate(elastic_lines):
        A_guess = float(np.trapezoid(y - np.min(y), x))
        if A_guess <= 0:
            A_guess = 1.0

        params.add(
            f"mu_bar_{i}",
            value=barycenter,
            min=barycenter - 50,
            max=barycenter + 50,
        )
        params.add(f"A_{i}", value=A_guess, min=0)
        params.add(
            f"offset_{i}",
            value=1e-5,
            min=1e-6,
            max=1e-3,
        )

    result = minimize(  # type: ignore
        residual_global,
        params,
        args=(elastic_lines,),
        method="least_squares",
    )

    fitted_lines = []
    calibration_pairs = []

    for i, (x, y, barycenter, energy) in enumerate(elastic_lines):
        z = scaled_coordinate(barycenter)
        mu = float(result.params[f"mu_bar_{i}"].value)  # type: ignore
        A = float(result.params[f"A_{i}"].value)  # type: ignore
        offset = float(result.params[f"offset_{i}"].value)  # type: ignore
        elastic_lines[i][2] = mu
        calibration_pairs.append((energy, mu))

        if double_gaussian_model:
            sigma1 = get_poly_param(result.params, "sigma1", z, poly_orders)  # type: ignore
            sigma2 = get_poly_param(result.params, "sigma2", z, poly_orders)  # type: ignore
            R = get_poly_param(result.params, "R", z, poly_orders)  # type: ignore
            delta = get_poly_param(result.params, "delta", z, poly_orders)  # type: ignore

            y_fit = double_gaussian(
                x,
                mu_bar=mu,
                sigma1=sigma1,
                sigma2=sigma2,
                A=A,
                R=R,
                delta=delta,
                offset=offset,  # type: ignore
            )
            gaussian_1, gaussian_2 = double_gaussian_components(
                x,
                mu_bar=mu,
                sigma1=sigma1,
                sigma2=sigma2,
                A=A,
                R=R,
                delta=delta,
            )

            fitted_lines.append(
                {
                    "x": x,
                    "y": y,
                    "mu": mu,
                    "scan": int(scans[i]),
                    "energy": float(energy),
                    "y_fit": y_fit,
                    "component_1": gaussian_1 + offset,
                    "component_2": gaussian_2 + offset,
                    "sigma1": float(sigma1),
                    "sigma2": float(sigma2),
                    "R": float(R),
                    "delta": float(delta),
                }
            )
        else:
            sigma = get_poly_param(result.params, "sigma", z, poly_orders)  # type: ignore
            y_fit = single_gaussian(
                x,
                mu=mu,
                sigma=sigma,
                A=A,
                offset=offset,  # type: ignore
            )
            fitted_lines.append(
                {
                    "x": x,
                    "y": y,
                    "mu": mu,
                    "scan": int(scans[i]),
                    "energy": float(energy),
                    "y_fit": y_fit,
                    "sigma": float(sigma),
                }
            )

    calibration_pairs = np.asarray(calibration_pairs, dtype=float)
    energies_eV = calibration_pairs[:, 0]
    energies_pixel = calibration_pairs[:, 1]

    coefficients = np.polyfit(
        energies_pixel,
        energies_eV,
        fit_poly_order,
    )
    a_3, a_2, a_1, a_0 = get_coefficients(coefficients)

    fitted_energies_eV = (
        a_3 * energies_pixel**3
        + a_2 * energies_pixel**2
        + a_1 * energies_pixel
        + a_0
    )
    deviations_eV = fitted_energies_eV - energies_eV
    degrees_of_freedom = max(
        int(len(deviations_eV) - (fit_poly_order + 1)),
        1,
    )
    sigma_calibration_eV = float(
        np.sqrt(np.sum(deviations_eV**2) / degrees_of_freedom)
    )

    gaussian_fwhm_factor = 2.0 * np.sqrt(2.0 * np.log(2.0))
    line_fwhm_pixel: list[float] = []
    line_fwhm_eV: list[float] = []

    for fitted_line in fitted_lines:
        if double_gaussian_model:
            ratio = float(fitted_line["R"])
            weight_1 = ratio / (1.0 + ratio)
            weight_2 = 1.0 / (1.0 + ratio)
            weighted_sigma = (
                weight_1 * float(fitted_line["sigma1"])
                + weight_2 * float(fitted_line["sigma2"])
            )
            fwhm_pixel = float(gaussian_fwhm_factor * weighted_sigma)
        else:
            fwhm_pixel = float(
                gaussian_fwhm_factor * float(fitted_line["sigma"])
            )

        center_pixel = float(fitted_line["mu"])
        local_dispersion = abs(
            3.0 * a_3 * center_pixel**2
            + 2.0 * a_2 * center_pixel
            + a_1
        )
        fwhm_eV = float(fwhm_pixel * local_dispersion)
        fitted_line["fwhm_pixel"] = fwhm_pixel
        fitted_line["fwhm_eV"] = fwhm_eV
        line_fwhm_pixel.append(fwhm_pixel)
        line_fwhm_eV.append(fwhm_eV)

    mean_fwhm_eV = float(np.mean(line_fwhm_eV))

    parameter_coefficients = {
        name: [
            float(result.params[f"{name}_c{coefficient_index}"].value)  # type: ignore
            for coefficient_index in range(int(order) + 1)
        ]
        for name, order in poly_orders.items()
    }

    pixel_plot = np.linspace(
        np.min(energies_pixel),
        np.max(energies_pixel),
        500,
    )
    energy_plot = (
        a_3 * pixel_plot**3
        + a_2 * pixel_plot**2
        + a_1 * pixel_plot
        + a_0
    )

    calibration_figures: dict[str, Figure] = {}

    def create_calibration_subplots(
        number_of_rows: int = 1,
        number_of_columns: int = 1,
        *,
        figsize: tuple[float, float],
        squeeze: bool = True,
    ):
        if return_figures:
            figure = Figure(figsize=figsize)
            axes = figure.subplots(
                number_of_rows,
                number_of_columns,
                squeeze=squeeze,
            )
            return figure, axes
        return plt.subplots(
            number_of_rows,
            number_of_columns,
            figsize=figsize,
            squeeze=squeeze,
        )

    if show_calibration_plots or return_figures:
        z_plot = np.linspace(
            np.min(scaled_coordinate(centers_of_mass)),
            np.max(scaled_coordinate(centers_of_mass)),
            300,
        )
        com_plot = z_plot * com_scale_factor + com_scale_factor

        number_of_parameters = len(poly_orders)
        number_of_columns = min(2, number_of_parameters)
        number_of_rows = int(
            np.ceil(number_of_parameters / number_of_columns)
        )

        dark_style(figsize=(4, 4))
        fig_parameters, axes_parameters = create_calibration_subplots(
            number_of_rows,
            number_of_columns,
            figsize=(5 * number_of_columns, 4 * number_of_rows),
            squeeze=False,
        )
        calibration_figures["parameters"] = fig_parameters
        label_dictionary= {}
        label_dictionary_short = {}

        label_dictionary['sigma1'] = r'Gaussian 1 $\sigma$'
        label_dictionary_short['sigma1'] = r'$\sigma_1$'
        label_dictionary['sigma2'] = r'Gaussian 2 $\sigma$'
        label_dictionary_short['sigma2'] = r'$\sigma_2$'
        label_dictionary['R'] = r'Gaussian 1 / Gaussian 2 amplitude ratio'
        label_dictionary_short['R'] = r'$R$'
        label_dictionary['delta'] = r'Gaussian 1 - Gaussian 2 center separation'
        label_dictionary_short['delta'] = r'$\delta$'

        label_dictionary['sigma'] = r'Gaussian $\sigma$'
        label_dictionary_short['sigma'] = r'$\sigma$'

        c_o_m = []
        for fitted_line in fitted_lines:
                center_pixel = float(fitted_line["mu"])
                c_o_m.append(center_pixel)

        for i, name in enumerate(poly_orders):
            axis = axes_parameters[
                i // number_of_columns,
                i % number_of_columns,
            ]
            values_lines = np.asarray(
                [
                    get_poly_param(result.params, name, scaled_coordinate(com), poly_orders)  # type: ignore
                    for com in c_o_m
                ]
            )
            values = np.asarray(
                [
                    get_poly_param(result.params, name, z, poly_orders)  # type: ignore
                    for z in z_plot
                ]
            )
            axis.plot(com_plot, values,  color = 'white', lw=2, label=label_dictionary[name], zorder = 1)
            axis.plot(c_o_m, values_lines, 'o', mfc = 'lime', mec = 'white', markersize=12, zorder = 2)
            axis.set_title(label_dictionary[name])
            axis.set_xlabel("Detector position (pixel)")
            axis.set_ylabel(f"{label_dictionary_short[name]}")


        for i in range(number_of_parameters, axes_parameters.size):
            axes_parameters.flat[i].set_visible(False)

        fig_parameters.tight_layout()

        number_of_fit_columns = 4
        number_of_fit_rows = int(
            np.ceil(len(fitted_lines) / number_of_fit_columns)
        )

        dark_style(figsize=(4, 4))
        fig_fits, axes_fits = create_calibration_subplots(
            number_of_fit_rows,
            number_of_fit_columns,
            figsize=(3* 3, 3 * number_of_fit_rows),
            squeeze=False,
        )
        calibration_figures["line_fits"] = fig_fits

        for i, fitted_line in enumerate(fitted_lines):
            axis = axes_fits[
                i // number_of_fit_columns,
                i % number_of_fit_columns,
            ]
            axis.plot(fitted_line["x"], fitted_line["y"], label="Data", color = (1,1,1,.75))

            if double_gaussian_model:
                axis.plot(
                    fitted_line["x"],
                    fitted_line["component_1"],
                    "--",
                    label="Gaussian 1",
                    color = 'lime'
                )
                axis.plot(
                    fitted_line["x"],
                    fitted_line["component_2"],
                    "--",
                    label="Gaussian 2",
                    color = 'magenta'
                )
                fit_label = "Double Gaussian fit"
            else:
                fit_label = "Gaussian fit"

            axis.plot(
                fitted_line["x"],
                fitted_line["y_fit"],
                "-",
                label=fit_label,
                color = 'black',
                lw = 2
            )
            axis.axvline(
                fitted_line["mu"],
                linestyle="--",
                linewidth=1,
                color = 'yellow',
                label="Fitted center",
            )
            axis.text(
                0.02,
                0.98,
                (
                    f"{i + 1} · Scan {int(fitted_line['scan'])}\n"
                    f"$E_{{nom}}$ = {float(fitted_line['energy']):.3f} eV"
                ),
                transform=axis.transAxes,
                ha="left",
                va="top",
                bbox=dict(
                    facecolor="gray",
                    edgecolor="none",
                    alpha=0.5,
                    boxstyle="round,pad=0.3",
                ),
            )
            axis.set_yticklabels([])

            y_min, y_max = axis.get_ylim()
            axis.set_ylim(y_min, y_max + 0.2 * (y_max - y_min))

        for i in range(len(fitted_lines), axes_fits.size):
            axes_fits.flat[i].set_visible(False)

        legend_handles, legend_labels = (
            axes_fits.flat[0].get_legend_handles_labels()
        )
        fig_fits.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            fontsize=14,
            bbox_to_anchor=(0.5, 1.0),
            ncol=min(len(legend_labels), 5),
        )
        fig_fits.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

        dark_style(figsize=(8, 6))
        fig_calibration, axis_calibration = create_calibration_subplots(
            figsize=(10, 8)
        )
        calibration_figures["energy_calibration"] = fig_calibration
        axis_calibration.plot(
            pixel_plot,
            energy_plot,
            label=f"Polynomial order {fit_poly_order}", lw = 2, color = 'white',
        )
        axis_calibration.plot(
            energies_pixel,
            energies_eV,
            "+", markersize=22, markerfacecolor="lime", markeredgecolor="lime", markeredgewidth=2,
            label="Calibration points",
        )
        axis_calibration.set_xlabel("Detector position / pixel")
        axis_calibration.set_ylabel("Energy / eV")
        axis_calibration.legend()
        fig_calibration.tight_layout()

    if show_calibration_plots or show_test_plot or return_figures:
        dark_style(figsize=(8, 6))
        fig_test, axis_test = create_calibration_subplots(figsize=(10, 8))
        calibration_figures["energy_deviation"] = fig_test
        colors = plt.colormaps["tab20"](
            np.linspace(0, 1, len(elastic_lines))
        )
        for i, (x, y, barycenter, energy) in enumerate(elastic_lines):
            calibrated_x = (
                a_3 * x**3
                + a_2 * x**2
                + a_1 * x
                + a_0
            )
            deviation = (
                a_3 * barycenter**3
                + a_2 * barycenter**2
                + a_1 * barycenter
                + a_0
                - energy
            )
            x_fit = fitted_lines[i]["x"]
            x_fit_calibrated = (
                a_3 * x_fit**3
                + a_2 * x_fit**2
                + a_1 * x_fit
                + a_0
            )
            axis_test.plot(x_fit_calibrated - energy, fitted_lines[i]["y_fit"],'-', lw=2,
                           color=colors[i], label=f"Scan {scans[i]}: {energy:.2f} eV",)

            x_plot, y_plot = bin_data(calibrated_x - energy, y, 1)
            axis_test.plot(
                x_plot,
                y_plot,
                color=colors[i],
                lw = .5,
            )
            axis_test.axvline(
                deviation,
                color=colors[i],
                linestyle="--",
                linewidth=.75,
            )

        axis_test.text(
            0.98,
            0.98,
            (
                rf"Cal. $\sigma$: {sigma_calibration_eV:.3f} eV"
                "\n"
                rf"Mean FWHM: {mean_fwhm_eV:.3f} eV"
            ),
            transform=axis_test.transAxes,
            ha="right",
            va="top",
        )
        axis_test.legend(fontsize=14)
        axis_test.set_xlabel("Deviation from reference energy (eV)")
        fig_test.tight_layout()

    if show_calibration_plots or show_test_plot:
        plt.show()

    energy_coefficients = (
        float(a_3),
        float(a_2),
        float(a_1),
        float(a_0),
    )

    result_payload: EnergyCalibrationPayload = energy_coefficients

    if return_details:
        calibration_lines = []

        for fitted_line in fitted_lines:
            plot_data = {
                "x_pixel": np.asarray(fitted_line["x"], dtype=float).tolist(),
                "counts": np.asarray(fitted_line["y"], dtype=float).tolist(),
                "fit_counts": np.asarray(
                    fitted_line["y_fit"],
                    dtype=float,
                ).tolist(),
            }
            line_details = {
                "scan": int(fitted_line["scan"]),
                "nominal_energy_eV": float(fitted_line["energy"]),
                "fitted_center_pixel": float(fitted_line["mu"]),
                "fwhm_pixel": float(fitted_line["fwhm_pixel"]),
                "fwhm_eV": float(fitted_line["fwhm_eV"]),
                "plot_data": plot_data,
            }

            if double_gaussian_model:
                line_details["sigma1_pixel"] = float(fitted_line["sigma1"])
                line_details["sigma2_pixel"] = float(fitted_line["sigma2"])
                line_details["amplitude_ratio"] = float(fitted_line["R"])
                line_details["center_separation_pixel"] = float(
                    fitted_line["delta"]
                )
                plot_data["component_1_counts"] = np.asarray(
                    fitted_line["component_1"],
                    dtype=float,
                ).tolist()
                plot_data["component_2_counts"] = np.asarray(
                    fitted_line["component_2"],
                    dtype=float,
                ).tolist()
            else:
                line_details["sigma_pixel"] = float(fitted_line["sigma"])

            calibration_lines.append(line_details)

        result_payload = {
            "version": 1,
            "energy_coefficients": list(energy_coefficients),
            "energy_polynomial_order": int(fit_poly_order),
            "peak_selection_buffer": int(buffer),
            "calibration_scans": [int(scan) for scan in scans],
            "line_shape_model": (
                "double_gaussian"
                if double_gaussian_model
                else "single_gaussian"
            ),
            "line_shape_coordinate": {
                "center_pixel": float(com_scale_factor),
                "scale_pixel": float(com_scale_factor),
            },
            "line_shape_polynomial_orders": {
                name: int(order) for name, order in poly_orders.items()
            },
            "line_shape_polynomial_coefficients": parameter_coefficients,
            "calibration_lines": calibration_lines,
            "calibration_sigma_eV": sigma_calibration_eV,
            "mean_fwhm_eV": mean_fwhm_eV,
            "fwhm_definition": (
                "Amplitude-weighted component FWHM"
                if double_gaussian_model
                else "Gaussian FWHM"
            ),
        }

    if return_figures:
        return result_payload, calibration_figures

    return result_payload


def build_calibration_figures_from_details(
    details: dict,
) -> dict[str, Figure]:
    """Rebuild the four diagnostic figures from a saved calibration payload."""

    if not isinstance(details, dict):
        raise TypeError("Calibration details must be a dictionary.")

    calibration_lines = details.get("calibration_lines")
    if not isinstance(calibration_lines, list) or not calibration_lines:
        raise ValueError("The calibration cache contains no calibration lines.")
    if any(
        not isinstance(line, dict) or not isinstance(line.get("plot_data"), dict)
        for line in calibration_lines
    ):
        raise ValueError(
            "The saved calibration predates cached diagnostic plot data."
        )

    coefficients = np.asarray(details.get("energy_coefficients", []), dtype=float)
    if coefficients.size != 4:
        raise ValueError("The calibration cache has invalid energy coefficients.")

    model_name = str(details.get("line_shape_model", "single_gaussian"))
    is_double = model_name == "double_gaussian"
    parameter_orders = details.get("line_shape_polynomial_orders", {})
    parameter_coefficients = details.get(
        "line_shape_polynomial_coefficients",
        {},
    )
    coordinate = details.get("line_shape_coordinate", {})
    if not isinstance(parameter_orders, dict) or not isinstance(
        parameter_coefficients,
        dict,
    ):
        raise ValueError("The calibration cache has invalid line-shape data.")

    center_pixel = float(coordinate.get("center_pixel", 2048.0))
    scale_pixel = float(coordinate.get("scale_pixel", 2048.0))
    if scale_pixel == 0:
        scale_pixel = 2048.0

    dark_style(figsize=(8, 6))
    figures: dict[str, Figure] = {}

    parameter_labels = {
        "sigma1": (r"Gaussian 1 $\sigma$", r"$\sigma_1$", "sigma1_pixel"),
        "sigma2": (r"Gaussian 2 $\sigma$", r"$\sigma_2$", "sigma2_pixel"),
        "R": (
            "Gaussian 1 / Gaussian 2 amplitude ratio",
            r"$R$",
            "amplitude_ratio",
        ),
        "delta": (
            "Gaussian 1 - Gaussian 2 center separation",
            r"$\delta$",
            "center_separation_pixel",
        ),
        "sigma": (r"Gaussian $\sigma$", r"$\sigma$", "sigma_pixel"),
    }
    parameter_names = [
        name for name in parameter_orders if name in parameter_labels
    ]
    if not parameter_names:
        parameter_names = ["sigma1", "sigma2", "R", "delta"] if is_double else ["sigma"]

    parameter_columns = min(2, len(parameter_names))
    parameter_rows = int(np.ceil(len(parameter_names) / parameter_columns))
    parameter_figure = Figure(
        figsize=(5 * parameter_columns, 4 * parameter_rows)
    )
    parameter_axes = parameter_figure.subplots(
        parameter_rows,
        parameter_columns,
        squeeze=False,
    )
    figures["parameters"] = parameter_figure

    fitted_centers = np.asarray(
        [float(line["fitted_center_pixel"]) for line in calibration_lines],
        dtype=float,
    )
    curve_pixels = np.linspace(
        float(np.min(fitted_centers)),
        float(np.max(fitted_centers)),
        300,
    )
    scaled_curve = (curve_pixels - center_pixel) / scale_pixel

    for index, name in enumerate(parameter_names):
        axis = parameter_axes[index // parameter_columns, index % parameter_columns]
        title, short_label, value_key = parameter_labels[name]
        coeffs = np.asarray(parameter_coefficients.get(name, []), dtype=float)
        if coeffs.size:
            curve_values = np.polynomial.polynomial.polyval(scaled_curve, coeffs)
            axis.plot(curve_pixels, curve_values, color="white", lw=2, zorder=1)
        point_values = np.asarray(
            [float(line[value_key]) for line in calibration_lines],
            dtype=float,
        )
        axis.plot(
            fitted_centers,
            point_values,
            "o",
            mfc="lime",
            mec="white",
            markersize=12,
            zorder=2,
        )
        axis.set_title(title)
        axis.set_xlabel("Detector position (pixel)")
        axis.set_ylabel(short_label)

    for index in range(len(parameter_names), parameter_axes.size):
        parameter_axes.flat[index].set_visible(False)
    parameter_figure.tight_layout()

    fit_columns = 4
    fit_rows = int(np.ceil(len(calibration_lines) / fit_columns))
    fits_figure = Figure(figsize=(9, 3 * fit_rows))
    fit_axes = fits_figure.subplots(fit_rows, fit_columns, squeeze=False)
    figures["line_fits"] = fits_figure

    for index, line in enumerate(calibration_lines):
        axis = fit_axes[index // fit_columns, index % fit_columns]
        plot_data = line["plot_data"]
        x_values = np.asarray(plot_data["x_pixel"], dtype=float)
        counts = np.asarray(plot_data["counts"], dtype=float)
        fit_counts = np.asarray(plot_data["fit_counts"], dtype=float)
        axis.plot(x_values, counts, label="Data", color=(1, 1, 1, 0.75))
        if is_double:
            axis.plot(
                x_values,
                np.asarray(plot_data["component_1_counts"], dtype=float),
                "--",
                label="Gaussian 1",
                color="lime",
            )
            axis.plot(
                x_values,
                np.asarray(plot_data["component_2_counts"], dtype=float),
                "--",
                label="Gaussian 2",
                color="magenta",
            )
            fit_label = "Double Gaussian fit"
        else:
            fit_label = "Gaussian fit"
        axis.plot(x_values, fit_counts, "-", label=fit_label, color="black", lw=2)
        axis.axvline(
            float(line["fitted_center_pixel"]),
            linestyle="--",
            linewidth=1,
            color="yellow",
            label="Fitted center",
        )
        axis.text(
            0.02,
            0.98,
            (
                f"{index + 1} · Scan {int(line['scan'])}\n"
                f"$E_{{nom}}$ = {float(line['nominal_energy_eV']):.3f} eV"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            bbox={
                "facecolor": "gray",
                "edgecolor": "none",
                "alpha": 0.5,
                "boxstyle": "round,pad=0.3",
            },
        )
        axis.set_yticklabels([])
        y_min, y_max = axis.get_ylim()
        axis.set_ylim(y_min, y_max + 0.2 * (y_max - y_min))

    for index in range(len(calibration_lines), fit_axes.size):
        fit_axes.flat[index].set_visible(False)
    handles, labels = fit_axes.flat[0].get_legend_handles_labels()
    fits_figure.legend(
        handles,
        labels,
        loc="upper center",
        fontsize=14,
        bbox_to_anchor=(0.5, 1.0),
        ncol=min(len(labels), 5),
    )
    fits_figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    calibration_figure = Figure(figsize=(10, 8))
    calibration_axis = calibration_figure.subplots()
    figures["energy_calibration"] = calibration_figure
    pixel_plot = np.linspace(
        float(np.min(fitted_centers)),
        float(np.max(fitted_centers)),
        500,
    )
    energy_plot = np.polyval(coefficients, pixel_plot)
    nominal_energies = np.asarray(
        [float(line["nominal_energy_eV"]) for line in calibration_lines],
        dtype=float,
    )
    calibration_axis.plot(
        pixel_plot,
        energy_plot,
        label=f"Polynomial order {int(details.get('energy_polynomial_order', 3))}",
        lw=2,
        color="white",
    )
    calibration_axis.plot(
        fitted_centers,
        nominal_energies,
        "+",
        markersize=22,
        markerfacecolor="lime",
        markeredgecolor="lime",
        markeredgewidth=2,
        label="Calibration points",
    )
    calibration_axis.set_xlabel("Detector position / pixel")
    calibration_axis.set_ylabel("Energy / eV")
    calibration_axis.legend()
    calibration_figure.tight_layout()

    deviation_figure = Figure(figsize=(10, 8))
    deviation_axis = deviation_figure.subplots()
    figures["energy_deviation"] = deviation_figure
    colors = plt.colormaps["tab20"](
        np.linspace(0, 1, len(calibration_lines))
    )
    for index, line in enumerate(calibration_lines):
        plot_data = line["plot_data"]
        x_values = np.asarray(plot_data["x_pixel"], dtype=float)
        counts = np.asarray(plot_data["counts"], dtype=float)
        fit_counts = np.asarray(plot_data["fit_counts"], dtype=float)
        nominal_energy = float(line["nominal_energy_eV"])
        calibrated_x = np.polyval(coefficients, x_values) - nominal_energy
        deviation_axis.plot(
            calibrated_x,
            fit_counts,
            "-",
            lw=2,
            color=colors[index],
            label=f"Scan {int(line['scan'])}: {nominal_energy:.2f} eV",
        )
        deviation_axis.plot(calibrated_x, counts, color=colors[index], lw=0.5)
        deviation = (
            float(np.polyval(coefficients, float(line["fitted_center_pixel"])))
            - nominal_energy
        )
        deviation_axis.axvline(
            deviation,
            color=colors[index],
            linestyle="--",
            linewidth=0.75,
        )
    deviation_axis.text(
        0.98,
        0.98,
        (
            rf"Cal. $\sigma$: {float(details['calibration_sigma_eV']):.3f} eV"
            "\n"
            rf"Mean FWHM: {float(details['mean_fwhm_eV']):.3f} eV"
        ),
        transform=deviation_axis.transAxes,
        ha="right",
        va="top",
    )
    deviation_axis.legend(fontsize=14)
    deviation_axis.set_xlabel("Deviation from reference energy (eV)")
    deviation_figure.tight_layout()

    return figures


# =============================================================================
# Histogram core: this is the main refactor
# =============================================================================

Array2D = np.ndarray
HistogramTransform = Callable[[Array2D], Array2D]
SpectrumTransform = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(slots=True)
class HistogramSettings:
    """All user-controlled values that affect the current histogram products."""

    tilt: float = 0.0
    lower_percentile: float = 0.0
    upper_percentile: float = 100.0
    bottom_cut: int = 0
    top_cut: int = 0
    spectrum_bin: int = 1
    display_bin_x: int = 1
    display_bin_y: int = 1

    symmetric_fill_enabled: bool = False

    median_filter_enabled: bool = False
    median_filter_window: int = 3

    local_filter_enabled: bool = False
    local_filter_window: int = 15
    local_filter_bottom_limit: int = 1
    local_filter_upper_limit: int = 99


@dataclass(slots=True)
class HistogramProducts:
    """
    Derived products for the current settings.

    base:     tilted and optionally symmetrically filled histogram
    filtered: base after percentile filtering
    cut:      filtered after row cut; this is the canonical current histogram
    display:  rebinned copy used only for imshow
    spectrum: 1D projection derived from cut
    """

    base: np.ndarray
    filtered: np.ndarray
    cut: np.ndarray
    display: np.ndarray
    spectrum_x: np.ndarray
    spectrum_y: np.ndarray
    x_edges: np.ndarray
    y_edges: np.ndarray
    lower_bound: float
    upper_bound: float
    bottom_cut: int
    top_cut: int
    top_index: int
    ridge: int | None = None
    local_filter_complement: np.ndarray | None = None


@dataclass(slots=True)
class HistogramHooks:
    """
    Extension points for future functions.

    after_base:      affects display, saved histogram and 1D spectrum
    before_display: affects only imshow
    before_spectrum: affects only the 1D spectrum
    """

    after_base: list[HistogramTransform] = field(default_factory=list)
    before_display: list[HistogramTransform] = field(default_factory=list)
    before_spectrum: list[SpectrumTransform] = field(default_factory=list)


def apply_transforms(histogram: np.ndarray, transforms: Iterable[HistogramTransform]) -> np.ndarray:
    result = histogram
    for transform in transforms:
        result = transform(result)
    return result


def normalize_median_window(value: int) -> int:
    """Clamp median-filter window to odd values between 3 and 49."""
    value = int(round(value))
    value = int(np.clip(value, 3, 49))
    if value % 2 == 0:
        value += 1 if value < 49 else -1
    return int(value)


@njit
def reflect_index(i, n):
    if n <= 1:
        return 0
    while i < 0 or i >= n:
        if i < 0:
            i = -i - 1
        else:
            i = 2 * n - i - 1
    return i


@njit(parallel=True)
def median_filter_numpy(arr, size, higher_idx=False):
    if size % 2 == 0:
        raise ValueError("size must be odd")
    if size < 3:
        raise ValueError("size must be >= 3")

    pad = size // 2
    h, w = arr.shape
    out = np.empty_like(arr)

    for i in prange(h):
        win = np.empty(size * size - 1, dtype=arr.dtype)

        for j in range(w):
            idx = 0

            for di in range(-pad, pad + 1):
                ii = reflect_index(i + di, h)  # type: ignore

                for dj in range(-pad, pad + 1):
                    jj = reflect_index(j + dj, w)  # type: ignore

                    if ii == i and jj == j:
                        continue

                    win[idx] = arr[ii, jj]
                    idx += 1

            sorted_win = np.sort(win)
            median_index = sorted_win.size // 2

            if higher_idx:
                higher_index = min(median_index + 5, sorted_win.size - 1)
                out[i, j] = sorted_win[higher_index]
            else:
                out[i, j] = sorted_win[median_index]

    return out

@njit(parallel=True)
def percentile_filter_numpy(arr, size, bottom_limit=5, upper_limit=95):
    if size % 2 == 0:
        raise ValueError("size must be odd")
    pad = size // 2
    h, w = arr.shape
    out = np.empty_like(arr)
    complement_out = np.empty_like(arr)
    if bottom_limit <= 0 and upper_limit >= 100:
        out[:] = arr
        complement_out[:] = 0
        return out, complement_out

    for i in prange(h):
        # The center pixel is intentionally included in the local window.
        win = np.empty(size * size, dtype=arr.dtype)
        for j in range(w):
            idx = 0
            for di in range(-pad, pad + 1):
                ii = reflect_index(i + di, h)  # type: ignore[misc]
                for dj in range(-pad, pad + 1):
                    jj = reflect_index(j + dj, w)  # type: ignore[misc]
                    win[idx] = arr[ii, jj]
                    idx += 1
            sorted_win = np.sort(win)
            median_index = sorted_win.size // 2
            bottom_percentile_index = int(sorted_win.size * bottom_limit / 100)
            upper_percentile_index = int(sorted_win.size * upper_limit / 100)
            bottom_percentile_index = min(max(bottom_percentile_index, 0), sorted_win.size - 1)
            upper_percentile_index = min(max(upper_percentile_index, 0), sorted_win.size - 1)
            if sorted_win[bottom_percentile_index] <= arr[i, j] <= sorted_win[upper_percentile_index]:
                out[i, j] = arr[i, j]
                complement_out[i, j] = 0
            else:
                complement_out[i, j] = 1
                out[i, j] = sorted_win[median_index]
    return out, complement_out


def median_filter(window_size: int, *, higher_idx: bool = False) -> HistogramTransform:
    window_size = normalize_median_window(window_size)

    def transform(histogram: np.ndarray) -> np.ndarray:
        return median_filter_numpy(histogram, window_size, higher_idx)

    return transform


def local_percentile_filter(window_size: int, *, bottom_limit: int = 5, upper_limit: int = 95) -> HistogramTransform:
    window_size = normalize_median_window(window_size)
    bottom_limit = int(np.clip(bottom_limit, 0, 99))
    upper_limit = int(np.clip(upper_limit, bottom_limit + 1, 100))

    def transform(histogram: np.ndarray) -> np.ndarray:
        filtered, _ = percentile_filter_numpy(histogram, window_size, bottom_limit, upper_limit)
        return filtered

    return transform

def normalize_hist_bin(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, (tuple, list)):
        if len(value) != 2:
            raise ValueError("hist_bin must be an int or a tuple (x_bin, y_bin).")
        x_bin, y_bin = value
    else:
        x_bin = y_bin = value

    x_bin = int(x_bin)
    y_bin = int(y_bin)
    if x_bin < 1 or y_bin < 1:
        raise ValueError("hist_bin values must be >= 1.")
    return x_bin, y_bin


def normalize_bin(value: int) -> int:
    return max(1, int(value))


def rebin_histogram_sum(histogram: np.ndarray, bin_x: int, bin_y: int) -> np.ndarray:
    """Sum-rebin a 2D histogram. Used for display only."""
    bin_x = normalize_bin(bin_x)
    bin_y = normalize_bin(bin_y)

    if bin_x == 1 and bin_y == 1:
        return histogram

    x_starts = np.arange(0, histogram.shape[0], bin_x)
    y_starts = np.arange(0, histogram.shape[1], bin_y)

    return np.add.reduceat(
        np.add.reduceat(histogram, x_starts, axis=0),
        y_starts,
        axis=1,
    )


def rebin_projection_mean(projection: np.ndarray, bin_size: int) -> tuple[np.ndarray, np.ndarray]:
    bin_size = normalize_bin(bin_size)
    n_bins = len(projection) // bin_size

    if n_bins == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    y = projection[: n_bins * bin_size]
    x = np.arange(len(y), dtype=float)

    binned_y = y.reshape(n_bins, bin_size).sum(axis=1) / bin_size
    binned_x = x.reshape(n_bins, bin_size).mean(axis=1)
    return binned_x, binned_y


def apply_percentile_filter(
    histogram: np.ndarray,
    lower_percentile: float,
    upper_percentile: float,
) -> tuple[np.ndarray, float, float]:
    lower_percentile = float(lower_percentile)
    upper_percentile = float(upper_percentile)

    if lower_percentile >= upper_percentile:
        upper_percentile = min(lower_percentile + 0.1, 100.0)

    nonzero = histogram[histogram > 0]
    if nonzero.size == 0:
        return np.zeros_like(histogram), 0.0, 0.0

    lower_bound = float(np.percentile(nonzero, lower_percentile))
    upper_bound = float(np.percentile(nonzero, upper_percentile))

    filtered = np.where(
        (histogram >= lower_bound) & (histogram <= upper_bound),
        histogram,
        0,
    )
    return filtered, lower_bound, upper_bound


def apply_row_cut(
    histogram: np.ndarray,
    bottom_cut: int,
    top_cut: int,
) -> tuple[np.ndarray, int, int, int]:
    bottom_cut = max(0, int(bottom_cut))
    top_cut = max(0, int(top_cut))

    top_index = histogram.shape[1] - top_cut if top_cut > 0 else histogram.shape[1]
    top_index = max(top_index, bottom_cut + 1)
    top_index = min(top_index, histogram.shape[1])

    return histogram[:, bottom_cut:top_index], bottom_cut, top_cut, top_index


def build_histogram_fast(
    x: np.ndarray,
    y: np.ndarray,
    *,
    tilt: float,
    x_min: int = 0,
    x_max: int = 4095,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    corrected_y = y + float(tilt) * x
    finite_mask = np.isfinite(x) & np.isfinite(corrected_y)
    x_valid = x[finite_mask]
    y_valid = corrected_y[finite_mask]

    if x_valid.size == 0:
        raise ValueError("No finite x/y values available for histogram.")

    y_min = int(np.floor(y_valid.min()))
    y_max = int(np.ceil(y_valid.max())) + 1

    x_edges = np.arange(x_min, x_max + 2, dtype=np.float64)
    y_edges = np.arange(y_min, y_max + 1, dtype=np.float64)

    x_bins = len(x_edges) - 1
    y_bins = len(y_edges) - 1

    x_idx = np.floor(x_valid - x_min).astype(np.int64, copy=False)
    y_idx = np.floor(y_valid - y_min).astype(np.int64, copy=False)

    inside = (
        (x_idx >= 0)
        & (x_idx < x_bins)
        & (y_idx >= 0)
        & (y_idx < y_bins)
    )

    flat_idx = x_idx[inside] * y_bins + y_idx[inside]
    histogram = np.bincount(flat_idx, minlength=x_bins * y_bins).reshape(x_bins, y_bins)
    return histogram, x_edges, y_edges


def mirror_horizontal(histogram: np.ndarray, y0: float) -> np.ndarray:
    """Mirror a histogram horizontally around row index y0."""
    mirrored = np.zeros_like(histogram)
    n_y = histogram.shape[1]

    for y in range(n_y):
        y_mirror = int(-y + 2 * y0)
        if 0 <= y_mirror < n_y:
            mirrored[:, y_mirror] = histogram[:, y]

    return mirrored


class HistogramPipeline:
    """Owns x/y arrays and computes all current histogram products from settings."""

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        x_min: int = 0,
        x_max: int = 4095,
        hooks: HistogramHooks | None = None,
    ) -> None:
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.x_min = int(x_min)
        self.x_max = int(x_max)
        self.hooks = hooks or HistogramHooks()
        self._base_cache_key: tuple | None = None
        self._base_cache_value: tuple[np.ndarray, np.ndarray, np.ndarray, int | None, np.ndarray | None] | None = None

    def clear_cache(self) -> None:
        self._base_cache_key = None
        self._base_cache_value = None

    @staticmethod
    def _fit_ridge(
        histogram: np.ndarray,
        bottom_edge: int,
        top_edge: int,
    ) -> float | None:
        # Both ridge estimation paths use the same Gaussian projection fit.
        cropped_histogram = histogram[:, bottom_edge:top_edge]
        if cropped_histogram.shape[1] < 3:
            return None

        y_projection = np.sum(cropped_histogram, axis=0)
        if not np.any(y_projection > 0):
            return None

        model = Model(gauss)
        params = model.make_params(
            amp=float(np.max(y_projection)),
            mu=cropped_histogram.shape[1] / 2,
            sigma=50,
            offset=0,
        )
        params["amp"].min = 1
        params["sigma"].min = 10
        params['mu'].min = 1
        params['mu'].max = cropped_histogram.shape[1]
        result = model.fit(
            y_projection,
            params,
            x=np.arange(cropped_histogram.shape[1]),
        )
        return float(result.params["mu"].value) + bottom_edge

    def build_symmetric_histogram(
        self,
        tilt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int | None]:
        """Your original symmetric-fill logic, moved into the pipeline."""
        if abs(float(tilt)) < 1e-15:
            hist, x_edges, y_edges = build_histogram_fast(
                self.x,
                self.y,
                tilt=tilt,
                x_min=self.x_min,
                x_max=self.x_max,
            )
            return hist, x_edges, y_edges, None

        x_ = self.x
        y_ = self.y
        y_2 = y_ + float(tilt) * x_

        finite = np.isfinite(x_) & np.isfinite(y_2)
        x_valid = x_[finite]
        y_valid = y_2[finite]
        if x_valid.size == 0:
            raise ValueError("No finite x/y values available for symmetric histogram.")

        bins_x = np.arange(self.x_min, self.x_max + 2, dtype=np.float64)
        bins_y = np.arange(0, int(np.ceil(np.max(y_valid))) + 2, dtype=np.float64)
        hist, _, _ = np.histogram2d(x_valid, y_valid, bins=[bins_x, bins_y])

        xc = 0.5 * (bins_x[:-1] + bins_x[1:])
        yc = 0.5 * (bins_y[:-1] + bins_y[1:])
        X, Y = np.meshgrid(xc, yc, indexing="ij")

        hist_work = hist.copy()
        nonzero = hist_work[hist_work > 0]
        if nonzero.size == 0:
            return hist, bins_x, bins_y, None

        upper_bound = float(np.percentile(nonzero, 99.9))
        hist_work[hist_work > upper_bound] = 0

        bottom_edge = int(0.5 * float(tilt) * (self.x_max + 1))
        top_edge = int(0.5 * (np.max(y_) + np.max(y_2)))
        bottom_edge = int(np.clip(bottom_edge, 0, hist.shape[1] - 2))
        top_edge = int(np.clip(top_edge, bottom_edge + 1, hist.shape[1]))

        ridge = self._fit_ridge(hist_work, bottom_edge, top_edge)
        if ridge is None:
            return hist, bins_x, bins_y, None

        mask_top = Y >= (-float(tilt) * X + 2 * ridge)
        hist_top = np.where(mask_top, hist, 0)
        hist_mirrored_top = mirror_horizontal(hist_top, ridge)

        mask_bottom = (
            Y <= -float(tilt) * X + (2 * ridge - np.max(y_))
        ) & (Y > float(tilt) * X)
        hist_bottom = np.where(mask_bottom, hist, 0)
        hist_mirrored_bottom = mirror_horizontal(hist_bottom, ridge)

        hist_filled = hist + hist_mirrored_top + hist_mirrored_bottom
        return hist_filled, bins_x, bins_y, int(ridge)

    def estimate_ridge(self, tilt: float) -> int | None:
        """Calculate the ridge for the current tilt without enabling symmetric fill."""
        x_ = self.x
        y_ = self.y
        y_2 = y_ + float(tilt) * x_

        finite = np.isfinite(x_) & np.isfinite(y_2)
        x_valid = x_[finite]
        y_valid = y_2[finite]
        if x_valid.size == 0:
            return None

        bins_x = np.arange(self.x_min, self.x_max + 2, dtype=np.float64)
        bins_y = np.arange(0, int(np.ceil(np.max(y_valid))) + 2, dtype=np.float64)
        hist, _, _ = np.histogram2d(x_valid, y_valid, bins=[bins_x, bins_y])

        nonzero = hist[hist > 0]
        if nonzero.size == 0:
            return None
        upper_bound = float(np.percentile(nonzero, 99.9))
        hist[hist > upper_bound] = 0

        bottom_edge = int(0.5 * float(tilt) * (self.x_max + 1))
        top_edge = int(0.5 * (np.max(y_) + np.max(y_2)))
        bottom_edge = int(np.clip(bottom_edge, 0, hist.shape[1] - 2))
        top_edge = int(np.clip(top_edge, bottom_edge + 1, hist.shape[1]))

        ridge = self._fit_ridge(hist, bottom_edge, top_edge)
        return int(ridge) if ridge is not None else None

    def build_base(self, settings: HistogramSettings) -> tuple[np.ndarray, np.ndarray, np.ndarray, int | None, np.ndarray | None]:
        cache_key = (
            round(float(settings.tilt), 12),
            bool(settings.symmetric_fill_enabled),

            bool(settings.median_filter_enabled),
            normalize_median_window(settings.median_filter_window)
            if bool(settings.median_filter_enabled)
            else 0,

            bool(settings.local_filter_enabled),
            normalize_median_window(settings.local_filter_window)
            if bool(settings.local_filter_enabled)
            else 0,
            int(settings.local_filter_bottom_limit)
            if bool(settings.local_filter_enabled)
            else 0,
            int(settings.local_filter_upper_limit)
            if bool(settings.local_filter_enabled)
            else 0,
        )
        if self._base_cache_key == cache_key and self._base_cache_value is not None:
            return self._base_cache_value

        if bool(settings.symmetric_fill_enabled):
            histogram, x_edges, y_edges, ridge = self.build_symmetric_histogram(settings.tilt)
        else:
            histogram, x_edges, y_edges = build_histogram_fast(
                self.x,
                self.y,
                tilt=settings.tilt,
                x_min=self.x_min,
                x_max=self.x_max,
            )
            ridge = None

        local_filter_complement = None

        if bool(settings.median_filter_enabled):
            histogram = median_filter_numpy(
                histogram,
                normalize_median_window(settings.median_filter_window),
            )
        if bool(settings.local_filter_enabled):
            bottom_limit = int(np.clip(settings.local_filter_bottom_limit, 0, 99))
            upper_limit = int(np.clip(settings.local_filter_upper_limit, bottom_limit + 1, 100))

            histogram, local_filter_complement = percentile_filter_numpy(
                histogram,
                normalize_median_window(settings.local_filter_window),
                bottom_limit,
                upper_limit,
            )

        histogram = apply_transforms(histogram, self.hooks.after_base)
        self._base_cache_key = cache_key
        self._base_cache_value = (histogram, x_edges, y_edges, ridge, local_filter_complement)
        return histogram, x_edges, y_edges, ridge, local_filter_complement

    def compute(self, settings: HistogramSettings) -> HistogramProducts:
        base, x_edges, y_edges, ridge, local_filter_complement = self.build_base(settings)

        filtered, lower_bound, upper_bound = apply_percentile_filter(
            base,
            settings.lower_percentile,
            settings.upper_percentile,
        )

        cut, bottom_cut, top_cut, top_index = apply_row_cut(
            filtered,
            settings.bottom_cut,
            settings.top_cut,
        )

        display = rebin_histogram_sum(
            apply_transforms(filtered, self.hooks.before_display),
            settings.display_bin_x,
            settings.display_bin_y,
        )

        spectrum_y = cut.sum(axis=1)
        spectrum_x, spectrum_y = rebin_projection_mean(spectrum_y, settings.spectrum_bin)

        for transform in self.hooks.before_spectrum:
            spectrum_x, spectrum_y = transform(spectrum_x, spectrum_y)

        return HistogramProducts(
            base=base,
            filtered=filtered,
            cut=cut,
            display=display,
            spectrum_x=spectrum_x,
            spectrum_y=spectrum_y,
            x_edges=x_edges,
            y_edges=y_edges,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            bottom_cut=bottom_cut,
            top_cut=top_cut,
            top_index=top_index,
            ridge=ridge,
            local_filter_complement=local_filter_complement,
        )


# =============================================================================
# Viewer state synchronization helpers
# =============================================================================


def set_slider_val_silent(
    slider: ModernSliderControl,
    value: float,
) -> None:
    old_eventson = slider.eventson
    slider.eventson = False
    slider.set_val(value)
    slider.eventson = old_eventson


def set_toggle_button_style(
    button: ModernButtonControl,
    enabled: bool,
) -> None:
    button.set_enabled_style(enabled)


def next_numbered_path(directory: Path, stem: str, suffix: str = ".npy") -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    used_indices: list[int] = []
    for path in directory.glob(f"{stem}_*{suffix}"):
        number = path.stem.removeprefix(f"{stem}_")
        if number.isdigit():
            used_indices.append(int(number))

    return directory / f"{stem}_{max(used_indices, default=-1) + 1}{suffix}"


# =============================================================================
# Tilt estimation
# =============================================================================


def estimate_tilt_from_arrays(
    x_: np.ndarray,
    y_: np.ndarray,
    *,
    m_region: np.ndarray | None = None,
    x_max_default: int = 4095,
) -> float:
    """Your original Calc Tilt logic, as a reusable function."""
    if m_region is None:
        m_region = np.linspace(0.028, 0.04, 20)

    amps: list[float] = []
    m_values_fit: list[float] = []
    if Model is None:
        raise ImportError("lmfit is required for tilt estimation. Install it with: pip install lmfit")

    model = Model(gauss)

    xmax = int(max(np.max(x_), x_max_default))
    bins_x = np.arange(xmax + 2)

    for tilt_m in m_region:
        y_2 = y_ + float(tilt_m) * x_
        bins_y = np.arange(int(np.ceil(np.max(y_2))) + 2)
        hist, _, _ = np.histogram2d(x_, y_2, bins=[bins_x, bins_y])
        upper_bound = float(np.percentile(hist[hist > 0], 99.9))
        hist[hist > upper_bound] = 0

        xc = 0.5 * (bins_x[:-1] + bins_x[1:])
        yc = 0.5 * (bins_y[:-1] + bins_y[1:])
        X, Y = np.meshgrid(xc, yc, indexing="ij")

        mask_1 = Y <= (np.max(y_2) - np.max(y_))
        hist_temp_1 = np.where(mask_1, hist, 0)
        hist_shifted_1 = np.roll(hist_temp_1, hist_temp_1.shape[1] - int(tilt_m * 4096), axis=1)

        mask_2 = (Y >= int(np.max(y_))) & (Y <= X * tilt_m + np.max(y_))
        hist_temp_2 = np.where(mask_2, hist, 0)
        hist_shifted_2 = np.roll(hist_temp_2, int(np.max(y_2) - np.max(y_)), axis=1)

        hist = hist + hist_shifted_1 + hist_shifted_2

        nonzero = hist[hist > 0]
        if nonzero.size == 0:
            continue

        upper_bound = float(np.percentile(nonzero, 99.9))
        hist[hist > upper_bound] = 0

        bottom_edge = int(0.5 * tilt_m * 4096)
        top_edge = int(0.5 * (np.max(y_) + np.max(y_2)))
        bottom_edge = int(np.clip(bottom_edge, 0, hist.shape[1] - 2))
        top_edge = int(np.clip(top_edge, bottom_edge + 1, hist.shape[1]))

        cropped_hist = hist[:, bottom_edge:top_edge]
        y_projection = np.sum(cropped_hist, axis=0)
        if y_projection.size < 3 or not np.any(y_projection > 0):
            continue

        params = model.make_params(
            amp=float(np.max(y_projection)),
            mu=cropped_hist.shape[1] / 2,
            sigma=50,
            offset=0,
        )
        params["sigma"].min = 1
        params["mu"].min = 0
        params["mu"].max = cropped_hist.shape[1]
        params["amp"].min = 0

        try:
            result = model.fit(y_projection, params, x=np.arange(cropped_hist.shape[1]))
            sigma = float(result.params["sigma"].value)
            if sigma != 0:
                amps.append(float(result.params["amp"].value) / abs(sigma))
                m_values_fit.append(float(tilt_m))
        except Exception as exc:
            print(f"{tilt_m} failed: {exc}")

    if len(m_values_fit) < 4:
        raise RuntimeError("Tilt estimation failed: not enough valid fit points.")

    x = np.array(m_values_fit)
    amps_arr = np.array(amps)

    model = Model(poly_model)
    params = model.make_params(a0=np.max(amps_arr), a1=0, a2=-100, a4=-100)
    fit_result = model.fit(amps_arr, params, x=x)

    a0 = float(fit_result.params["a0"].value)
    a1 = float(fit_result.params["a1"].value)
    a2 = float(fit_result.params["a2"].value)
    a4 = float(fit_result.params["a4"].value)

    coeffs = [4 * a4, 0, 2 * a2, a1]
    roots = np.roots(coeffs)
    real_roots = roots[np.isreal(roots)].real
    candidates = real_roots[(real_roots >= x.min()) & (real_roots <= x.max())]
    candidates = np.r_[candidates, x.min(), x.max()]

    y_candidates = poly_model(candidates, a0, a1, a2, a4)
    return float(candidates[int(np.argmax(y_candidates))])


# =============================================================================
# Main viewer
# =============================================================================

@dataclass(slots=True)
class ViewerMode:
    equal_cut_rows_enabled: bool = False
    manual_bottom_cut: int = 0
    manual_top_cut: int = 0
    zoom_to_cut_enabled: bool = False
    filter_statistics_enabled: bool = False
    reference_line_visible: bool = True


@dataclass(frozen=True, slots=True)
class SpectrumAxisCalibration:
    """Convert detector pixels to energy coordinates and back."""

    a3: float | None = None
    a2: float | None = None
    a1: float | None = None
    a0: float | None = None
    incident_energy: float | None = None
    pixel_min: float = 0.0
    pixel_max: float = 4095.0

    @classmethod
    def from_values(
        cls,
        energy_calibration: tuple[float, float, float, float] | None,
        incident_energy: float | None,
    ) -> SpectrumAxisCalibration:
        if energy_calibration is None:
            return cls(
                incident_energy=(
                    None if incident_energy is None else float(incident_energy)
                )
            )

        if len(energy_calibration) != 4:
            raise ValueError(
                "energy_calibration must contain four coefficients "
                "(a3, a2, a1, a0) as returned by "
                "compute_energy_calibration_2()."
            )

        a3, a2, a1, a0 = energy_calibration

        return cls(
            a3=float(a3),
            a2=float(a2),
            a1=float(a1),
            a0=float(a0),
            incident_energy=(
                None if incident_energy is None else float(incident_energy)
            ),
        )

    @property
    def has_energy_axis(self) -> bool:
        return all(
            coefficient is not None
            for coefficient in (self.a3, self.a2, self.a1, self.a0)
        )

    @property
    def has_loss_axis(self) -> bool:
        return self.has_energy_axis and self.incident_energy is not None

    @property
    def is_invertible(self) -> bool:
        if not self.has_energy_axis:
            return False

        pixels = np.linspace(self.pixel_min, self.pixel_max, 4096)
        derivative = (
            3.0 * self.a3 * pixels**2  # type: ignore[operator]
            + 2.0 * self.a2 * pixels  # type: ignore[operator]
            + self.a1  # type: ignore[operator]
        )

        tolerance = 1e-12
        return bool(
            np.all(derivative > tolerance)
            or np.all(derivative < -tolerance)
        )

    def pixel_to_energy(self, pixel):
        pixel = np.asarray(pixel, dtype=float)

        if not self.has_energy_axis:
            return pixel

        return (
            self.a3 * pixel**3  # type: ignore[operator]
            + self.a2 * pixel**2  # type: ignore[operator]
            + self.a1 * pixel  # type: ignore[operator]
            + self.a0  # type: ignore[operator]
        )

    def _energy_value_to_pixel(self, energy: float) -> float:
        coefficients = np.asarray(
            [
                self.a3,
                self.a2,
                self.a1,
                self.a0 - energy,  # type: ignore[operator]
            ],
            dtype=float,
        )

        nonzero = np.flatnonzero(np.abs(coefficients) > 1e-15)
        if nonzero.size == 0:
            return np.nan

        coefficients = coefficients[nonzero[0]:]

        if coefficients.size == 2:
            return float(-coefficients[1] / coefficients[0])

        roots = np.roots(coefficients)
        real_roots = roots.real[np.abs(roots.imag) < 1e-7]

        if real_roots.size == 0:
            return np.nan

        inside = real_roots[
            (real_roots >= self.pixel_min - 1e-6)
            & (real_roots <= self.pixel_max + 1e-6)
        ]

        if inside.size == 1:
            return float(inside[0])

        if inside.size > 1:
            midpoint = 0.5 * (self.pixel_min + self.pixel_max)
            return float(inside[np.argmin(np.abs(inside - midpoint))])

        distance_to_interval = np.where(
            real_roots < self.pixel_min,
            self.pixel_min - real_roots,
            real_roots - self.pixel_max,
        )
        return float(real_roots[np.argmin(distance_to_interval)])

    def energy_to_pixel(self, energy):
        energy_array = np.asarray(energy, dtype=float)

        if not self.has_energy_axis:
            return energy_array

        flat_energy = energy_array.ravel()
        flat_pixel = np.asarray(
            [self._energy_value_to_pixel(value) for value in flat_energy],
            dtype=float,
        )
        pixel = flat_pixel.reshape(energy_array.shape)

        if energy_array.ndim == 0:
            return float(pixel)

        return pixel

    def pixel_to_loss(self, pixel):
        pixel = np.asarray(pixel, dtype=float)

        if not self.has_loss_axis:
            return pixel

        return self.incident_energy - self.pixel_to_energy(pixel)  # type: ignore[operator]

    def loss_to_pixel(self, loss):
        loss = np.asarray(loss, dtype=float)

        if not self.has_loss_axis:
            return loss

        return self.energy_to_pixel(self.incident_energy - loss)  # type: ignore[operator]


def view_spectra(
    scans: np.ndarray,
    scans_dir: Path | None = None,
    cmap: str = "gnuplot",
    fig_zoom: float = 1.0,
    *,
    hist_bin: int | tuple[int, int] = 1,
    aspect="auto",
    spectra_dir: str | Path | None = None,
    histogram_dir: str | Path | None = None,
    search_roots: Iterable[Path] | None = None,
    scan_files_by_scan: dict[
        int,
        tuple[Path, Path],
    ] | None = None,
    fallback_recursive_search: bool = True,
    tilt_speedup: int = 4,
    plot_dark_style: bool = True,
    colored_background: bool = False,
    median_filter_enabled: bool = False,
    median_filter_window: int = 25,
    local_filter_enabled: bool = False,
    local_filter_window: int = 15,
    local_filter_bottom_limit: int = 0,
    local_filter_upper_limit: int = 100,
    energy_calibration: tuple[float, float, float, float] | None = None,
    incident_energy: float | None = None,
    hooks: HistogramHooks | None = None,
    window_title: str | None = None,
    tk_parent=None,
    viewer_font_sizes: dict[str, int] | None = None,
    viewer_font_factory: Callable[..., ctk.CTkFont] | None = None,
    session_state: dict | None = None,
    export_metadata: dict | None = None,

):
    """
    Interactive 2D histogram + 1D spectrum viewer.

    Refactor contract:
    - products.base: tilted / symmetric-filled raw histogram
    - products.filtered: percentile-filtered histogram used for imshow
    - products.cut: canonical current histogram used for saving and 1D spectrum
    - products.display: display-rebinned copy used only for imshow
    """
    x_min = 0
    x_max = 4095
    # The controls no longer consume Figure space, so a compact plot-oriented
    # reference size gives more balanced labels and tick text when embedded.
    figsize = (9.4 * fig_zoom, 8.2 * fig_zoom)
    viewer_fonts = build_viewer_fonts(
        viewer_font_sizes,
        viewer_font_factory,
    )

    spectra_save_dir: Path = (
        Path("spectra")
        if spectra_dir is None
        else Path(spectra_dir)
    )

    histogram_save_dir: Path = (
        Path("histograms")
        if histogram_dir is None
        else Path(histogram_dir)
    )


    def set_save_directories(
        new_spectra_dir: str | Path | None,
        new_histogram_dir: str | Path | None,
    ) -> None:
        """Update output directories used by an already open viewer."""

        nonlocal spectra_save_dir, histogram_save_dir

        spectra_save_dir = (
            Path("spectra")
            if new_spectra_dir is None
            else Path(new_spectra_dir)
        )

        histogram_save_dir = (
            Path("histograms")
            if new_histogram_dir is None
            else Path(new_histogram_dir)
        )

    scans = np.asarray(scans)
    if scans_dir is not None:
        scans_dir = Path(scans_dir)

    scans = np.asarray(
        scans,
        dtype=int,
    )

    if scans_dir is not None:
        scans_dir = Path(scans_dir)

    if scan_files_by_scan is None:
        scan_files = _find_scan_files(
            scans,
            scans_dir,
            search_roots=search_roots,
            fallback_recursive_search=fallback_recursive_search,
        )

        if len(scan_files) != scans.size:
            raise FileNotFoundError(
                "Not all requested scan files were found."
            )

        resolved_scan_files_by_scan = {
            int(scan): file_pair
            for scan, file_pair in zip(
                scans,
                scan_files,
                strict=True,
            )
        }

    else:
        missing_scan_files = [
            int(scan)
            for scan in scans
            if int(scan) not in scan_files_by_scan
        ]

        if missing_scan_files:
            raise FileNotFoundError(
                "No cached scan files available for scans: "
                f"{missing_scan_files}"
            )

        resolved_scan_files_by_scan = {
            int(scan): scan_files_by_scan[int(scan)]
            for scan in scans
        }

        scan_files = [
            resolved_scan_files_by_scan[int(scan)]
            for scan in scans
        ]
    export_metadata = dict(export_metadata or {})
    export_metadata["source_files"] = [
        {
            "x_file": str(x_file),
            "y_file": str(y_file),
        }
        for x_file, y_file in scan_files
    ]
    x_array_sum, y_array_sum = _load_scan_arrays(scan_files)
    y_array_sum -= np.min(y_array_sum)

    reference_size = (figsize[0] + figsize[1]) / 2
    if plot_dark_style:
        dark_style(figsize=(reference_size / 2, reference_size / 2))
    else:
        light_style(figsize=(reference_size / 2, reference_size / 2))

    hist_display_bin_x, hist_display_bin_y = normalize_hist_bin(hist_bin)

    pipeline = HistogramPipeline(
        x_array_sum,
        y_array_sum,
        x_min=x_min,
        x_max=x_max,
        hooks=hooks,
    )

    settings = HistogramSettings(
        tilt=0.0,
        lower_percentile=0.0,
        upper_percentile=100.0,
        bottom_cut=0,
        top_cut=0,
        spectrum_bin=1,
        display_bin_x=hist_display_bin_x,
        display_bin_y=hist_display_bin_y,
        symmetric_fill_enabled=False,
        median_filter_enabled=bool(median_filter_enabled),
        median_filter_window=normalize_median_window(median_filter_window),
        local_filter_enabled=bool(local_filter_enabled),
        local_filter_window=normalize_median_window(local_filter_window),
        local_filter_bottom_limit=local_filter_bottom_limit,
        local_filter_upper_limit=local_filter_upper_limit,
    )
    mode = ViewerMode(
        equal_cut_rows_enabled=True,
        manual_bottom_cut=0,
        manual_top_cut=0,
    )
    products = pipeline.compute(settings)

    standalone_root = None
    embedded_toolbar_frame = None
    embedded_canvas = None
    embedded_widget = None
    embedded_toolbar = None

    if tk_parent is None:
        standalone_root = ctk.CTk()
        standalone_root.geometry("1480x920")
        standalone_root.minsize(1050, 700)
        standalone_root.title(str(window_title or "meV Viewer"))
        tk_parent = standalone_root

    background_color = VIEWER_BG

    embedded_container = ctk.CTkFrame(
        tk_parent,
        fg_color=background_color,
        corner_radius=0,
    )
    setattr(embedded_container, "_viewer_fonts", viewer_fonts)
    embedded_container.pack(
        fill="both",
        expand=True,
    )

    default_sidebar_column_width = 358
    embedded_container.grid_columnconfigure(0, weight=1)
    embedded_container.grid_columnconfigure(1, weight=0, minsize=10)
    embedded_container.grid_columnconfigure(
        2,
        weight=0,
        minsize=default_sidebar_column_width,
    )
    embedded_container.grid_rowconfigure(0, weight=1)

    plot_shell = ctk.CTkFrame(
        embedded_container,
        fg_color=VIEWER_PANEL,
        border_color=VIEWER_BORDER,
        border_width=2,
        corner_radius=14,
    )
    plot_shell.grid(row=0, column=0, sticky="nsew", padx=(8, 0), pady=8)
    plot_shell.grid_columnconfigure(0, weight=1)
    plot_shell.grid_rowconfigure(0, weight=1)

    controls_sidebar = ctk.CTkScrollableFrame(
        embedded_container,
        width=350,
        fg_color=VIEWER_PANEL,
        scrollbar_button_color=VIEWER_CONTROL,
        scrollbar_button_hover_color=VIEWER_CONTROL_HOVER,
        corner_radius=14,
        border_width=2,
        border_color=VIEWER_BORDER,
    )
    setattr(controls_sidebar, "_viewer_fonts", viewer_fonts)
    controls_sidebar.grid(row=0, column=2, sticky="nsew", padx=(0, 8), pady=8)
    controls_sidebar.grid_columnconfigure(0, weight=1)

    viewer_splitter = tk.Frame(
        embedded_container,
        width=10,
        bd=0,
        highlightthickness=0,
        bg=VIEWER_BG,
        cursor="sb_h_double_arrow",
    )
    viewer_splitter.grid(row=0, column=1, sticky="ns", pady=14)
    viewer_splitter.grid_propagate(False)
    splitter_line = ctk.CTkFrame(
        viewer_splitter,
        width=2,
        corner_radius=1,
        fg_color=VIEWER_BG,
    )
    splitter_line.place(relx=0.5, rely=0.5, relheight=0.94, anchor="center")

    start_pointer_x = 0
    pointer_offset = 0.0
    start_column_width = default_sidebar_column_width
    pending_column_width = default_sidebar_column_width
    drag_moved = False
    splitter_guide = tk.Frame(
        embedded_container,
        width=3,
        bd=0,
        highlightthickness=0,
        bg=VIEWER_ACCENT,
    )

    def center_for_pointer(pointer_x: int) -> float:
        local_pointer_x = pointer_x - embedded_container.winfo_rootx()
        available_width = max(1, embedded_container.winfo_width())
        splitter_width = max(1, viewer_splitter.winfo_width())
        half_splitter = splitter_width / 2.0
        minimum_center = 440 + half_splitter
        maximum_center = max(
            minimum_center,
            available_width - 300 - half_splitter,
        )
        return float(np.clip(
            local_pointer_x - pointer_offset,
            minimum_center,
            maximum_center,
        ))

    def begin_viewer_split_drag(event) -> None:
        nonlocal start_pointer_x, pointer_offset
        nonlocal start_column_width, pending_column_width, drag_moved

        embedded_container.update_idletasks()
        start_pointer_x = int(event.x_root)
        start_column_width = max(
            1,
            embedded_container.winfo_width()
            - viewer_splitter.winfo_x()
            - viewer_splitter.winfo_width(),
        )
        pending_column_width = start_column_width
        splitter_center = (
            viewer_splitter.winfo_x()
            + viewer_splitter.winfo_width() / 2.0
        )
        pointer_offset = (
            start_pointer_x
            - embedded_container.winfo_rootx()
            - splitter_center
        )
        drag_moved = False

        # The grid column owns the physical width. Keeping the CTk request at
        # one logical pixel prevents DPI scaling from changing it a second time.
        embedded_container.grid_columnconfigure(
            2,
            minsize=start_column_width,
        )
        controls_sidebar.configure(width=1)

    def drag_viewer_split(event) -> None:
        nonlocal pending_column_width, drag_moved

        pointer_x = int(event.x_root)
        if not drag_moved and abs(pointer_x - start_pointer_x) < 3:
            return

        drag_moved = True
        splitter_center = center_for_pointer(pointer_x)
        pending_column_width = int(round(
            embedded_container.winfo_width()
            - splitter_center
            - viewer_splitter.winfo_width() / 2.0
        ))
        splitter_guide.place(
            x=int(round(splitter_center)),
            rely=0.5,
            relheight=0.94,
            anchor="center",
        )
        splitter_guide.lift()

    def cover_viewer_layout() -> tk.Frame:
        # Hide the intermediate reflow until Tk and Matplotlib have settled.
        cover = tk.Frame(
            embedded_container,
            bg=VIEWER_BG,
            bd=0,
            highlightthickness=0,
        )
        cover.place(relx=0, rely=0, relwidth=1, relheight=1)
        cover.lift()
        return cover

    def viewer_layout_cover_exists(cover: tk.Frame) -> bool:
        try:
            return bool(cover.winfo_exists())
        except tk.TclError:
            return False

    def reveal_viewer_layout_after_idle(cover: tk.Frame) -> None:
        def reveal() -> None:
            try:
                if cover.winfo_exists():
                    cover.destroy()
            except tk.TclError:
                return

        embedded_container.after(70, reveal)

    def end_viewer_split_drag(_event=None) -> None:
        splitter_guide.place_forget()
        if not drag_moved or pending_column_width == start_column_width:
            return

        cover = cover_viewer_layout()

        def apply_width() -> None:
            if not viewer_layout_cover_exists(cover):
                return
            embedded_container.grid_columnconfigure(
                2,
                minsize=pending_column_width,
            )
            if embedded_canvas is not None:
                embedded_canvas.draw_idle()
            reveal_viewer_layout_after_idle(cover)

        # Let the button-release callback finish before geometry events fire.
        embedded_container.after(16, apply_width)

    for splitter_widget in (viewer_splitter, splitter_line):
        splitter_widget.bind("<ButtonPress-1>", begin_viewer_split_drag)
        splitter_widget.bind("<B1-Motion>", drag_viewer_split)
        splitter_widget.bind("<ButtonRelease-1>", end_viewer_split_drag)

    ctk.CTkLabel(
        controls_sidebar,
        text="SPECTRA VIEWER",
        anchor="w",
        text_color=VIEWER_TEXT,
        font=viewer_fonts.title,
    ).grid(row=0, column=0, sticky="ew", padx=3, pady=(1, 0))

    viewer_subtitle_label = ctk.CTkLabel(
        controls_sidebar,
        text=str(window_title or "Interactive detector analysis"),
        anchor="w",
        justify="left",
        text_color=VIEWER_MUTED_TEXT,
        font=viewer_fonts.subtitle,
    )
    viewer_subtitle_label.grid(
        row=1,
        column=0,
        sticky="ew",
        padx=3,
        pady=(0, 10),
    )
    bind_responsive_label_wrap(
        viewer_subtitle_label,
        controls_sidebar,
        horizontal_padding=24,
    )

    def set_viewer_title(title: str) -> None:
        resolved_title = str(title or "Interactive detector analysis")
        viewer_subtitle_label.configure(text=resolved_title)
        refresh_wrap = getattr(
            viewer_subtitle_label,
            "_responsive_wrap_update",
            None,
        )
        if callable(refresh_wrap):
            refresh_wrap()
        if standalone_root is not None:
            standalone_root.title(resolved_title)

    display_section = make_viewer_card(
        controls_sidebar,
        "Display",
        "Color mapping, intensity and display rebinning",
        initially_expanded=False,
    )
    display_section.container.grid(row=2, column=0, sticky="ew", pady=(0, 9))
    display_controls = display_section.body

    alignment_section = make_viewer_card(
        controls_sidebar,
        "Alignment & ROI",
        "Tilt, reference line and detector region",
    )
    alignment_section.container.grid(row=3, column=0, sticky="ew", pady=(0, 9))
    alignment_controls = alignment_section.body

    filter_section = make_viewer_card(
        controls_sidebar,
        "Filters",
        "Percentile, median and local cleanup",
    )
    filter_section.container.grid(row=5, column=0, sticky="ew", pady=(0, 9))
    filter_controls = filter_section.body

    spectrum_section = make_viewer_card(
        controls_sidebar,
        "1D Spectrum",
        "Binning, axis mode and vertical scale",
    )
    spectrum_section.container.grid(row=4, column=0, sticky="ew", pady=(0, 9))
    spectrum_controls = spectrum_section.body

    action_section = make_viewer_card(controls_sidebar, "Output")
    action_section.container.grid(row=6, column=0, sticky="ew", pady=(0, 2))
    action_controls = action_section.body

    fig_ = Figure(figsize=figsize)
    ax = fig_.subplots(2, 1, sharex=True)

    figure_background = VIEWER_PANEL if plot_dark_style else "#F5F7FA"
    fig_.patch.set_facecolor(figure_background)

    for axis in np.atleast_1d(ax):
        axis.set_facecolor("none")

    embedded_canvas = FigureCanvasTkAgg(
        fig_,
        master=plot_shell,
    )

    embedded_widget = embedded_canvas.get_tk_widget()
    embedded_widget.configure(
        background=background_color,
        highlightthickness=0,
        borderwidth=0,
    )
    embedded_widget.grid(
        row=0,
        column=0,
        sticky="nsew",
        padx=7,
        pady=(7, 0),
    )

    embedded_toolbar_frame = tk.Frame(
        plot_shell,
        background=background_color,
        highlightthickness=0,
        borderwidth=0,
    )
    # The native toolbar remains hidden as the Matplotlib navigation backend.
    # All visible controls are CustomTkinter widgets below.
    embedded_toolbar = NavigationToolbar2Tk(
        embedded_canvas,
        embedded_toolbar_frame,
        pack_toolbar=False,
    )
    embedded_toolbar.update()

    modern_toolbar = ctk.CTkFrame(
        plot_shell,
        fg_color="transparent",
        corner_radius=0,
    )
    modern_toolbar.grid(row=1, column=0, sticky="ew", padx=12, pady=(4, 6))
    modern_toolbar.grid_columnconfigure(6, weight=1)

    toolbar_buttons: dict[str, ctk.CTkButton] = {}

    def refresh_axes_after_show() -> None:
        """Redraw the Viewer after it becomes visible."""

        embedded_widget.update_idletasks()

        update_spectrum_axis_label()
        fig_.canvas.draw_idle()

    def update_toolbar_modes() -> None:
        mode_name = str(getattr(embedded_toolbar, "mode", "")).lower()
        for name in ("pan", "zoom"):
            active = name in mode_name
            toolbar_buttons[name].configure(
                fg_color="#174C36" if active else VIEWER_CONTROL,
                border_color=VIEWER_SUCCESS if active else VIEWER_BORDER,
            )

    def run_toolbar_action(action: str) -> None:
        getattr(embedded_toolbar, action)()
        update_toolbar_modes()

    for toolbar_column, (action, text) in enumerate(
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
            text=text,
            width=40 if action != "save_figure" else 88,
            height=31,
            corner_radius=8,
            fg_color=VIEWER_CONTROL,
            hover_color=VIEWER_CONTROL_HOVER,
            border_width=1,
            border_color=VIEWER_BORDER,
            font=viewer_fonts.toolbar,
            command=lambda selected_action=action: run_toolbar_action(selected_action),
        )
        toolbar_button.grid(row=0, column=toolbar_column, padx=(0, 5))
        toolbar_buttons[action] = toolbar_button

    cursor_position_label = ModernStatusLabel(
        modern_toolbar,
        text="",
        width=330,
        anchor="e",
        text_color=VIEWER_MUTED_TEXT,
        fg_color="transparent",
        font=viewer_fonts.coordinates,
    )
    cursor_position_label.grid(
        row=0,
        column=6,
        sticky="ew",
        padx=(8, 2),
    )

    cursor_toolbar_below_tools: bool | None = None

    def refresh_cursor_toolbar_layout(event=None) -> None:
        nonlocal cursor_toolbar_below_tools
        # Move the coordinates below the tools before either area gets clipped.
        toolbar_width = int(
            getattr(event, "width", 0) or modern_toolbar.winfo_width()
        )
        reverse_scaling = cast(
            Callable[[float], float] | None,
            getattr(modern_toolbar, "_reverse_widget_scaling", None),
        )
        if callable(reverse_scaling):
            toolbar_width = int(reverse_scaling(toolbar_width))

        below_tools = toolbar_width < 680
        if cursor_toolbar_below_tools == below_tools:
            return
        cursor_toolbar_below_tools = below_tools

        if below_tools:
            cursor_position_label.grid_configure(
                row=1,
                column=0,
                columnspan=7,
                sticky="ew",
                padx=(2, 2),
                pady=(5, 0),
            )
        else:
            cursor_position_label.grid_configure(
                row=0,
                column=6,
                columnspan=1,
                sticky="ew",
                padx=(8, 2),
                pady=0,
            )

    modern_toolbar.bind("<Configure>", refresh_cursor_toolbar_layout, add=True)
    modern_toolbar.after_idle(refresh_cursor_toolbar_layout)

    embedded_canvas.draw()

    # -------------------------------------------------------------------------
    # 1D spectrum x-axis handling
    #
    # The real x-data always stays in pixel coordinates.
    # This preserves sharex=True between the 2D histogram and the 1D spectrum.
    #
    # Display modes:
    #   pixel  -> x = pixel
    #   energy -> x = E(pixel)
    #   loss   -> x = E_in - E(pixel)
    # -------------------------------------------------------------------------
    spectrum_axis_mode = "pixel"
    axis_calibration = SpectrumAxisCalibration.from_values(
        energy_calibration,
        incident_energy,
    )
    incident_energy = axis_calibration.incident_energy
    pixel_to_energy = axis_calibration.pixel_to_energy
    energy_to_pixel = axis_calibration.energy_to_pixel
    pixel_to_loss = axis_calibration.pixel_to_loss
    loss_to_pixel = axis_calibration.loss_to_pixel

    energy_axis = None
    loss_axis = None

    if axis_calibration.has_energy_axis:
        energy_axis = ax[1].secondary_xaxis(
            "bottom",
            functions=(pixel_to_energy, energy_to_pixel),
        )
        energy_axis.set_visible(False)
        energy_axis.set_xlabel("Energy")

        energy_axis.tick_params(axis="x", colors=rcParams["xtick.color"])
        energy_axis.xaxis.label.set_color(rcParams["axes.labelcolor"])

    if axis_calibration.has_loss_axis:
        loss_axis = ax[1].secondary_xaxis(
            "bottom",
            functions=(pixel_to_loss, loss_to_pixel),
        )
        loss_axis.set_visible(False)
        loss_axis.set_xlabel("Energy loss")

        loss_axis.tick_params(axis="x", colors=rcParams["xtick.color"])
        loss_axis.xaxis.label.set_color(rcParams["axes.labelcolor"])


    def current_spectrum_x_label() -> str:
        """
        Return a label for the current 1D x-axis mode.
        """

        if spectrum_axis_mode == "energy":
            return "Energy"

        if spectrum_axis_mode == "loss":
            return "Energy loss"

        return "Pixel"


    def update_spectrum_axis_label() -> None:
        """
        Update visible x-axis labels without changing the real pixel x-axis.
        """

        if energy_axis is not None:
            energy_axis.set_visible(False)

        if loss_axis is not None:
            loss_axis.set_visible(False)

        if spectrum_axis_mode == "energy" and energy_axis is not None:
            ax[1].set_xlabel("")
            ax[1].tick_params(axis="x", labelbottom=False)
            energy_axis.set_visible(True)
            energy_axis.set_xlabel("Energy")
            return

        if spectrum_axis_mode == "loss" and loss_axis is not None:
            ax[1].set_xlabel("")
            ax[1].tick_params(axis="x", labelbottom=False)
            loss_axis.set_visible(True)
            loss_axis.set_xlabel("Energy loss")
            return

        ax[1].tick_params(axis="x", labelbottom=True)
        ax[1].set_xlabel("Pixel")

    last_cursor_data: tuple[int, float, float] | None = None

    def format_cursor_number(value: float) -> str:
        absolute_value = abs(value)
        if absolute_value >= 100_000 or (
            absolute_value > 0 and absolute_value < 0.001
        ):
            return f"{value:.2e}"
        return f"{value:.4g}"

    def histogram_count_at(pixel_x: float, pixel_y: float) -> float | None:
        """Return the value of the displayed histogram bin under the cursor."""

        histogram = products.display
        if histogram.size == 0:
            return None

        x_min = float(products.x_edges[0])
        x_max = float(products.x_edges[-1])
        y_min = float(products.y_edges[0])
        y_max = float(products.y_edges[-1])

        if not (x_min <= pixel_x <= x_max and y_min <= pixel_y <= y_max):
            return None
        if x_max <= x_min or y_max <= y_min:
            return None

        # imshow stretches the rebinned array over the full histogram extent.
        # Map the plot coordinates back to that exact displayed array.
        x_index = int((pixel_x - x_min) / (x_max - x_min) * histogram.shape[0])
        y_index = int((pixel_y - y_min) / (y_max - y_min) * histogram.shape[1])
        x_index = min(x_index, histogram.shape[0] - 1)
        y_index = min(y_index, histogram.shape[1] - 1)

        return float(histogram[x_index, y_index])

    def refresh_cursor_position() -> None:
        if last_cursor_data is None:
            cursor_position_label.set_text("")
            return

        plot_index, pixel_x, y_value = last_cursor_data

        if (
            spectrum_axis_mode == "energy"
            and axis_calibration.has_energy_axis
        ):
            displayed_x = float(pixel_to_energy(pixel_x))
            x_text = f"E =  {displayed_x:.3f}"
        elif (
            spectrum_axis_mode == "loss"
            and axis_calibration.has_loss_axis
        ):
            displayed_x = float(pixel_to_loss(pixel_x))
            x_text = f"E = {displayed_x:.3f}"
        else:
            x_text = f"x = {pixel_x:.0f}"

        if plot_index == 0:
            y_text = f"y = {y_value:.0f}"
            count = histogram_count_at(pixel_x, y_value)
            count_text = (
                f"counts = {format_cursor_number(count)}"
                if count is not None
                else "counts = —"
            )
        else:
            y_text = f"y = {format_cursor_number(y_value)}"
            count_text = ""

        cursor_position_label.set_text(
            f"{x_text}  ·  {y_text}"
            + (f"  ·  {count_text}" if count_text else "")
        )

    def update_cursor_position(event: Event) -> None:
        nonlocal last_cursor_data

        if not isinstance(event, MouseEvent):
            last_cursor_data = None
            refresh_cursor_position()
            return

        if event.xdata is None or event.ydata is None:
            last_cursor_data = None
        elif event.inaxes is ax[0]:
            last_cursor_data = (0, float(event.xdata), float(event.ydata))
        elif event.inaxes is ax[1]:
            last_cursor_data = (1, float(event.xdata), float(event.ydata))
        else:
            last_cursor_data = None

        refresh_cursor_position()

    def clear_cursor_position(_event: Event | None = None) -> None:
        nonlocal last_cursor_data
        last_cursor_data = None
        refresh_cursor_position()

    # -------------------------------------------------------------------------
    # Plot area
    # -------------------------------------------------------------------------


    # The Figure now contains plots only; controls live in the responsive
    # CustomTkinter sidebar. This lets both subplots use the available canvas.
    fig_.subplots_adjust(
        top=0.975,
        right=0.975,
        bottom=0.105,
        left=0.105,
        hspace=0.18,
    )

    def nonzero_percentile(histogram: np.ndarray, percentile: float, default: float) -> float:
        nonzero = histogram[histogram > 0]
        if nonzero.size == 0:
            return default
        return float(np.percentile(nonzero, percentile))

    vmax_default = int(nonzero_percentile(products.display, 90, 1.0))
    vmax_slider_max = int(nonzero_percentile(products.display, 99.995, 2.0))
    vmax_slider_max = max(vmax_slider_max, 2)
    vmax_default = max(min(vmax_default, vmax_slider_max), 1)
    vmin_default = 0 # max(hist_display_bin_x, hist_display_bin_y, 0)

    if cmap not in plt.colormaps():
        cmap = "gist_stern"

    im = ax[0].imshow(
        products.display.T,
        origin="lower",
        aspect=aspect,
        cmap=cmap,
        vmin=vmin_default,
        vmax=max(vmax_default, 1.0),
        extent=[
            products.x_edges[0],
            products.x_edges[-1],
            products.y_edges[0],
            products.y_edges[-1],
        ],
    )
    if not plot_dark_style:
        ax[0].grid(color="white")

    cmap_options = ["gnuplot", "gnuplot2", "viridis", "inferno", "hot", "bone", "gist_stern"]
    if cmap not in cmap_options and cmap in plt.colormaps():
        cmap_options.append(cmap)

    ctk.CTkLabel(
        display_controls,
        text="Colormap",
        anchor="w",
        text_color=VIEWER_TEXT,
        font=viewer_fonts.control_label,
    ).grid(row=0, column=0, sticky="w")

    cmap_control_row = ctk.CTkFrame(
        display_controls,
        fg_color="transparent",
        corner_radius=0,
    )
    cmap_control_row.grid(row=1, column=0, sticky="ew", pady=(3, 8))
    cmap_control_row.grid_columnconfigure(0, weight=1)

    cmap_radio = ModernChoiceControl(
        cmap_control_row,
        cmap_options,
        cmap,
        width=210,
    )
    cmap_radio.grid(row=0, column=0, sticky="w")

    add_colormap_button_widget = ModernButtonControl(
        cmap_control_row,
        "+",
        width=40,
    )
    add_colormap_button_widget.grid(row=0, column=1, sticky="e", padx=(7, 0))
    colormap_dialog: ctk.CTkToplevel | None = None

    color_1d = "cyan" if plot_dark_style else "black"
    projection_lines = ax[1].plot(
        products.spectrum_x,
        products.spectrum_y,
        color=color_1d,
        linewidth=2.5,
    )
    projection_line = projection_lines[0]

    ax[0].set_xlim(products.x_edges[0], products.x_edges[-1])
    update_spectrum_axis_label()

    def get_spectrum_y_limits() -> tuple[float, float]:
        """
        Return automatic y-limits for the current 1D spectrum.
        """

        if products.spectrum_y.size == 0:
            return 0.0, 1.0

        finite_y = products.spectrum_y[np.isfinite(products.spectrum_y)]

        if finite_y.size == 0:
            return 0.0, 1.0

        ymax = float(np.max(finite_y))

        return 0.0, max(ymax * 1.05, 1.0)

    ax[1].set_ylim(*get_spectrum_y_limits())


    def y_from_row_index(row_index: int) -> float:
        row_index = int(np.clip(row_index, 0, len(products.y_edges) - 1))
        return float(products.y_edges[row_index])

    def row_index_from_y(y_value: float) -> int:
        max_row_index = max(int(products.base.shape[1] - 1), 0)
        return int(np.clip(np.argmin(np.abs(products.y_edges - float(y_value))), 0, max_row_index))

    cut_bottom_line = ax[0].axhline(
        y_from_row_index(products.bottom_cut),
        linestyle="-",
        lw=1.5,
        color="orange",
    )
    cut_top_line = ax[0].axhline(
        y_from_row_index(products.top_index),
        linestyle="-",
        lw=1.5,
        color="orange",
    )

    y_line_min = float(products.y_edges[0])
    y_line_max = float(products.y_edges[-1])
    y_line_default = 0.5 * (y_line_min + y_line_max)
    line = ax[0].axhline(y_line_default,
                        linestyle=(0, (0.1, 2.5)),
                        dash_capstyle="round",
                        lw=3, color="fuchsia")

    # -------------------------------------------------------------------------
    # Compact CustomTkinter controls
    # -------------------------------------------------------------------------

    vmin_slider = ModernSliderControl(
        display_controls,
        "Display Min",
        0.0,
        max(float(vmax_slider_max - 1), 0.0),
        valinit=vmin_default,
        valstep=1.0,
        slider_width=145,
    )
    vmin_slider.grid(row=2, column=0, sticky="ew", pady=(1, 5))

    vmax_slider = ModernSliderControl(
        display_controls,
        "Display Max",
        1.0,
        vmax_slider_max,
        valinit=max(vmax_default, 1.0),
        valstep=1.0,
        slider_width=145,
    )
    vmax_slider.grid(row=3, column=0, sticky="ew", pady=5)

    cmap_range_section = make_viewer_subsection(
        display_controls,
        "Colormap Range",
        initially_expanded=False
    )
    cmap_range_section.container.grid(
        row=4,
        column=0,
        sticky="ew",
        pady=(7, 5),
    )
    cmap_range_controls = cmap_range_section.body
    cmap_range_values = [
        float(value) for value in np.linspace(0.0, 1.0, 100)
    ]

    cmap_start_slider = ModernSliderControl(
        cmap_range_controls,
        "Start",
        0.0,
        1.0,
        valinit=0.0,
        valstep=cmap_range_values,
        slider_width=124,
        value_formatter=lambda value: f"{value:.2f}",
    )
    cmap_start_slider.grid(row=0, column=0, sticky="ew", pady=(0, 5))

    cmap_end_slider = ModernSliderControl(
        cmap_range_controls,
        "End",
        0.0,
        1.0,
        valinit=1.0,
        valstep=cmap_range_values,
        slider_width=124,
        value_formatter=lambda value: f"{value:.2f}",
    )
    cmap_end_slider.grid(row=1, column=0, sticky="ew")

    allowed_bin_vals = [1, 2, 4, 8, 16, 32, 64]
    hist_bin_slider_x = ModernSliderControl(
        display_controls,
        "2D bin size X",
        1,
        64,
        valinit=settings.display_bin_x,
        valstep=allowed_bin_vals,
        slider_width=126,
    )
    hist_bin_slider_x.grid(row=5, column=0, sticky="ew", pady=5)

    hist_bin_slider_y = ModernSliderControl(
        display_controls,
        "2D bin size Y",
        1,
        64,
        valinit=settings.display_bin_y,
        valstep=allowed_bin_vals,
        slider_width=126,
    )
    hist_bin_slider_y.grid(row=6, column=0, sticky="ew", pady=(5, 0))

    m_slider = ModernSliderControl(
        alignment_controls,
        "Tilt",
        0.0,
        0.045,
        valinit=settings.tilt,
        valstep=0.0001,
        slider_width=124,
    )
    m_slider.grid(row=0, column=0, sticky="ew", pady=(0, 5))

    tilt_action_row = ctk.CTkFrame(
        alignment_controls,
        fg_color="transparent",
        corner_radius=0,
    )
    tilt_action_row.grid(row=1, column=0, sticky="ew", pady=(3, 6))
    for column in range(2):
        tilt_action_row.grid_columnconfigure(column, weight=1)

    calculate_m_button_widget = ModernButtonControl(
        tilt_action_row,
        "Calc Tilt",
        width=128,
    )
    calculate_m_button_widget.grid(row=0, column=0, sticky="ew", padx=(0, 3))

    calculate_ridge_button_widget = ModernButtonControl(
        tilt_action_row,
        "Find Ridge",
        width=128,
    )
    calculate_ridge_button_widget.grid(row=0, column=1, sticky="ew", padx=(3, 0))

    tilt_speedup_row = ctk.CTkFrame(
        alignment_controls,
        fg_color="transparent",
        corner_radius=0,
    )
    tilt_speedup_row.grid(row=2, column=0, sticky="ew", pady=(1, 6))
    tilt_speedup_row.grid_columnconfigure(0, weight=1)

    ctk.CTkLabel(
        tilt_speedup_row,
        text="Tilt speedup",
        anchor="w",
        text_color=VIEWER_TEXT,
        font=viewer_fonts.control_label,
    ).grid(row=0, column=0, sticky="w")

    tilt_speedup_values = [str(value) for value in range(1, 11)]
    initial_tilt_speedup = str(int(np.clip(int(tilt_speedup), 1, 10)))
    tilt_speedup_control = ModernChoiceControl(
        tilt_speedup_row,
        tilt_speedup_values,
        initial_tilt_speedup,
        width=116,
    )
    tilt_speedup_control.grid(row=0, column=1, sticky="e")

    line_slider = ModernSliderControl(
        alignment_controls,
        "Reference line height",
        y_line_min,
        y_line_max,
        valinit=y_line_default,
        valstep=1.0,
        slider_width=142,
    )
    line_slider.grid(row=3, column=0, sticky="ew", pady=5)

    hide_ref_line_var = tk.BooleanVar(value=not mode.reference_line_visible)
    hide_ref_line_checkbox_widget = ctk.CTkCheckBox(
        alignment_controls,
        text="Hide Reference Line",
        variable=hide_ref_line_var,
        onvalue=True,
        offvalue=False,
        width=128,
        checkbox_width=19,
        checkbox_height=19,
        corner_radius=5,
        border_width=2,
        border_color=VIEWER_BORDER,
        fg_color=VIEWER_ACCENT,
        hover_color=VIEWER_ACCENT_HOVER,
        text_color=VIEWER_TEXT,
        font=viewer_fonts.control_label,
    )
    hide_ref_line_checkbox_widget.grid(
        row=4,
        column=0,
        sticky="w",
        pady=(2, 6),
    )

    row_count = int(products.base.shape[1])
    max_row = row_count - 1
    bottom_slider = ModernSliderControl(
        alignment_controls,
        "ROI bottom",
        0,
        max_row,
        valinit=0,
        valstep=1,
        slider_width=142,
    )
    bottom_slider.grid(row=6, column=0, sticky="ew", pady=5)

    top_slider = ModernSliderControl(
        alignment_controls,
        "ROI top",
        1,
        row_count,
        valinit=products.top_index,
        valstep=1,
        slider_width=142,
    )
    top_slider.grid(row=5, column=0, sticky="ew", pady=5)

    initial_reference_row = row_index_from_y(y_line_default)
    initial_symmetric_margin = max(
        min(
            initial_reference_row,
            int(products.base.shape[1] - initial_reference_row - 1),
        ),
        0,
    )

    c_slider = ModernSliderControl(
        alignment_controls,
        "Symmetric cut margin",
        0,
        max_row,
        valinit=initial_symmetric_margin,
        valstep=1,
        slider_width=142,
    )
    c_slider.grid(row=7, column=0, sticky="ew", pady=5)
    c_ax = c_slider.ax

    roi_button_row = ctk.CTkFrame(
        alignment_controls,
        fg_color="transparent",
        corner_radius=0,
    )
    roi_button_row.grid(row=8, column=0, sticky="ew", pady=(5, 0))
    for column in range(2):
        roi_button_row.grid_columnconfigure(column, weight=1)

    equal_cut_rows_button_widget = ModernButtonControl(
        roi_button_row,
        "Symm. ROI: ON",
        width=128,
    )
    equal_cut_rows_button_widget.grid(row=0, column=0, sticky="ew", padx=(0, 3))

    zoom_to_cut_button_widget = ModernButtonControl(
        roi_button_row,
        "Focus ROI",
        width=128,
    )
    zoom_to_cut_button_widget.grid(row=0, column=1, sticky="ew", padx=(3, 0))

    symmetric_fill_section = make_viewer_subsection(
        alignment_controls,
        "Symmetric Fill",
        initially_expanded=False
    )
    symmetric_fill_section.container.grid(
        row=9,
        column=0,
        sticky="ew",
        pady=(12, 0),
    )
    symmetric_fill_controls = symmetric_fill_section.body

    fill_symmetric_button_widget = ModernButtonControl(
        symmetric_fill_controls,
        "Symmetric Fill: OFF",
        width=260,
    )
    fill_symmetric_button_widget.grid(
        row=0,
        column=0,
        sticky="ew",
    )

    global_filter_section = make_viewer_subsection(
        filter_controls,
        "Global",
    )
    global_filter_section.container.grid(
        row=0,
        column=0,
        sticky="ew",
        pady=(0, 7),
    )
    global_filter_controls = global_filter_section.body

    local_filter_section = make_viewer_subsection(
        filter_controls,
        "Local",
    )
    local_filter_section.container.grid(
        row=1,
        column=0,
        sticky="ew",
        pady=7,
    )
    local_filter_controls = local_filter_section.body

    median_filter_section = make_viewer_subsection(
        filter_controls,
        "Median",
        initially_expanded=False,
    )
    median_filter_section.container.grid(
        row=2,
        column=0,
        sticky="ew",
        pady=(7, 0),
    )
    median_filter_controls = median_filter_section.body

    lower_slider = ModernSliderControl(
        global_filter_controls,
        "Minimum percentile",
        0.0,
        99.0,
        valinit=settings.lower_percentile,
        valstep=1,
        slider_width=140,
    )
    lower_slider.grid(row=0, column=0, sticky="ew", pady=(0, 5))

    upper_slider = ModernSliderControl(
        global_filter_controls,
        "Maximum percentile",
        99.0,
        100.0,
        valinit=settings.upper_percentile,
        valstep=0.001,
        slider_width=140,
    )
    upper_slider.grid(row=1, column=0, sticky="ew", pady=5)

    local_filter_button_widget = ModernButtonControl(
        local_filter_controls,
        "Local: ON" if settings.local_filter_enabled else "Local: OFF",
        width=260,
    )
    local_filter_button_widget.grid(row=0, column=0, sticky="ew", pady=(0, 5))

    median_filter_button_widget = ModernButtonControl(
        median_filter_controls,
        "Median: ON" if settings.median_filter_enabled else "Median: OFF",
        width=260,
    )
    median_filter_button_widget.grid(row=0, column=0, sticky="ew", pady=(0, 5))

    allowed_median_windows = list(range(3, 50, 2))
    median_window_slider = ModernSliderControl(
        median_filter_controls,
        "Median window",
        3,
        49,
        valinit=settings.median_filter_window,
        valstep=allowed_median_windows,
        slider_width=124,
    )
    median_window_slider.grid(row=1, column=0, sticky="ew", pady=(5, 0))

    local_window_values = list(range(3, 50, 2))
    local_min_perc_values = list(range(0, 101))
    local_max_perc_values = list(range(0, 101))
    local_controls_are_syncing = False

    def set_step_control_value(control, values, value) -> None:
        value = int(value)
        value = min(values, key=lambda allowed: abs(int(allowed) - value))
        control["set_index"](values.index(value))

    def update_local_filter_controls(_value=None) -> None:
        nonlocal local_controls_are_syncing

        if local_controls_are_syncing:
            return

        local_controls_are_syncing = True
        try:
            window = normalize_median_window(
                int(local_window_control["get_value"]())
            )
            bottom_limit = int(local_min_perc_control["get_value"]())
            upper_limit = int(local_max_perc_control["get_value"]())

            bottom_limit = int(np.clip(bottom_limit, 0, 99))
            upper_limit = int(np.clip(upper_limit, bottom_limit + 1, 100))

            if int(local_window_control["get_value"]()) != window:
                set_step_control_value(local_window_control, local_window_values, window)
            if int(local_min_perc_control["get_value"]()) != bottom_limit:
                set_step_control_value(
                    local_min_perc_control,
                    local_min_perc_values,
                    bottom_limit,
                )
            if int(local_max_perc_control["get_value"]()) != upper_limit:
                set_step_control_value(
                    local_max_perc_control,
                    local_max_perc_values,
                    upper_limit,
                )

            settings.local_filter_window = window
            settings.local_filter_bottom_limit = bottom_limit
            settings.local_filter_upper_limit = upper_limit
        finally:
            local_controls_are_syncing = False

        pipeline.clear_cache()
        if settings.local_filter_enabled:
            recompute_all(draw=True, reset_xlim=False)

    def make_local_control(
        row: int,
        label: str,
        values: list[int],
        initial: int,
        *,
        slider_width: int,
    ) -> ModernStepControl:
        control_widget = ModernSliderControl(
            local_filter_controls,
            label,
            min(values),
            max(values),
            valinit=initial,
            valstep=values,
            slider_width=slider_width,
        )
        control_widget.grid(row=row, column=0, sticky="ew", pady=5)
        callback_id = control_widget.on_changed(update_local_filter_controls)

        def set_index(index: int) -> None:
            index = int(np.clip(index, 0, len(values) - 1))
            control_widget.set_val(values[index])

        return {
            "widget": control_widget,
            "values": values,
            "get_value": lambda: control_widget.val,
            "set_index": set_index,
            "callback_ids": (callback_id,),
        }

    local_window_control = make_local_control(
        1,
        "Local window",
        local_window_values,
        settings.local_filter_window,
        slider_width=124,
    )
    local_min_perc_control = make_local_control(
        2,
        "Local minimum %",
        local_min_perc_values,
        settings.local_filter_bottom_limit,
        slider_width=138,
    )
    local_max_perc_control = make_local_control(
        3,
        "Local maximum %",
        local_max_perc_values,
        settings.local_filter_upper_limit,
        slider_width=138,
    )

    filter_statistics_button_widget = ModernButtonControl(
        local_filter_controls,
        "Show filter statistics",
        width=260,
    )
    filter_statistics_button_widget.grid(row=4, column=0, sticky="ew", pady=(6, 0))

    allowed_1D_bin_vals = [
        int(element)
        for element in np.arange(1, 65, dtype=np.int16)
        if element == 1 or element % 2 == 0
    ]
    bin_slider = ModernSliderControl(
        spectrum_controls,
        "1D bin",
        1,
        64,
        valinit=settings.spectrum_bin,
        valstep=allowed_1D_bin_vals,
        slider_width=124,
    )
    bin_slider.grid(row=0, column=0, sticky="ew", pady=(0, 6))

    spectrum_button_row = ctk.CTkFrame(
        spectrum_controls,
        fg_color="transparent",
        corner_radius=0,
    )
    spectrum_button_row.grid(row=1, column=0, sticky="ew")
    for column in range(3):
        spectrum_button_row.grid_columnconfigure(column, weight=1)

    reset_spectrum_y_button_widget = ModernButtonControl(
        spectrum_button_row,
        "Autoscale Y",
        width=80,
    )
    reset_spectrum_y_button_widget.grid(row=0, column=0, sticky="ew", padx=(0, 3))

    energy_axis_button_widget = None
    loss_axis_button_widget = None

    if energy_calibration is not None:
        energy_axis_button_widget = ModernButtonControl(
            spectrum_button_row,
            "Energy",
            width=80,
        )
        energy_axis_button_widget.grid(row=0, column=1, sticky="ew", padx=3)

    if energy_calibration is not None and incident_energy is not None:
        loss_axis_button_widget = ModernButtonControl(
            spectrum_button_row,
            "Loss",
            width=80,
        )
        loss_axis_button_widget.grid(row=0, column=2, sticky="ew", padx=(3, 0))

    action_button_row = ctk.CTkFrame(
        action_controls,
        fg_color="transparent",
        corner_radius=0,
    )
    action_button_row.grid(row=0, column=0, sticky="ew")
    action_button_row.grid_columnconfigure(0, weight=1)

    save_spectrum_button_widget = ModernButtonControl(
        action_button_row,
        "Save spectrum",
        width=126,
    )
    save_spectrum_button_widget.grid(row=0, column=0, sticky="ew")

    save_histogram_button_widget = ModernButtonControl(
        action_button_row,
        "Save histogram",
        width=126,
    )
    save_histogram_button_widget.grid(
        row=1,
        column=0,
        sticky="ew",
        pady=(6, 0),
    )

    include_metadata_var = tk.BooleanVar(value=False)
    include_metadata_checkbox = ctk.CTkCheckBox(
        action_controls,
        text="Include Metadata (.h5)",
        variable=include_metadata_var,
        onvalue=True,
        offvalue=False,
        width=180,
        checkbox_width=19,
        checkbox_height=19,
        corner_radius=5,
        border_width=2,
        border_color=VIEWER_BORDER,
        fg_color=VIEWER_ACCENT,
        hover_color=VIEWER_ACCENT_HOVER,
        text_color=VIEWER_TEXT,
        font=viewer_fonts.control_label,
    )
    include_metadata_checkbox.grid(
        row=1,
        column=0,
        sticky="w",
        pady=(7, 1),
    )

    choose_save_path_var = tk.BooleanVar(value=False)
    choose_save_path_checkbox = ctk.CTkCheckBox(
        action_controls,
        text="Choose file name and folder",
        variable=choose_save_path_var,
        onvalue=True,
        offvalue=False,
        width=210,
        checkbox_width=19,
        checkbox_height=19,
        corner_radius=5,
        border_width=2,
        border_color=VIEWER_BORDER,
        fg_color=VIEWER_ACCENT,
        hover_color=VIEWER_ACCENT_HOVER,
        text_color=VIEWER_TEXT,
        font=viewer_fonts.control_label,
    )
    choose_save_path_checkbox.grid(
        row=2,
        column=0,
        sticky="w",
        pady=(5, 1),
    )

    reset_button_widget = ModernButtonControl(
        controls_sidebar,
        "Reset viewer",
        width=260,
    )
    reset_button_widget.grid(row=7, column=0, sticky="ew", pady=(7, 2))

    status_text = ModernStatusLabel(
        plot_shell,
        text="Ready",
        height=48,          # Space for approximately two lines
        justify="left",     # Align multiple lines to the left
        anchor="w",
        text_color=VIEWER_MUTED_TEXT,
        fg_color="transparent",
        font=viewer_fonts.status,
    )
    status_text.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 7))
    bind_responsive_label_wrap(
        status_text,
        plot_shell,
        horizontal_padding=28,
    )

    sliders = [m_slider, line_slider, c_slider, bin_slider, median_window_slider]
    small_sliders = [
        hist_bin_slider_x,
        hist_bin_slider_y,
        bottom_slider,
        top_slider,
        vmin_slider,
        vmax_slider,
        cmap_start_slider,
        cmap_end_slider,
        lower_slider,
        upper_slider,
    ]
    all_sliders = sliders + small_sliders
    step_buttons = [
        button
        for slider in all_sliders
        for button in (slider.minus_button, slider.plus_button)
    ]
    set_toggle_button_style(equal_cut_rows_button_widget, mode.equal_cut_rows_enabled)
    set_toggle_button_style(zoom_to_cut_button_widget, mode.zoom_to_cut_enabled)
    set_toggle_button_style(fill_symmetric_button_widget, settings.symmetric_fill_enabled)
    set_toggle_button_style(median_filter_button_widget, settings.median_filter_enabled)
    set_toggle_button_style(local_filter_button_widget, settings.local_filter_enabled)
    set_toggle_button_style(filter_statistics_button_widget, mode.filter_statistics_enabled)
    c_ax.set_visible(mode.equal_cut_rows_enabled)
    filter_statistics_button_widget.active = bool(settings.local_filter_enabled)


    # -------------------------------------------------------------------------
    # Reset Defaults
    # -------------------------------------------------------------------------

    reset_defaults = {
        "tilt": 0.0,
        "line": y_line_default,
        "lower_percentile": 0.0,
        "upper_percentile": 100.0,
        "bottom_cut": 0,
        "top_cut": 0,
        "c": initial_symmetric_margin,
        "display_bin_x": hist_display_bin_x,
        "display_bin_y": hist_display_bin_y,
        "spectrum_bin": 1,
        "vmin": vmin_default,
        "vmax": max(vmax_default, 1.0),
        "cmap_start": 0.0,
        "cmap_end": 1.0,
        "median_filter_enabled": False,
        "median_filter_window": normalize_median_window(median_filter_window),
        "local_filter_enabled": False,
        "local_filter_window": normalize_median_window(local_filter_window),
        "local_filter_bottom_limit": int(local_filter_bottom_limit),
        "local_filter_upper_limit": int(local_filter_upper_limit),
        "symmetric_fill_enabled": False,
        "equal_cut_rows_enabled": True,
        "zoom_to_cut_enabled": False,
    }

    # -------------------------------------------------------------------------
    # Drawing/update helpers
    # -------------------------------------------------------------------------

    def normalize_percentile_sliders() -> tuple[float, float]:
        lower = float(lower_slider.val)
        upper = float(upper_slider.val)

        lower = min(lower, upper - 0.1)
        upper = max(upper, lower + 0.1)
        lower = max(float(lower_slider.valmin), min(lower, float(lower_slider.valmax)))
        upper = max(float(upper_slider.valmin), min(upper, float(upper_slider.valmax)))

        if lower != lower_slider.val:
            set_slider_val_silent(lower_slider, lower)
        if upper != upper_slider.val:
            set_slider_val_silent(upper_slider, upper)
        return lower, upper

    def update_cut_lines() -> None:
        cut_bottom_line.set_ydata([y_from_row_index(products.bottom_cut), y_from_row_index(products.bottom_cut)])
        cut_top_line.set_ydata([y_from_row_index(products.top_index), y_from_row_index(products.top_index)])

    def top_slider_value_from_cut(top_cut: int) -> int:
        """Convert the stored number of top rows cut to a bottom-up boundary."""
        row_count = max(int(products.base.shape[1]), 1)
        return int(np.clip(row_count - int(top_cut), 1, row_count))

    def top_cut_from_slider_value(slider_value: float) -> int:
        """Convert the displayed bottom-up top boundary to rows cut from above."""
        row_count = max(int(products.base.shape[1]), 1)
        top_index = int(np.clip(round(slider_value), 1, row_count))
        return row_count - top_index

    def sync_reference_line_visibility(draw: bool = False) -> None:
        line.set_visible(bool(mode.reference_line_visible))
        hide_ref_line_var.set(not bool(mode.reference_line_visible))

        if draw:
            fig_.canvas.draw_idle()

    sync_reference_line_visibility(draw=False)

    def disable_zoom_to_cut(draw: bool = False) -> None:
        if not mode.zoom_to_cut_enabled:
            return

        mode.zoom_to_cut_enabled = False

        set_toggle_button_style(
            zoom_to_cut_button_widget,
            False,
        )

        if draw:
            fig_.canvas.draw_idle()
    def update_color_limits(draw: bool = True) -> None:
        if bool(mode.filter_statistics_enabled):
            im.set_clim(vmin=0.0, vmax=1.0)

            if draw:
                fig_.canvas.draw_idle()

            return
        if vmin_slider.val > vmax_slider.val - 1.0:
            set_slider_val_silent(vmin_slider, max(vmax_slider.val - 1.0, 0.0))

        vmin = float(vmin_slider.val)
        vmax = min(float(vmax_slider.val), float(vmax_slider.valmax))
        vmax = max(vmax, vmin + 1.0)
        im.set_clim(vmin=vmin, vmax=vmax)

        if draw:
            fig_.canvas.draw_idle()

    def update_vmax_slider_range() -> None:
        nonzero = products.display[products.display > 0]

        if nonzero.size == 0:
            vmax_limit = 1.0
        else:
            vmax_limit = float(np.percentile(nonzero, 99.995))

        vmax_limit = max(vmax_limit, 1.0)

        old_vmin = float(vmin_slider.val)
        old_vmax = float(vmax_slider.val)

        new_vmax_slider_max = int(max(np.ceil(vmax_limit), 1))


        vmax_slider.valmax = new_vmax_slider_max
        vmin_slider.valmax = max(float(vmax_slider.valmax) - 1.0, 0.0)

        try:
            vmax_slider.ax.set_xlim(vmax_slider.valmin, vmax_slider.valmax)
            vmin_slider.ax.set_xlim(vmin_slider.valmin, vmin_slider.valmax)
        except Exception:
            pass


        new_vmax = old_vmax
        new_vmin = old_vmin

        if new_vmax > float(vmax_slider.valmax):
            new_vmax = float(vmax_slider.valmax)

        if new_vmax < float(vmax_slider.valmin):
            new_vmax = float(vmax_slider.valmin)

        if new_vmin > float(vmin_slider.valmax):
            new_vmin = float(vmin_slider.valmax)

        if new_vmin < float(vmin_slider.valmin):
            new_vmin = float(vmin_slider.valmin)


        if new_vmin > new_vmax - 1.0:
            new_vmin = max(float(vmin_slider.valmin), new_vmax - 1.0)

        if new_vmax < new_vmin + 1.0:
            new_vmax = min(float(vmax_slider.valmax), new_vmin + 1.0)

        # Nur setzen, wenn sich wirklich etwas ändern musste
        if new_vmin != old_vmin:
            set_slider_val_silent(vmin_slider, new_vmin)

        if new_vmax != old_vmax:
            set_slider_val_silent(vmax_slider, new_vmax)

    def update_projection(
        draw: bool = True,
        reset_xlim: bool = False,
        reset_ylim: bool = False,
    ) -> None:
        """
        Update the 1D spectrum line.

        By default, keep the current y-axis zoom.
        Only reset the y-axis when reset_ylim=True.
        """

        projection_line.set_data(products.spectrum_x, products.spectrum_y)

        if reset_xlim:
            ax[1].set_xlim(products.x_edges[0], products.x_edges[-1])

        if reset_ylim:
            ax[1].set_ylim(*get_spectrum_y_limits())

        if draw:
            fig_.canvas.draw_idle()

    def sync_equal_cut_slider_limits() -> None:
        max_row_index = max(int(products.base.shape[1] - 1), 0)

        if not mode.equal_cut_rows_enabled:
            bottom_slider.valmin = 0
            bottom_slider.valmax = max_row_index
            top_slider.valmin = 1
            top_slider.valmax = max_row_index + 1
            try:
                bottom_slider.ax.set_xlim(bottom_slider.valmin, bottom_slider.valmax)
                top_slider.ax.set_xlim(top_slider.valmin, top_slider.valmax)
            except Exception:
                pass
            return

        ref_row = row_index_from_y(line_slider.val)
        max_margin = max(min(ref_row, int(products.base.shape[1] - ref_row - 1)), 0)

        c_slider.valmin = 0
        c_slider.valmax = max_margin
        try:
            c_slider.ax.set_xlim(c_slider.valmin, c_slider.valmax)
        except Exception:
            pass
        if float(c_slider.val) > float(c_slider.valmax):
            set_slider_val_silent(c_slider, c_slider.valmax)

    def sync_dynamic_limits_after_new_histogram() -> None:
        max_row_index = max(int(products.base.shape[1] - 1), 0)
        bottom_slider.valmax = max_row_index
        top_slider.valmin = 1
        top_slider.valmax = max_row_index + 1

        if bottom_slider.val > bottom_slider.valmax:
            set_slider_val_silent(bottom_slider, bottom_slider.valmax)
        set_slider_val_silent(
            top_slider,
            top_slider_value_from_cut(products.top_cut),
        )

        line_slider.valmin = float(products.y_edges[0])
        line_slider.valmax = float(products.y_edges[-1])
        try:
            line_slider.ax.set_xlim(line_slider.valmin, line_slider.valmax)
        except Exception:
            pass

        if line_slider.val < line_slider.valmin:
            set_slider_val_silent(line_slider, line_slider.valmin)
        if line_slider.val > line_slider.valmax:
            set_slider_val_silent(line_slider, line_slider.valmax)
        line.set_ydata([line_slider.val, line_slider.val])

        sync_equal_cut_slider_limits()

    def derive_equal_cut_rows() -> tuple[int, int, int]:
        ref_row = row_index_from_y(line_slider.val)
        c_value = int(round(c_slider.val))
        max_margin = max(min(ref_row, int(products.base.shape[1] - ref_row - 1)), 0)
        c_value = int(np.clip(c_value, 0, max_margin))

        bottom_cut = max(ref_row - c_value, 0)
        top_index = min(ref_row + c_value + 1, int(products.base.shape[1]))
        top_cut = int(products.base.shape[1] - top_index)
        return bottom_cut, top_cut, c_value

    def sync_equal_cut_control_visibility() -> None:
        equal_enabled = bool(mode.equal_cut_rows_enabled)
        bottom_slider.ax.set_visible(not equal_enabled)
        top_slider.ax.set_visible(not equal_enabled)
        c_ax.set_visible(equal_enabled)

        bottom_slider.active = not equal_enabled
        top_slider.active = not equal_enabled
        c_slider.active = equal_enabled

        fig_.canvas.draw_idle()

    def sync_settings_from_sliders() -> None:
        lower, upper = normalize_percentile_sliders()
        settings.tilt = float(m_slider.val)
        settings.lower_percentile = lower
        settings.upper_percentile = upper
        settings.spectrum_bin = int(bin_slider.val)
        settings.display_bin_x = int(hist_bin_slider_x.val)
        settings.display_bin_y = int(hist_bin_slider_y.val)
        settings.symmetric_fill_enabled = bool(settings.symmetric_fill_enabled)
        settings.median_filter_window = normalize_median_window(int(round(median_window_slider.val)))
        settings.local_filter_window = normalize_median_window(
            int(local_window_control["get_value"]())
        )

        settings.local_filter_bottom_limit = int(
            np.clip(int(local_min_perc_control["get_value"]()), 0, 99)
        )

        settings.local_filter_upper_limit = int(
            np.clip(
                int(local_max_perc_control["get_value"]()),
                settings.local_filter_bottom_limit + 1,
                100,
            )
        )

        if mode.equal_cut_rows_enabled:
            # bottom/top are calculated in recompute_all after product limits are current.
            return

        settings.bottom_cut = int(round(bottom_slider.val))
        settings.top_cut = top_cut_from_slider_value(top_slider.val)
        mode.manual_bottom_cut = settings.bottom_cut
        mode.manual_top_cut = settings.top_cut

    def update_main_image(draw: bool = True) -> None:
        stats_image = products.local_filter_complement

        if (
            bool(mode.filter_statistics_enabled)
            and bool(settings.local_filter_enabled)
            and stats_image is not None
        ):
            im.set_data(stats_image.T)
            im.set_extent([
                products.x_edges[0],
                products.x_edges[-1],
                products.y_edges[0],
                products.y_edges[-1],
            ])

            im.set_clim(vmin=0.0, vmax=1.0)

            changed_pixels = int(np.count_nonzero(stats_image))
            total_pixels = int(stats_image.size)
            fraction = 100.0 * changed_pixels / max(total_pixels, 1)

            status_text.set_text(
                f"Changed pixels:  {fraction:.3f}%"
            )

        else:
            im.set_data(products.display.T)
            im.set_extent([
                products.x_edges[0],
                products.x_edges[-1],
                products.y_edges[0],
                products.y_edges[-1],
            ])

            update_vmax_slider_range()
            update_color_limits(draw=False)

        if draw:
            fig_.canvas.draw_idle()

    def clear_filter_statistics_status() -> None:
        text = status_text.get_text()

        if (
            text.startswith("Changed pixels:")
            or text.startswith("No local-filter statistics")
        ):
            status_text.set_text("")

    def recompute_all(*, draw: bool = True, reset_xlim: bool = False) -> None:
        nonlocal products

        sync_settings_from_sliders()

        products = pipeline.compute(settings)
        sync_dynamic_limits_after_new_histogram()

        if mode.equal_cut_rows_enabled:
            bottom_cut, top_cut, c_value = derive_equal_cut_rows()
            settings.bottom_cut = bottom_cut
            settings.top_cut = top_cut

            if int(round(c_slider.val)) != c_value:
                set_slider_val_silent(c_slider, c_value)
            if int(round(bottom_slider.val)) != bottom_cut:
                set_slider_val_silent(bottom_slider, bottom_cut)
            top_slider_value = top_slider_value_from_cut(top_cut)
            if int(round(top_slider.val)) != top_slider_value:
                set_slider_val_silent(top_slider, top_slider_value)

            products = pipeline.compute(settings)

        update_main_image(draw=False)
        update_cut_lines()
        update_projection(draw=False, reset_xlim=reset_xlim)

        if draw:
            fig_.canvas.draw_idle()


    def sync_filter_statistics_button_visibility(draw: bool = False) -> None:
        available = bool(settings.local_filter_enabled)

        # Local parameters remain editable while the filter is off. Only the
        # statistics action requires an active local-filter result.
        filter_statistics_button_widget.active = available

        if not available:
            mode.filter_statistics_enabled = False

        set_toggle_button_style(
            filter_statistics_button_widget,
            mode.filter_statistics_enabled,
        )

        if draw:
            fig_.canvas.draw_idle()

    def sync_toggle_controls() -> None:
        # Reset and session restore both change several flags at once. Update
        # every related label and visual state from those flags in one place.
        equal_cut_rows_button_widget.label.set_text(
            "Symm. ROI: ON" if mode.equal_cut_rows_enabled else "Symm. ROI: OFF"
        )
        fill_symmetric_button_widget.label.set_text(
            "Symmetric Fill: ON"
            if settings.symmetric_fill_enabled
            else "Symmetric Fill: OFF"
        )
        median_filter_button_widget.label.set_text(
            "Median: ON" if settings.median_filter_enabled else "Median: OFF"
        )
        local_filter_button_widget.label.set_text(
            "Local: ON" if settings.local_filter_enabled else "Local: OFF"
        )
        zoom_to_cut_button_widget.label.set_text("Focus ROI")

        toggle_states = (
            (equal_cut_rows_button_widget, mode.equal_cut_rows_enabled),
            (fill_symmetric_button_widget, settings.symmetric_fill_enabled),
            (median_filter_button_widget, settings.median_filter_enabled),
            (local_filter_button_widget, settings.local_filter_enabled),
            (zoom_to_cut_button_widget, mode.zoom_to_cut_enabled),
            (filter_statistics_button_widget, mode.filter_statistics_enabled),
        )
        for button, enabled in toggle_states:
            set_toggle_button_style(button, enabled)

    # -------------------------------------------------------------------------
    # Callbacks
    # -------------------------------------------------------------------------

    def update_tilt(_value) -> None:
        settings.tilt = float(m_slider.val)
        recompute_all(draw=True, reset_xlim=False)

    def update_filter(_value) -> None:
        recompute_all(draw=True)

    def update_cut(_value) -> None:
        disable_zoom_to_cut(draw=False)
        recompute_all(draw=True)

    def update_cut_margin(_value) -> None:
        if mode.equal_cut_rows_enabled:
            disable_zoom_to_cut(draw=False)
            recompute_all(draw=True)

    def update_bin(_value) -> None:
        recompute_all(draw=True)

    def update_vmin(_value) -> None:
        update_color_limits(draw=True)

    def update_vmax(_value) -> None:
        update_color_limits(draw=True)

    def update_line(_value) -> None:
        line.set_ydata([line_slider.val, line_slider.val])

        # Visibility remains controlled by the Hide Ref. checkbox.
        line.set_visible(bool(mode.reference_line_visible))

        sync_equal_cut_slider_limits()

        if mode.equal_cut_rows_enabled:
            disable_zoom_to_cut(draw=False)
            recompute_all(draw=True)
            return

        fig_.canvas.draw_idle()

    def toggle_reference_line_visibility() -> None:
        mode.reference_line_visible = not bool(hide_ref_line_var.get())
        sync_reference_line_visibility(draw=True)

    def normalize_colormap_range(changed_control: str) -> tuple[float, float]:
        values = np.asarray(cmap_range_values, dtype=float)
        start_index = int(np.argmin(np.abs(values - cmap_start_slider.val)))
        end_index = int(np.argmin(np.abs(values - cmap_end_slider.val)))

        if start_index >= end_index:
            if changed_control == "start":
                start_index = max(end_index - 1, 0)
                if start_index == end_index:
                    end_index = min(start_index + 1, len(values) - 1)
            else:
                end_index = min(start_index + 1, len(values) - 1)
                if start_index == end_index:
                    start_index = max(end_index - 1, 0)

        start_value = float(values[start_index])
        end_value = float(values[end_index])
        set_slider_val_silent(cmap_start_slider, start_value)
        set_slider_val_silent(cmap_end_slider, end_value)
        return start_value, end_value

    def update_colormap(selected_cmap: str | None) -> None:
        if selected_cmap is None:
            return

        start_value, end_value = normalize_colormap_range("selection")
        if np.isclose(start_value, 0.0) and np.isclose(end_value, 1.0):
            im.set_cmap(selected_cmap)
        else:
            source_colormap = plt.get_cmap(selected_cmap)
            color_samples = source_colormap(
                np.linspace(start_value, end_value, 256)
            )
            ranged_colormap = LinearSegmentedColormap.from_list(
                f"{selected_cmap}_{start_value:.3f}_{end_value:.3f}",
                color_samples,
                N=256,
            )
            im.set_cmap(ranged_colormap)

        fig_.canvas.draw_idle()

    def update_colormap_start(_value) -> None:
        normalize_colormap_range("start")
        update_colormap(cmap_radio.get())

    def update_colormap_end(_value) -> None:
        normalize_colormap_range("end")
        update_colormap(cmap_radio.get())

    def open_colormap_dialog(_event) -> None:
        nonlocal colormap_dialog

        if colormap_dialog is not None and colormap_dialog.winfo_exists():
            colormap_dialog.lift()
            colormap_dialog.focus_force()
            return

        dialog = ctk.CTkToplevel(embedded_container)
        colormap_dialog = dialog
        dialog.title("Add colormap")
        dialog.geometry("420x520")
        dialog.minsize(360, 400)
        dialog.configure(fg_color=VIEWER_BG)
        dialog.transient(embedded_container.winfo_toplevel())
        dialog.grid_columnconfigure(0, weight=1)
        dialog.grid_rowconfigure(3, weight=1)

        ctk.CTkLabel(
            dialog,
            text="ADD COLORMAP",
            anchor="w",
            text_color=VIEWER_TEXT,
            font=viewer_fonts.card_title,
        ).grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 1))

        ctk.CTkLabel(
            dialog,
            text="Choose an additional Matplotlib colormap",
            anchor="w",
            text_color=VIEWER_MUTED_TEXT,
            font=viewer_fonts.card_note,
        ).grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 10))

        search_var = tk.StringVar()
        search_entry = ctk.CTkEntry(
            dialog,
            textvariable=search_var,
            placeholder_text="Search colormaps...",
            height=36,
            corner_radius=9,
            border_width=1,
            border_color=VIEWER_BORDER,
            fg_color=VIEWER_CONTROL,
            text_color=VIEWER_TEXT,
            font=viewer_fonts.value,
        )
        search_entry.grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 10))

        result_frame = ctk.CTkScrollableFrame(
            dialog,
            fg_color=VIEWER_PANEL,
            border_color=VIEWER_BORDER,
            border_width=1,
            corner_radius=11,
        )
        result_frame.grid(row=3, column=0, sticky="nsew", padx=18, pady=(0, 14))
        result_frame.grid_columnconfigure(0, weight=1)

        def close_dialog() -> None:
            nonlocal colormap_dialog
            try:
                dialog.grab_release()
            except tk.TclError:
                pass
            dialog.destroy()
            colormap_dialog = None

        def add_colormap(selected_cmap: str) -> None:
            if selected_cmap not in cmap_options:
                cmap_options.append(selected_cmap)
                cmap_radio.configure(values=list(cmap_options))

            cmap_radio.set(selected_cmap)
            update_colormap(selected_cmap)
            status_text.set_text(f"Colormap: {selected_cmap}")
            close_dialog()

        def refresh_colormap_results(*_args) -> None:
            for child in result_frame.winfo_children():
                child.destroy()

            query = search_var.get().strip().casefold()
            available = [
                name
                for name in sorted(plt.colormaps(), key=str.casefold)
                if name not in cmap_options and query in name.casefold()
            ]

            if not available:
                ctk.CTkLabel(
                    result_frame,
                    text="No additional colormaps found",
                    text_color=VIEWER_MUTED_TEXT,
                    font=viewer_fonts.card_note,
                ).grid(row=0, column=0, sticky="ew", padx=10, pady=18)
                return

            for row, colormap_name in enumerate(available):
                ctk.CTkButton(
                    result_frame,
                    text=colormap_name,
                    height=32,
                    anchor="w",
                    corner_radius=7,
                    fg_color="transparent",
                    hover_color=VIEWER_CONTROL_HOVER,
                    text_color=VIEWER_TEXT,
                    font=viewer_fonts.value,
                    command=lambda name=colormap_name: add_colormap(name),
                ).grid(row=row, column=0, sticky="ew", padx=5, pady=2)

        search_var.trace_add("write", refresh_colormap_results)
        dialog.protocol("WM_DELETE_WINDOW", close_dialog)
        refresh_colormap_results()
        dialog.update_idletasks()

        parent = embedded_container.winfo_toplevel()
        x_position = parent.winfo_rootx() + max(
            (parent.winfo_width() - dialog.winfo_width()) // 2,
            0,
        )
        y_position = parent.winfo_rooty() + max(
            (parent.winfo_height() - dialog.winfo_height()) // 2,
            0,
        )
        dialog.geometry(f"+{x_position}+{y_position}")
        dialog.grab_set()
        search_entry.focus_set()

    def toggle_equal_cut_rows(_event) -> None:
        disable_zoom_to_cut(draw=False)

        mode.equal_cut_rows_enabled = not mode.equal_cut_rows_enabled

        if mode.equal_cut_rows_enabled:
            mode.manual_bottom_cut = int(round(bottom_slider.val))
            mode.manual_top_cut = top_cut_from_slider_value(top_slider.val)
            sync_equal_cut_control_visibility()
            sync_equal_cut_slider_limits()
            equal_cut_rows_button_widget.label.set_text("Symm. ROI: ON")
        else:
            settings.bottom_cut = mode.manual_bottom_cut
            settings.top_cut = mode.manual_top_cut
            set_slider_val_silent(bottom_slider, mode.manual_bottom_cut)
            set_slider_val_silent(
                top_slider,
                top_slider_value_from_cut(mode.manual_top_cut),
            )
            sync_equal_cut_control_visibility()
            sync_equal_cut_slider_limits()
            equal_cut_rows_button_widget.label.set_text("Symm. ROI: OFF")

        set_toggle_button_style(
            equal_cut_rows_button_widget,
            mode.equal_cut_rows_enabled,
        )

        recompute_all(draw=True)

    def toggle_zoom_to_cut(_event) -> None:
        mode.zoom_to_cut_enabled = not bool(mode.zoom_to_cut_enabled)

        if mode.zoom_to_cut_enabled:
            ax[0].set_xlim(products.x_edges[0], products.x_edges[-1])
            ax[0].set_ylim(
                y_from_row_index(products.bottom_cut),
                y_from_row_index(products.top_index),
            )
        else:
            ax[0].set_xlim(products.x_edges[0], products.x_edges[-1])
            ax[0].set_ylim(products.y_edges[0], products.y_edges[-1])

        set_toggle_button_style(
            zoom_to_cut_button_widget,
            mode.zoom_to_cut_enabled,
        )

        fig_.canvas.draw_idle()

    def calculate_tilt_(_event) -> None:
        selected_speedup = max(int(tilt_speedup_control.get()), 1)
        selected = scans[::selected_speedup]
        if selected.size and selected[-1] != scans[-1]:
            selected = np.append(selected, scans[-1])

        selected_scan_files = [
            resolved_scan_files_by_scan[int(scan)]
            for scan in selected
        ]

        x_, y_ = _load_scan_arrays(
            selected_scan_files
        )

        try:
            x_max_fit = estimate_tilt_from_arrays(x_, y_)

        except Exception as exc:
            status_text.set_text(f"Tilt failed: {exc}")
            fig_.canvas.draw_idle()
            return

        x_max_fit = float(np.clip(x_max_fit, m_slider.valmin, m_slider.valmax))
        m_slider.set_val(x_max_fit)
        calculate_ridge_(_event)

    def calculate_ridge_(_event) -> None:
        try:
            ridge = pipeline.estimate_ridge(float(m_slider.val))
        except Exception as exc:
            status_text.set_text(f"Ridge failed: {exc}")
            fig_.canvas.draw_idle()
            return
        if ridge is None:
            status_text.set_text("Ridge failed")
            fig_.canvas.draw_idle()
            return
        line_slider.set_val(float(ridge))

    def toggle_fill_symmetric(_event) -> None:
        previous = bool(settings.symmetric_fill_enabled)
        settings.symmetric_fill_enabled = not previous

        fill_symmetric_button_widget.label.set_text(
            "Symmetric Fill: ON"
            if settings.symmetric_fill_enabled
            else "Symmetric Fill: OFF"
        )

        set_toggle_button_style(
            fill_symmetric_button_widget,
            settings.symmetric_fill_enabled,
        )

        pipeline.clear_cache()

        try:
            recompute_all(draw=True, reset_xlim=False)
        except Exception as exc:
            settings.symmetric_fill_enabled = previous

            fill_symmetric_button_widget.label.set_text(
                "Symmetric Fill: ON"
                if settings.symmetric_fill_enabled
                else "Symmetric Fill: OFF"
            )

            set_toggle_button_style(
                fill_symmetric_button_widget,
                settings.symmetric_fill_enabled,
            )

            pipeline.clear_cache()
            recompute_all(draw=True, reset_xlim=False)

            status_text.set_text(f"Symmetric fill failed: {exc}")
            fig_.canvas.draw_idle()
            return

        if products.ridge is not None:
            set_slider_val_silent(line_slider, float(products.ridge))
            line.set_ydata([line_slider.val, line_slider.val])

            if mode.equal_cut_rows_enabled:
                recompute_all(draw=True)
            else:
                fig_.canvas.draw_idle()

    def toggle_median_filter(_event) -> None:
        settings.median_filter_enabled = not bool(settings.median_filter_enabled)

        median_filter_button_widget.label.set_text(
            "Median: ON" if settings.median_filter_enabled else "Median: OFF"
        )

        set_toggle_button_style(
            median_filter_button_widget,
            settings.median_filter_enabled,
        )

        pipeline.clear_cache()
        recompute_all(draw=True, reset_xlim=False)

    def toggle_local_filter(_event) -> None:
        settings.local_filter_enabled = not bool(settings.local_filter_enabled)

        local_filter_button_widget.label.set_text(
            "Local: ON" if settings.local_filter_enabled else "Local: OFF"
        )

        set_toggle_button_style(
            local_filter_button_widget,
            settings.local_filter_enabled,
        )

        if not settings.local_filter_enabled:
            mode.filter_statistics_enabled = False
            clear_filter_statistics_status()

        sync_filter_statistics_button_visibility(draw=False)

        pipeline.clear_cache()
        recompute_all(draw=True, reset_xlim=False)

    def update_median_window(_value) -> None:
        new_window = normalize_median_window(int(round(median_window_slider.val)))

        if int(round(median_window_slider.val)) != new_window:
            set_slider_val_silent(median_window_slider, new_window)

        settings.median_filter_window = new_window
        pipeline.clear_cache()

        if settings.median_filter_enabled:
            recompute_all(draw=True, reset_xlim=False)

    # -------------------------------------------------------------------------
    # Data export
    # -------------------------------------------------------------------------

    def choose_output_file(
        directory: Path,
        default_name: str,
        extension: str,
        file_description: str,
    ) -> Path | None:
        """Open a native Save As dialog when manual output naming is enabled."""

        initial_directory = directory.expanduser()
        while not initial_directory.exists() and initial_directory != initial_directory.parent:
            initial_directory = initial_directory.parent
        if not initial_directory.is_dir():
            initial_directory = Path.cwd()

        selected_path = filedialog.asksaveasfilename(
            parent=tk_parent.winfo_toplevel(),
            title=f"Save {file_description}",
            initialdir=str(initial_directory),
            initialfile=f"{default_name}{extension}",
            defaultextension=extension,
            filetypes=[
                (f"{file_description} ({extension})", f"*{extension}"),
                ("All files", "*.*"),
            ],
        )
        return Path(selected_path) if selected_path else None

    def build_spectrum_export() -> np.ndarray:
        """Return Pixel, Energy, Loss and Counts with 1D binning fixed to one."""

        export_settings = replace(settings, spectrum_bin=1)
        export_products = pipeline.compute(export_settings)

        pixel_values = np.asarray(export_products.spectrum_x, dtype=float)
        count_values = np.asarray(export_products.spectrum_y, dtype=float)

        if axis_calibration.has_energy_axis:
            energy_values = np.asarray(pixel_to_energy(pixel_values), dtype=float)
        else:
            energy_values = pixel_values.copy()

        if axis_calibration.has_loss_axis:
            loss_values = np.asarray(pixel_to_loss(pixel_values), dtype=float)
        else:
            loss_values = pixel_values.copy()

        return np.vstack(
            (pixel_values, energy_values, loss_values, count_values)
        )

    def save_current_spectrum(_event) -> None:
        spectrum = build_spectrum_export()

        if not include_metadata_var.get():
            if choose_save_path_var.get():
                output_path = choose_output_file(
                    spectra_save_dir,
                    "spectrum",
                    ".npy",
                    "spectrum",
                )
                if output_path is None:
                    status_text.set_text("Spectrum save cancelled")
                    fig_.canvas.draw_idle()
                    return
            else:
                output_path = next_numbered_path(
                    spectra_save_dir,
                    "spectrum",
                )
            if choose_save_path_var.get():
                with output_path.open("wb") as output_file:
                    np.save(output_file, spectrum)
            else:
                np.save(output_path, spectrum)
            status_text.set_text(
                f"Saved as: {output_path.name} (4 axes: pixel, energy, loss, counts)\n"
                f"Folder: {output_path.parent.resolve()}"
            )
            fig_.canvas.draw_idle()
            return

        try:
            # A regular import is visible to PyInstaller's dependency analysis.
            import h5py
        except ImportError:
            status_text.set_text(
                "Metadata export requires the optional 'h5py' package"
            )
            fig_.canvas.draw_idle()
            return

        def write_hdf5_value(group, key: object, value: object) -> None:
            dataset_name = str(key).replace("/", "∕")
            if isinstance(value, dict):
                subgroup = group.create_group(dataset_name)
                for child_key, child_value in value.items():
                    write_hdf5_value(subgroup, child_key, child_value)
                return
            if value is None:
                dataset = group.create_dataset(
                    dataset_name,
                    data="",
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
                dataset.attrs["is_none"] = True
                return
            if isinstance(value, str):
                group.create_dataset(
                    dataset_name,
                    data=value,
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
                return
            if isinstance(value, (list, tuple)) and any(
                isinstance(item, (dict, list, tuple)) for item in value
            ):
                subgroup = group.create_group(dataset_name)
                subgroup.attrs["sequence"] = True
                for index, item in enumerate(value):
                    write_hdf5_value(subgroup, str(index), item)
                return

            array = np.asarray(value)
            if array.dtype.kind in {"U", "O"}:
                text_value = (
                    str(array.item())
                    if array.ndim == 0
                    else json.dumps(value, ensure_ascii=False)
                )
                group.create_dataset(
                    dataset_name,
                    data=text_value,
                    dtype=h5py.string_dtype(encoding="utf-8"),
                )
                return

            dataset_kwargs = {}
            if array.ndim > 0 and array.size >= 32:
                dataset_kwargs = {"compression": "gzip", "shuffle": True}
            group.create_dataset(dataset_name, data=array, **dataset_kwargs)

        metadata_payload = dict(export_metadata or {})
        calibration_metadata = metadata_payload.get("calibration")
        if isinstance(calibration_metadata, dict):
            calibration_metadata = dict(calibration_metadata)
            calibration_lines = calibration_metadata.get("calibration_lines")
            if isinstance(calibration_lines, list):
                calibration_metadata["calibration_lines"] = [
                    {
                        key: value
                        for key, value in line.items()
                        if key != "plot_data"
                    }
                    if isinstance(line, dict)
                    else line
                    for line in calibration_lines
                ]
            metadata_payload["calibration"] = calibration_metadata

        metadata_payload["processing"] = {
            "tilt": float(settings.tilt),
            "lower_percentile": float(settings.lower_percentile),
            "upper_percentile": float(settings.upper_percentile),
            "bottom_cut": int(settings.bottom_cut),
            "top_cut": int(settings.top_cut),
            "symmetric_fill_enabled": bool(settings.symmetric_fill_enabled),
            "symmetric_fill_margin": int(round(c_slider.val)),
            "median_filter_enabled": bool(settings.median_filter_enabled),
            "median_filter_window": int(settings.median_filter_window),
            "local_filter_enabled": bool(settings.local_filter_enabled),
            "local_filter_window": int(settings.local_filter_window),
            "local_filter_bottom_limit": int(
                settings.local_filter_bottom_limit
            ),
            "local_filter_upper_limit": int(
                settings.local_filter_upper_limit
            ),
            "spectrum_bin": 1,
        }

        if choose_save_path_var.get():
            output_path = choose_output_file(
                spectra_save_dir,
                "spectrum",
                ".h5",
                "spectrum with metadata",
            )
            if output_path is None:
                status_text.set_text("Spectrum save cancelled")
                fig_.canvas.draw_idle()
                return
        else:
            output_path = next_numbered_path(
                spectra_save_dir,
                "spectrum",
                suffix=".h5",
            )
        try:
            with h5py.File(output_path, "w") as output_file:
                output_file.attrs["format"] = "meV-RIXS spectrum"
                output_file.attrs["format_version"] = 1
                spectrum_group = output_file.create_group("spectrum")
                for index, (name, unit) in enumerate(
                    (
                        ("pixel", "pixel"),
                        ("energy", "eV"),
                        ("loss", "eV"),
                        ("counts", "counts"),
                    )
                ):
                    dataset = spectrum_group.create_dataset(
                        name,
                        data=np.asarray(spectrum[index], dtype=float),
                        compression="gzip",
                        shuffle=True,
                    )
                    dataset.attrs["unit"] = unit

                metadata_group = output_file.create_group("metadata")
                for key, value in metadata_payload.items():
                    write_hdf5_value(metadata_group, key, value)
        except Exception as exc:
            status_text.set_text(f"Spectrum export failed: {exc}")
            fig_.canvas.draw_idle()
            return

        status_text.set_text(
            f"Saved as: {output_path.name} (spectrum + metadata)\n"
            f"Folder: {output_path.parent.resolve()}"
        )
        fig_.canvas.draw_idle()

    def save_current_histogram(_event) -> None:
        if choose_save_path_var.get():
            output_path = choose_output_file(
                histogram_save_dir,
                "histogram",
                ".npy",
                "histogram",
            )
            if output_path is None:
                status_text.set_text("Histogram save cancelled")
                fig_.canvas.draw_idle()
                return
        else:
            output_path = next_numbered_path(
                histogram_save_dir,
                "histogram",
            )
        if choose_save_path_var.get():
            with output_path.open("wb") as output_file:
                np.save(output_file, products.cut)
        else:
            np.save(output_path, products.cut)
        status_text.set_text(f"Saved as: {output_path.name} \n"
                             f"Folder: {output_path.parent.resolve()}")
        fig_.canvas.draw_idle()

    def toggle_filter_statistics(_event) -> None:
        if not bool(settings.local_filter_enabled):
            mode.filter_statistics_enabled = False
            clear_filter_statistics_status()
            sync_filter_statistics_button_visibility(draw=True)
            update_main_image(draw=True)
            return

        if products.local_filter_complement is None:
            mode.filter_statistics_enabled = False
            sync_filter_statistics_button_visibility(draw=True)
            status_text.set_text("No local-filter statistics available")
            fig_.canvas.draw_idle()
            return

        mode.filter_statistics_enabled = not bool(mode.filter_statistics_enabled)

        set_toggle_button_style(
            filter_statistics_button_widget,
            mode.filter_statistics_enabled,
        )

        if not mode.filter_statistics_enabled:
            clear_filter_statistics_status()

        update_main_image(draw=True)

    def sync_axis_button_styles() -> None:
        """
        Update Energy/Loss button styles.
        """

        if energy_axis_button_widget is not None:
            set_toggle_button_style(
                energy_axis_button_widget,
                spectrum_axis_mode == "energy",
            )

        if loss_axis_button_widget is not None:
            set_toggle_button_style(
                loss_axis_button_widget,
                spectrum_axis_mode == "loss",
            )


    # -------------------------------------------------------------------------
    # Spectrum-axis calibration
    # -------------------------------------------------------------------------

    def get_calibration_state() -> dict[str, object]:
        """Return the calibration currently used by the open viewer."""

        current_coefficients = None

        if axis_calibration.has_energy_axis:
            assert axis_calibration.a3 is not None
            assert axis_calibration.a2 is not None
            assert axis_calibration.a1 is not None
            assert axis_calibration.a0 is not None

            current_coefficients = (
                float(axis_calibration.a3),
                float(axis_calibration.a2),
                float(axis_calibration.a1),
                float(axis_calibration.a0),
            )

        calibration_metadata = export_metadata.get("calibration")

        return {
            "energy_calibration": current_coefficients,
            "incident_energy": axis_calibration.incident_energy,
            "calibration_metadata": copy.deepcopy(calibration_metadata),
        }


    def apply_calibration_model(
        new_energy_calibration: tuple[float, float, float, float],
        new_incident_energy: float | None = None,
        calibration_metadata: dict | None = None,
    ) -> None:
        """Apply a new pixel-to-energy model without rebuilding the viewer."""

        nonlocal axis_calibration
        nonlocal energy_calibration
        nonlocal incident_energy
        nonlocal pixel_to_energy, energy_to_pixel
        nonlocal pixel_to_loss, loss_to_pixel
        nonlocal energy_axis, loss_axis
        nonlocal energy_axis_button_widget, loss_axis_button_widget
        nonlocal energy_axis_button_cid, loss_axis_button_cid
        nonlocal spectrum_axis_mode

        if len(new_energy_calibration) != 4:
            raise ValueError(
                "new_energy_calibration must contain four coefficients "
                "(a3, a2, a1, a0)."
            )
        normalized_coefficients = (
            float(new_energy_calibration[0]),
            float(new_energy_calibration[1]),
            float(new_energy_calibration[2]),
            float(new_energy_calibration[3]),
        )

        axis_calibration = SpectrumAxisCalibration.from_values(
            normalized_coefficients,
            new_incident_energy,
        )
        energy_calibration = normalized_coefficients
        incident_energy = axis_calibration.incident_energy

        pixel_to_energy = axis_calibration.pixel_to_energy
        energy_to_pixel = axis_calibration.energy_to_pixel
        pixel_to_loss = axis_calibration.pixel_to_loss
        loss_to_pixel = axis_calibration.loss_to_pixel

        if energy_axis is None:
            energy_axis = ax[1].secondary_xaxis(
                "bottom",
                functions=(pixel_to_energy, energy_to_pixel),
            )
            energy_axis.set_visible(False)
            energy_axis.set_xlabel("Energy")
            energy_axis.tick_params(axis="x", colors=rcParams["xtick.color"])
            energy_axis.xaxis.label.set_color(rcParams["axes.labelcolor"])
        else:
            energy_axis.set_functions((pixel_to_energy, energy_to_pixel))

        if axis_calibration.has_loss_axis:
            if loss_axis is None:
                loss_axis = ax[1].secondary_xaxis(
                    "bottom",
                    functions=(pixel_to_loss, loss_to_pixel),
                )
                loss_axis.set_visible(False)
                loss_axis.set_xlabel("Energy loss")
                loss_axis.tick_params(
                    axis="x",
                    colors=rcParams["xtick.color"],
                )
                loss_axis.xaxis.label.set_color(rcParams["axes.labelcolor"])
            else:
                loss_axis.set_functions((pixel_to_loss, loss_to_pixel))
        elif loss_axis is not None:
            loss_axis.set_visible(False)

        if energy_axis_button_widget is None:
            energy_axis_button_widget = ModernButtonControl(
                spectrum_button_row,
                "Energy",
                width=80,
            )
            energy_axis_button_widget.grid(
                row=0,
                column=1,
                sticky="ew",
                padx=3,
            )
            energy_axis_button_cid = energy_axis_button_widget.on_clicked(
                toggle_energy_axis
            )

        if axis_calibration.has_loss_axis:
            if loss_axis_button_widget is None:
                loss_axis_button_widget = ModernButtonControl(
                    spectrum_button_row,
                    "Loss",
                    width=80,
                )
                loss_axis_button_cid = loss_axis_button_widget.on_clicked(
                    toggle_loss_axis
                )
            loss_axis_button_widget.grid(
                row=0,
                column=2,
                sticky="ew",
                padx=(3, 0),
            )
        elif loss_axis_button_widget is not None:
            loss_axis_button_widget.grid_remove()

        if calibration_metadata is not None:
            export_metadata["calibration"] = copy.deepcopy(calibration_metadata)
            calibration_scans = calibration_metadata.get("calibration_scans")
            if calibration_scans is not None:
                export_metadata["calibration_scans"] = copy.deepcopy(
                    calibration_scans
                )

        set_spectrum_axis_mode("pixel")

        keepalive = getattr(fig_, "_view_spectra_keepalive", None)
        if isinstance(keepalive, dict):
            keepalive.update(
                {
                    "energy_calibration": energy_calibration,
                    "incident_energy": incident_energy,
                    "energy_axis": energy_axis,
                    "loss_axis": loss_axis,
                    "energy_axis_button": energy_axis_button_widget,
                    "energy_axis_button_cid": energy_axis_button_cid,
                    "loss_axis_button": loss_axis_button_widget,
                    "loss_axis_button_cid": loss_axis_button_cid,
                }
            )

        status_text.set_text("New calibration model applied")


    def set_spectrum_axis_mode(new_mode: str) -> None:
        """
        Set the visible x-axis mode for the 1D spectrum.
        """

        nonlocal spectrum_axis_mode

        if new_mode == "energy" and not axis_calibration.has_energy_axis:
            return

        if new_mode == "loss" and not axis_calibration.has_loss_axis:
            return

        if spectrum_axis_mode == new_mode:
            spectrum_axis_mode = "pixel"
        else:
            spectrum_axis_mode = new_mode

        update_spectrum_axis_label()
        sync_axis_button_styles()
        refresh_cursor_position()

        status_text.set_text(f"1D x-axis: {current_spectrum_x_label()}")

        # Do not reset zoom.
        # The real shared x-axis stays in pixel coordinates.
        update_projection(draw=True, reset_xlim=False)


    def toggle_energy_axis(_event) -> None:
        """
        Toggle between Pixel and Energy display mode.
        """

        set_spectrum_axis_mode("energy")


    def toggle_loss_axis(_event) -> None:
        """
        Toggle between Pixel and Loss Scale display mode.
        """

        set_spectrum_axis_mode("loss")

    # -------------------------------------------------------------------------
    # Reset and session persistence
    # -------------------------------------------------------------------------

    def reset_spectrum_y_axis(_event) -> None:
        """
        Reset only the y-axis of the 1D spectrum.
        """

        ax[1].set_ylim(*get_spectrum_y_limits())
        status_text.set_text("Reset 1D y-axis")
        fig_.canvas.draw_idle()

    def confirm_viewer_reset() -> bool:
        title = "Reset Viewer"
        message = "Do you really want to reset the Viewer?"
        detail = "All current Viewer settings and plot adjustments will be reset."
        dialog_parent = embedded_container.winfo_toplevel()
        styled_confirmation = getattr(
            dialog_parent,
            "ask_styled_confirmation",
            None,
        )
        if callable(styled_confirmation):
            return bool(styled_confirmation(
                title=title,
                message=message,
                detail=detail,
                destructive=True,
            ))
        return bool(messagebox.askyesno(
            title,
            f"{message}\n\n{detail}",
            parent=dialog_parent,
        ))

    def reset_viewer(_event) -> None:
        nonlocal products
        nonlocal local_controls_are_syncing

        if not confirm_viewer_reset():
            return

        # Step controls trigger callbacks, so keep them quiet during the reset.
        local_controls_are_syncing = True

        try:
            settings.tilt = reset_defaults["tilt"]
            settings.lower_percentile = reset_defaults["lower_percentile"]
            settings.upper_percentile = reset_defaults["upper_percentile"]
            settings.bottom_cut = reset_defaults["bottom_cut"]
            settings.top_cut = reset_defaults["top_cut"]

            settings.display_bin_x = reset_defaults["display_bin_x"]
            settings.display_bin_y = reset_defaults["display_bin_y"]
            settings.spectrum_bin = reset_defaults["spectrum_bin"]

            settings.symmetric_fill_enabled = reset_defaults["symmetric_fill_enabled"]

            settings.median_filter_enabled = reset_defaults["median_filter_enabled"]
            settings.median_filter_window = reset_defaults["median_filter_window"]

            settings.local_filter_enabled = reset_defaults["local_filter_enabled"]
            settings.local_filter_window = reset_defaults["local_filter_window"]
            settings.local_filter_bottom_limit = reset_defaults["local_filter_bottom_limit"]
            settings.local_filter_upper_limit = reset_defaults["local_filter_upper_limit"]

            mode.equal_cut_rows_enabled = reset_defaults["equal_cut_rows_enabled"]
            mode.manual_bottom_cut = reset_defaults["bottom_cut"]
            mode.manual_top_cut = reset_defaults["top_cut"]
            mode.filter_statistics_enabled = False
            clear_filter_statistics_status()
            mode.reference_line_visible = True
            sync_reference_line_visibility(draw=False)
            mode.zoom_to_cut_enabled = reset_defaults["zoom_to_cut_enabled"]

            set_slider_val_silent(m_slider, settings.tilt)
            set_slider_val_silent(line_slider, reset_defaults["line"])

            set_slider_val_silent(lower_slider, settings.lower_percentile)
            set_slider_val_silent(upper_slider, settings.upper_percentile)

            set_slider_val_silent(bottom_slider, settings.bottom_cut)
            set_slider_val_silent(
                top_slider,
                top_slider_value_from_cut(settings.top_cut),
            )
            set_slider_val_silent(c_slider, reset_defaults["c"])

            set_slider_val_silent(hist_bin_slider_x, settings.display_bin_x)
            set_slider_val_silent(hist_bin_slider_y, settings.display_bin_y)
            set_slider_val_silent(bin_slider, settings.spectrum_bin)

            set_slider_val_silent(median_window_slider, settings.median_filter_window)

            set_slider_val_silent(vmin_slider, reset_defaults["vmin"])
            set_slider_val_silent(vmax_slider, reset_defaults["vmax"])
            set_slider_val_silent(
                cmap_start_slider,
                reset_defaults["cmap_start"],
            )
            set_slider_val_silent(
                cmap_end_slider,
                reset_defaults["cmap_end"],
            )

            local_window_control["set_index"](
                local_window_values.index(settings.local_filter_window)
            )

            local_min_perc_control["set_index"](
                local_min_perc_values.index(settings.local_filter_bottom_limit)
            )

            local_max_perc_control["set_index"](
                local_max_perc_values.index(settings.local_filter_upper_limit)
            )

        finally:
            local_controls_are_syncing = False

        sync_toggle_controls()

        sync_equal_cut_control_visibility()
        sync_equal_cut_slider_limits()
        sync_filter_statistics_button_visibility(draw=False)

        pipeline.clear_cache()
        recompute_all(draw=False, reset_xlim=True)
        update_projection(draw=False, reset_xlim=True, reset_ylim=True)

        line.set_ydata([line_slider.val, line_slider.val])

        ax[0].set_xlim(products.x_edges[0], products.x_edges[-1])
        ax[0].set_ylim(products.y_edges[0], products.y_edges[-1])

        update_spectrum_axis_label()
        update_colormap(cmap_radio.get())

        status_text.set_text("Reset")
        fig_.canvas.draw_idle()

    viewer_sections = {
        "display": display_section,
        "alignment": alignment_section,
        "filters": filter_section,
        "spectrum": spectrum_section,
        "output": action_section,
        "colormap_range": cmap_range_section,
        "symmetric_fill": symmetric_fill_section,
        "global_filter": global_filter_section,
        "local_filter": local_filter_section,
        "median_filter": median_filter_section,
    }

    def refresh_viewer_section_layout() -> None:
        """Reapply collapsed states after a global CustomTkinter scaling change."""

        for section in viewer_sections.values():
            section.set_expanded(section.is_expanded())
        modern_toolbar.after_idle(refresh_cursor_toolbar_layout)

    def export_viewer_session_state() -> dict:
        return {
            "version": 1,
            "settings": asdict(settings),
            "mode": asdict(mode),
            "controls": {
                "reference_line": float(line_slider.val),
                "symmetric_margin": float(c_slider.val),
                "display_vmin": float(vmin_slider.val),
                "display_vmax": float(vmax_slider.val),
                "colormap": str(cmap_radio.get()),
                "colormap_options": list(cmap_options),
                "colormap_start": float(cmap_start_slider.val),
                "colormap_end": float(cmap_end_slider.val),
                "tilt_speedup": int(tilt_speedup_control.get()),
                "spectrum_axis_mode": str(spectrum_axis_mode),
                "include_metadata": bool(include_metadata_var.get()),
                "choose_save_path": bool(choose_save_path_var.get()),
            },
            "sections": {
                name: section.is_expanded()
                for name, section in viewer_sections.items()
            },
        }

    def restore_viewer_session_state(state: dict) -> None:
        nonlocal local_controls_are_syncing
        nonlocal spectrum_axis_mode

        if not isinstance(state, dict):
            return

        settings_state = state.get("settings", {})
        if isinstance(settings_state, dict):
            for field_name in settings.__dataclass_fields__:
                if field_name in settings_state:
                    setattr(settings, field_name, settings_state[field_name])

        mode_state = state.get("mode", {})
        if isinstance(mode_state, dict):
            for field_name in mode.__dataclass_fields__:
                if field_name in mode_state:
                    setattr(mode, field_name, mode_state[field_name])

        controls_state = state.get("controls", {})
        if not isinstance(controls_state, dict):
            controls_state = {}

        local_controls_are_syncing = True
        try:
            set_slider_val_silent(m_slider, settings.tilt)
            set_slider_val_silent(lower_slider, settings.lower_percentile)
            set_slider_val_silent(upper_slider, settings.upper_percentile)
            set_slider_val_silent(bottom_slider, settings.bottom_cut)
            set_slider_val_silent(
                top_slider,
                top_slider_value_from_cut(settings.top_cut),
            )
            set_slider_val_silent(bin_slider, settings.spectrum_bin)
            set_slider_val_silent(hist_bin_slider_x, settings.display_bin_x)
            set_slider_val_silent(hist_bin_slider_y, settings.display_bin_y)
            set_slider_val_silent(
                median_window_slider,
                settings.median_filter_window,
            )

            def set_nearest_step(control: ModernStepControl, value: float) -> None:
                values = np.asarray(control["values"], dtype=float)
                index = int(np.argmin(np.abs(values - float(value))))
                control["set_index"](index)

            set_nearest_step(local_window_control, settings.local_filter_window)
            set_nearest_step(
                local_min_perc_control,
                settings.local_filter_bottom_limit,
            )
            set_nearest_step(
                local_max_perc_control,
                settings.local_filter_upper_limit,
            )

            set_slider_val_silent(
                line_slider,
                controls_state.get("reference_line", line_slider.val),
            )
            set_slider_val_silent(
                c_slider,
                controls_state.get("symmetric_margin", c_slider.val),
            )
            set_slider_val_silent(
                vmin_slider,
                controls_state.get("display_vmin", vmin_slider.val),
            )
            set_slider_val_silent(
                vmax_slider,
                controls_state.get("display_vmax", vmax_slider.val),
            )
            set_slider_val_silent(
                cmap_start_slider,
                controls_state.get("colormap_start", 0.0),
            )
            set_slider_val_silent(
                cmap_end_slider,
                controls_state.get("colormap_end", 1.0),
            )
        finally:
            local_controls_are_syncing = False

        saved_colormaps = controls_state.get("colormap_options", [])
        if isinstance(saved_colormaps, list):
            for colormap_name in saved_colormaps:
                if (
                    isinstance(colormap_name, str)
                    and colormap_name in plt.colormaps()
                    and colormap_name not in cmap_options
                ):
                    cmap_options.append(colormap_name)
        cmap_radio.configure(values=list(cmap_options))

        selected_colormap = str(controls_state.get("colormap", cmap_radio.get()))
        if selected_colormap not in plt.colormaps():
            selected_colormap = "gnuplot"
        if selected_colormap not in cmap_options:
            cmap_options.append(selected_colormap)
            cmap_radio.configure(values=list(cmap_options))
        cmap_radio.set(selected_colormap)

        requested_speedup = int(controls_state.get("tilt_speedup", 4))
        tilt_speedup_control.set(str(int(np.clip(requested_speedup, 1, 10))))

        requested_axis_mode = str(
            controls_state.get("spectrum_axis_mode", "pixel")
        )
        include_metadata_var.set(
            bool(controls_state.get("include_metadata", False))
        )
        choose_save_path_var.set(
            bool(controls_state.get("choose_save_path", False))
        )
        if requested_axis_mode == "energy" and not axis_calibration.has_energy_axis:
            requested_axis_mode = "pixel"
        if requested_axis_mode == "loss" and not axis_calibration.has_loss_axis:
            requested_axis_mode = "pixel"
        if requested_axis_mode not in ("pixel", "energy", "loss"):
            requested_axis_mode = "pixel"
        spectrum_axis_mode = requested_axis_mode

        sync_toggle_controls()

        line.set_ydata([line_slider.val, line_slider.val])
        sync_reference_line_visibility(draw=False)
        sync_equal_cut_control_visibility()
        sync_equal_cut_slider_limits()
        sync_filter_statistics_button_visibility(draw=False)

        pipeline.clear_cache()
        recompute_all(draw=False, reset_xlim=True)
        update_projection(draw=False, reset_ylim=True)
        ax[0].set_xlim(products.x_edges[0], products.x_edges[-1])
        ax[0].set_ylim(products.y_edges[0], products.y_edges[-1])
        update_colormap(selected_colormap)
        update_spectrum_axis_label()
        sync_axis_button_styles()

        sections_state = state.get("sections", {})
        if isinstance(sections_state, dict):
            for name, expanded_state in sections_state.items():
                section = viewer_sections.get(str(name))
                if section is not None:
                    section.set_expanded(bool(expanded_state))

        status_text.set_text("Session state restored")
        fig_.canvas.draw_idle()

    cursor_motion_cid = fig_.canvas.mpl_connect(
        "motion_notify_event",
        update_cursor_position,
    )
    cursor_leave_cid = fig_.canvas.mpl_connect(
        "figure_leave_event",
        clear_cursor_position,
    )

    slider_callback_ids = [
        vmin_slider.on_changed(update_vmin),
        vmax_slider.on_changed(update_vmax),
        cmap_start_slider.on_changed(update_colormap_start),
        cmap_end_slider.on_changed(update_colormap_end),
        m_slider.on_changed(update_tilt),
        line_slider.on_changed(update_line),
        lower_slider.on_changed(update_filter),
        upper_slider.on_changed(update_filter),
        bottom_slider.on_changed(update_cut),
        top_slider.on_changed(update_cut),
        c_slider.on_changed(update_cut_margin),
        bin_slider.on_changed(update_bin),
        median_window_slider.on_changed(update_median_window),
        hist_bin_slider_x.on_changed(update_filter),
        hist_bin_slider_y.on_changed(update_filter),
    ]

    equal_cut_rows_button_cid = equal_cut_rows_button_widget.on_clicked(
        toggle_equal_cut_rows
    )
    zoom_to_cut_button_cid = zoom_to_cut_button_widget.on_clicked(toggle_zoom_to_cut)
    calculate_m_button_cid = calculate_m_button_widget.on_clicked(calculate_tilt_)
    calculate_ridge_button_cid = calculate_ridge_button_widget.on_clicked(
        calculate_ridge_
    )
    fill_symmetric_button_cid = fill_symmetric_button_widget.on_clicked(
        toggle_fill_symmetric
    )
    median_filter_button_cid = median_filter_button_widget.on_clicked(
        toggle_median_filter
    )
    local_filter_button_cid = local_filter_button_widget.on_clicked(
        toggle_local_filter
    )
    cmap_radio_cid = cmap_radio.on_clicked(update_colormap)
    add_colormap_button_cid = add_colormap_button_widget.on_clicked(
        open_colormap_dialog
    )
    save_spectrum_button_cid = save_spectrum_button_widget.on_clicked(
        save_current_spectrum
    )
    save_histogram_button_cid = save_histogram_button_widget.on_clicked(
        save_current_histogram
    )
    reset_button_cid = reset_button_widget.on_clicked(reset_viewer)
    filter_statistics_button_cid = filter_statistics_button_widget.on_clicked(
        toggle_filter_statistics
    )
    hide_ref_line_checkbox_widget.configure(
        command=toggle_reference_line_visibility,
    )
    reset_spectrum_y_button_cid = reset_spectrum_y_button_widget.on_clicked(
        reset_spectrum_y_axis
    )

    energy_axis_button_cid = None
    loss_axis_button_cid = None

    if energy_axis_button_widget is not None:
        energy_axis_button_cid = energy_axis_button_widget.on_clicked(
            toggle_energy_axis
        )

    if loss_axis_button_widget is not None:
        loss_axis_button_cid = loss_axis_button_widget.on_clicked(toggle_loss_axis)

    setattr(
        fig_,
        "_view_spectra_keepalive",
        {
            "settings": settings,
            "mode": mode,
            "pipeline": pipeline,
            "products": lambda: products,
            "sliders": sliders,
            "small_sliders": small_sliders,
            "slider_callback_ids": slider_callback_ids,
            "step_buttons": step_buttons,
            "cursor_position_label": cursor_position_label,
            "refresh_cursor_toolbar_layout": refresh_cursor_toolbar_layout,
            "histogram_count_at": histogram_count_at,
            "update_cursor_position": update_cursor_position,
            "cursor_motion_cid": cursor_motion_cid,
            "cursor_leave_cid": cursor_leave_cid,
            "save_spectrum_button": save_spectrum_button_widget,
            "save_spectrum_button_cid": save_spectrum_button_cid,
            "save_current_spectrum": save_current_spectrum,
            "save_histogram_button": save_histogram_button_widget,
            "save_histogram_button_cid": save_histogram_button_cid,
            "save_current_histogram": save_current_histogram,
            "set_save_directories": set_save_directories,
            "status_text": status_text,
            "equal_cut_rows_button": equal_cut_rows_button_widget,
            "equal_cut_rows_button_cid": equal_cut_rows_button_cid,
            "zoom_to_cut_button": zoom_to_cut_button_widget,
            "zoom_to_cut_button_cid": zoom_to_cut_button_cid,
            "fill_symmetric_button": fill_symmetric_button_widget,
            "fill_symmetric_button_cid": fill_symmetric_button_cid,
            "median_filter_button": median_filter_button_widget,
            "median_filter_button_cid": median_filter_button_cid,
            "median_window_slider": median_window_slider,
            "local_filter_button": local_filter_button_widget,
            "local_filter_button_cid": local_filter_button_cid,
            "calculate_m_button": calculate_m_button_widget,
            "calculate_m_button_cid": calculate_m_button_cid,
            "calculate_ridge_button": calculate_ridge_button_widget,
            "calculate_ridge_button_cid": calculate_ridge_button_cid,
            "tilt_speedup_control": tilt_speedup_control,
            "cmap_radio": cmap_radio,
            "cmap_radio_cid": cmap_radio_cid,
            "cmap_start_slider": cmap_start_slider,
            "cmap_end_slider": cmap_end_slider,
            "add_colormap_button": add_colormap_button_widget,
            "add_colormap_button_cid": add_colormap_button_cid,
            "colormap_dialog": lambda: colormap_dialog,

            "local_window_control": local_window_control,
            "local_min_perc_control": local_min_perc_control,
            "local_max_perc_control": local_max_perc_control,
            "reset_button": reset_button_widget,
            "reset_button_cid": reset_button_cid,
            "reset_spectrum_y_button": reset_spectrum_y_button_widget,
            "reset_spectrum_y_button_cid": reset_spectrum_y_button_cid,
            "filter_statistics_button": filter_statistics_button_widget,
            "filter_statistics_button_cid": filter_statistics_button_cid,
            "hide_ref_line_checkbox": hide_ref_line_checkbox_widget,
            "hide_ref_line_var": hide_ref_line_var,
            "include_metadata_checkbox": include_metadata_checkbox,
            "include_metadata_var": include_metadata_var,
            "choose_save_path_checkbox": choose_save_path_checkbox,
            "choose_save_path_var": choose_save_path_var,

            "energy_axis_button": energy_axis_button_widget,
            "energy_axis_button_cid": energy_axis_button_cid,
            "energy_calibration": energy_calibration,
            "get_calibration_state": get_calibration_state,
            "apply_calibration_model": apply_calibration_model,
            "loss_axis_button": loss_axis_button_widget,
            "loss_axis_button_cid": loss_axis_button_cid,
            "incident_energy": incident_energy,
            "energy_axis": energy_axis,
            "loss_axis": loss_axis,
            "tk_parent": tk_parent,
            "embedded_container": embedded_container,
            "embedded_toolbar_frame": embedded_toolbar_frame,
            "embedded_canvas": embedded_canvas,
            "embedded_widget": embedded_widget,
            "embedded_toolbar": embedded_toolbar,
            "modern_toolbar": modern_toolbar,
            "controls_sidebar": controls_sidebar,
            "viewer_splitter": viewer_splitter,
            "viewer_subtitle_label": viewer_subtitle_label,
            "set_viewer_title": set_viewer_title,
            "refresh_section_layout": refresh_viewer_section_layout,
            "export_session_state": export_viewer_session_state,
            "restore_session_state": restore_viewer_session_state,
            "refresh_axes_after_show": refresh_axes_after_show,
        },
    )

    sync_axis_button_styles()
    update_spectrum_axis_label()
    sync_equal_cut_control_visibility()
    sync_equal_cut_slider_limits()
    recompute_all(draw=False, reset_xlim=False)
    update_projection(draw=False, reset_ylim=True)

    if session_state is not None:
        restore_viewer_session_state(session_state)

    embedded_canvas.draw()

    if standalone_root is not None:
        standalone_root.mainloop()

    return fig_, ax


if __name__ == "__main__":
    print(
        "Import this file and call view_spectra(scans, scans_dir=...).\n"
        "Example:\n"
        "    from mev_viewer import view_spectra\n"
        "    view_spectra(np.array([1234, 1235]), scans_dir=Path('/path/to/scans'))"
    )
