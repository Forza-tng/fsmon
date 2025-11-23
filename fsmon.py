#!/usr/bin/env python3

# SPDX-License-Identifier: GPL-3.0-or-later

"""
fsmon: A Real-Time Btrfs I/O Monitor
------------------------------------

`fsmon` monitors the I/O activity of Btrfs filesystems in real time,
displaying bandwidth and IOPS statistics for detected filesystems
and their devices.

It combines I/O statistics from all member devices of each Btrfs
filesystem, providing a unified view of the filesystem's overall
activity.

Features:
---------
- Real-time monitoring of read/write bandwidth and IOPS.
- Visual charts for I/O statistics.
- Dynamic terminal size handling.
- Lightweight and efficient, leveraging sysfs for minimal overhead.

Requirements:
-------------
- Python 3.6 or higher.
- Btrfs filesystems mounted on the system.
- Sufficient permissions to access `/sys/fs/btrfs` and related
  devices in `/sys/block`.

Usage:
------
 `-h` or `--help`:    Display usage information.
 `-v` or `--version`: Display version.

License:
--------
This program is licensed under the GNU General Public License v3.0
or later.
"""

__description__ = "A real-time Btrfs I/O monitor for tracking filesystem activity."
__author__      = "Forza <forza@tnonline.net>"
__license__     = "GPL-3.0-or-later"
__version__     = "0.2.0"

import argparse
import curses
import os
import sys
import time

from collections import defaultdict, deque
from collections.abc import Sequence
from math import ceil, floor
from typing import Any, Optional


####### Configuration Defaults ########

# Colours.
USE_TERM_COLORS        = True  # Use terminal's default background colour
COLOR_CHART_BW_READ    = 3     # Green
COLOR_CHART_BW_WRITE   = 2     # Red
COLOR_CHART_IOPS_READ  = 7     # Cyan
COLOR_CHART_IOPS_WRITE = 6     # Magenta
COLOR_SELECTED         = 7     # Cyan
COLOR_COL_HEADER       = 4     # Yellow
COLOR_HEADER           = 8     # Terminal's default colour
COLOR_FOOTER           = 8     # Terminal's default colour
COLOR_HLINE            = 4     # Yellow
COLOR_POPUP_BORDER     = 4     # Yellow
COLOR_POPUP_TEXT       = 8     # Default FG

# Column widths.
COL_LABEL    = 12
COL_READ_BW  = 11
COL_WRITE_BW = 11
COL_IOPS     = 13

# Minimum allowed terminal width.
MIN_WIDTH = COL_LABEL + COL_READ_BW + COL_WRITE_BW + COL_IOPS

# Chart geometry.
DEFAULT_CHART_H   = 11
DEFAULT_CHART_W   = 35
MIN_CHART_H       = 7
MAX_CHART_H       = 25
MIN_CHART_W       = 20
Y_LABEL_PAD       = 11
COL_GAP           = 2

# UI refresh rate.
UI_FRAME_HZ       = 60
UI_TICK_S         = 1.0 / UI_FRAME_HZ

# Sampling defaults.
DEFAULT_INTERVAL_S   = 1.0
DEFAULT_HISTORY_STR  = "3600s"
DEFAULT_MIN_POINTS   = 300

# Allowed sampling interval.
MIN_INTERVAL_S       = 0.1
MAX_INTERVAL_S       = 60.0

# Btrfs labels and member devices are read from sysfs.
SYSFS_PATH = "/sys/fs/btrfs"

######## Configuration End ########


def init_colors() -> None:
    """Initialise curses colour pairs."""
    curses.start_color()

    if USE_TERM_COLORS:
        # Use terminal default background instead of forcing black.
        curses.use_default_colors()
        default_bg = -1
    else:
        default_bg = curses.COLOR_BLACK

    # Map logical colours to foreground colours on default_bg.
    curses.init_pair(1, curses.COLOR_BLACK,   default_bg)
    curses.init_pair(2, curses.COLOR_RED,     default_bg)
    curses.init_pair(3, curses.COLOR_GREEN,   default_bg)
    curses.init_pair(4, curses.COLOR_YELLOW,  default_bg)
    curses.init_pair(5, curses.COLOR_BLUE,    default_bg)
    curses.init_pair(6, curses.COLOR_MAGENTA, default_bg)
    curses.init_pair(7, curses.COLOR_CYAN,    default_bg)
    curses.init_pair(8, curses.COLOR_WHITE,   default_bg)


def parse_duration(s: str) -> float:
    """
    Parse a duration string like '300s', '5m', '1h' into seconds.

    Accepts bare numbers (seconds) and ms/s/m/h suffixes.
    """
    raw = s
    s = s.strip().lower()
    try:
        if s.endswith("ms"):
            return float(s[:-2]) / 1000.0
        if s.endswith("s"):
            return float(s[:-1])
        if s.endswith("m"):
            return float(s[:-1]) * 60.0
        if s.endswith("h"):
            return float(s[:-1]) * 3600.0
        return float(s)
    except ValueError:
        raise ValueError(
            f"Invalid history value '{raw}'. Use a number optionally followed by ms, s, m, or h."
        )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for fsmon."""
    parser = argparse.ArgumentParser(
        description=__description__,
        formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(
            prog, max_help_position=35
        ),
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"fsmon {__version__}",
        help="Show program version and exit.",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_S,
        help="Sampling interval in seconds.",
    )
    parser.add_argument(
        "-l",
        "--history",
        type=str,
        default=DEFAULT_HISTORY_STR,
        help="History length (e.g. '300s', '5m', '1h').",
    )
    return parser.parse_args()


def get_btrfs_filesystems() -> tuple[dict[str, list[str]], dict[str, str]]:
    """
    Discover Btrfs filesystems and their member devices.

    Returns:
        (btrfs_fs, labels) where:
        - btrfs_fs: uuid -> list of device stat paths
        - labels:   uuid -> human-readable filesystem label (or UUID)
    """
    btrfs_fs: dict[str, list[str]] = {}
    labels: dict[str, str] = {}

    # Enumerate all Btrfs filesystem UUIDs under /sys/fs/btrfs.
    try:
        uuids = os.listdir(SYSFS_PATH)
    except FileNotFoundError:
        print(f"Error: '{SYSFS_PATH}' does not exist. Ensure sysfs is mounted.", file=sys.stderr)
        sys.exit(1)
    except PermissionError:
        print(f"Error: Permission denied when accessing '{SYSFS_PATH}'", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Error: Unable to access '{SYSFS_PATH}': {e}", file=sys.stderr)
        sys.exit(1)

    for uuid in uuids:
        devices_path = os.path.join(SYSFS_PATH, uuid, "devices")
        label_path   = os.path.join(SYSFS_PATH, uuid, "label")

        # Skip entries that do not look like proper Btrfs filesystem directories.
        if not (os.path.isdir(devices_path) and os.path.exists(label_path)):
            continue

        # Resolve per-device stat file paths for this filesystem.
        try:
            devices = [
                os.path.join(devices_path, d, "stat")
                for d in os.listdir(devices_path)
            ]
            btrfs_fs[uuid] = devices
        except PermissionError:
            print(f"Error: Permission denied when accessing '{devices_path}'", file=sys.stderr)
            continue
        except OSError as e:
            print(f"Error: Unable to read devices from '{devices_path}': {e}", file=sys.stderr)
            continue

        # Read the user-visible label for this filesystem.
        try:
            with open(label_path, "r") as f:
                label = f.read().strip().encode("utf-8").decode("utf-8", errors="replace")
        except PermissionError:
            print(f"Error: Permission denied when reading '{label_path}'", file=sys.stderr)
            continue
        except OSError as e:
            print(f"Error: Unable to read '{label_path}': {e}", file=sys.stderr)
            continue

        # Use UUID as a fallback label if no label is set or it's empty.
        labels[uuid] = label if label else uuid

    if not btrfs_fs:
        print(f"Error: No Btrfs filesystems found in '{SYSFS_PATH}'", file=sys.stderr)
        sys.exit(1)

    return btrfs_fs, labels

def get_device_stats(btrfs_fs: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    """
    Aggregate I/O statistics for each Btrfs filesystem.

    Values are cumulative byte and io counters per filesystem summed
    over all member devices.
    """
    fs_stats: dict[str, dict[str, int]] = {}
    sector_size = 512
    # Linux device/stat files always use 512-byte sectors.
    # Reference: https://www.kernel.org/doc/Documentation/block/stat.txt

    # Iterate through each filesystem UUID and its associated devices.
    for uuid, devices in btrfs_fs.items():
        read_bytes  = 0
        write_bytes = 0
        read_ops    = 0
        write_ops   = 0

        # Read each device stat file and accumulate counters.
        for dev_stat_path in devices:
            try:
                with open(dev_stat_path, "r") as f:
                    stats = f.read().split()
                if len(stats) < 7:
                    print(f"Warning: Stat file '{dev_stat_path}' is malformed or incomplete.", file=sys.stderr)
                    continue

                # Stats fields are cumulative per device.
                read_ops     += int(stats[0])  # Reads completed
                write_ops    += int(stats[4])  # Writes completed
                read_sectors  = int(stats[2])  # Sectors read
                write_sectors = int(stats[6])  # Sectors written
                read_bytes   += read_sectors * sector_size
                write_bytes  += write_sectors * sector_size
            except FileNotFoundError:
                print(f"Warning: Stat file not found for '{dev_stat_path}'", file=sys.stderr)
                continue
            except PermissionError:
                print(f"Error: Permission denied when reading '{dev_stat_path}'", file=sys.stderr)
                continue
            except ValueError:
                print(f"Warning: Invalid data in stat file '{dev_stat_path}'", file=sys.stderr)
                continue
            except OSError as e:
                # Unexpected I/O error. Bailing out.
                print(f"Error: Unable to read '{dev_stat_path}': {e}", file=sys.stderr)
                sys.exit(1)

        fs_stats[uuid] = {
            "read_bytes":  read_bytes,
            "write_bytes": write_bytes,
            "read_ops":    read_ops,
            "write_ops":   write_ops,
        }

    return fs_stats


def format_iec(value: float) -> str:
    """
    Format a byte count using IEC units (B, KiB, MiB, GiB, TiB, PiB).

    Rolls to the next unit before 1000 to keep the numeric part ≤ 4 glyphs.
    """
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    limit = 1000.0
    v = float(value)

    for unit in units[:-1]:
        if v < limit:
            return f"{v:.1f} {unit}"
        v /= 1024.0

    return f"{v:.1f} {units[-1]}"


def format_base10(value: float) -> str:
    """
    Format a value using base-10 units (k, M, G, T, P).

    Rolls to the next unit before 1000 to keep the numeric part ≤ 4 glyphs.
    """
    units = ["", "k", "M", "G", "T", "P"]
    v = float(value)

    for suffix in units:
        if v < 1000.0:
            return f"{v:.1f}{suffix}"
        v /= 1000.0

    return f"{v:.1f}P"


def _window_data(seq: Sequence[float] | Sequence[int], width: int, offset: int) -> list[float]:
    """
    Return a right-aligned window of width samples from seq.

    The window is offset from the newest sample by `offset` steps and
    padded with zeros on the left if there are fewer than `width` samples.
    """
    arr = list(seq)
    end = max(0, len(arr) - int(offset))
    start = max(0, end - int(width))
    window = arr[start:end]
    if len(window) < width:
        window = [0.0] * (width - len(window)) + window
    return [float(v) for v in window]


def _format_ts(ts: float | int) -> str:
    """Format a timestamp into HH:MM:SS using local time."""
    return time.strftime("%H:%M:%S", time.localtime(ts))


def _format_secs_brief(val: float) -> str:
    """
    Format a relative age in seconds.

    Under 10 minutes: integer seconds.
    10 minutes and up: whole minutes with trailing 'm'.
    """
    if val < 600:
        return f"{int(round(val)):d}"
    m = int(round(val / 60.0))
    return f"{m}m"


def display_chart(
    stdscr,
    title: str,
    data_seq: Sequence[float],
    ts_seq: Optional[Sequence[int]],
    row_start: int,
    col_start: int,
    chart_w: int,
    chart_h: int,
    color: int,
    use_iops: bool = False,
    offset: int = 0,
    y_override: Optional[float] = None,
    title_suffix: str = "",
    interval_s: float = 1.0
) -> None:
    """
    Render a single chart at the given position.

    The newest sample is at the left edge; moving right shows older samples.
    """
    # Window the data and associated timestamps, then reverse to draw left→right newest→oldest.
    window = _window_data(data_seq, chart_w, offset)[::-1]
    ts_win = _window_data(ts_seq or [], chart_w, offset)[::-1]

    # Oldest and newest timestamps visible in this chart window.
    left_ts  = next((t for t in ts_win if t), None)
    right_ts = next((t for t in reversed(ts_win) if t), None)

    # Determine Y-axis max (automatic or manual override).
    max_val = max(max(window), 1.0)
    if y_override is not None and y_override > 0.0:
        max_val = max(y_override, 1.0)

    # Title centred above the chart region.
    full_title = title + (f" {title_suffix}" if title_suffix else "")
    title_x = col_start + Y_LABEL_PAD - 2 + (chart_w // 2) - (len(full_title) // 2)
    title_x = max(col_start, title_x)
    stdscr.addstr(row_start, title_x, full_title)

    # Draw Y-axis labels and chart rows.
    y_axis_rows = max(1, chart_h - 2)
    for i in range(y_axis_rows - 1):
        # Threshold for this row (higher row = higher value).
        threshold = max_val * ((y_axis_rows - 1) - i) / (y_axis_rows - 1)
        # Draw bar: "|" when above threshold, "." for background, " " on X-axis row.
        chart_row = "".join(
            "|" if val > threshold else " " if i == (y_axis_rows - 2) else "."
            for val in window
        ).ljust(chart_w - 1)

        # Left-hand numeric Y label.
        y_axis_label = format_base10(threshold) if use_iops else format_iec(threshold)
        stdscr.addstr(row_start + 1 + i, col_start, f"{y_axis_label:>10} ")

        # Plot the row, underlining the last row to emphasise the X-axis.
        if i == (y_axis_rows - 2):
            stdscr.addstr(
                row_start + 1 + i,
                col_start + Y_LABEL_PAD,
                chart_row,
                curses.A_UNDERLINE | curses.color_pair(color),
            )
        else:
            stdscr.addstr(
                row_start + 1 + i,
                col_start + Y_LABEL_PAD,
                chart_row,
                curses.color_pair(color),
            )

    # Draw X-axis ticks: relative age in seconds from the newest sample.
    labels: list[str] = []
    for col in range(0, chart_w, 5):
        secs = (offset + col) * interval_s
        labels.append(f"{_format_secs_brief(secs):>4}")
    x_axis = " ".join(labels)
    stdscr.addstr(
        row_start + y_axis_rows,
        col_start + 8,
        x_axis[: max(0, chart_w + 8)],
    )

    # Draw wall-clock timestamps at the left/right edges.
    if left_ts and right_ts:
        left_label  = _format_ts(left_ts)
        right_label = _format_ts(right_ts)
        stdscr.addstr(
            row_start + y_axis_rows + 1,
            col_start + Y_LABEL_PAD,
            left_label,
        )
        rx = col_start + Y_LABEL_PAD + chart_w - len(right_label) - 1
        if rx > col_start + Y_LABEL_PAD:
            stdscr.addstr(
                row_start + y_axis_rows + 1,
                rx,
                right_label,
            )


def _fit_layout_auto(
    available_w: int,
    available_h: int,
    ncharts: int,
    chart_h: int,
) -> tuple[int, int, int]:
    """
    Compute auto layout for charts.

    Returns (cols, rows, chart_w).
    """
    for cols in range(1, ncharts + 1):  # Prefer stacking
        rows = ceil(ncharts / cols)
        if rows * chart_h > available_h:
            continue

        # Compute chart width in this column layout.
        chart_w = floor((available_w - (cols - 1) * COL_GAP) / cols) - Y_LABEL_PAD
        if chart_w >= MIN_CHART_W:
            return cols, rows, chart_w
    return 0, 0, 0


def _fit_layout_custom(
    available_w: int,
    available_h: int,
    ncharts: int,
    chart_h: int,
    custom_chart_w: int,
) -> tuple[int, int, int]:
    """
    Compute layout for charts with a fixed custom width.

    Returns (cols, rows, chart_w) or (0, 0, 0) if the layout does not fit.
    """
    per_col = custom_chart_w + Y_LABEL_PAD
    max_cols = max(1, (available_w + COL_GAP) // (per_col + COL_GAP))
    max_cols = int(min(max_cols, ncharts))

    for cols in range(1, max_cols + 1):  # Prefer stacking
        rows = ceil(ncharts / cols)
        if rows * chart_h > available_h:
            continue
        total_w = cols * (custom_chart_w + Y_LABEL_PAD) + (cols - 1) * COL_GAP
        if total_w <= available_w:
            return cols, rows, custom_chart_w
    return 0, 0, 0


def _mmss(seconds: float) -> str:
    """Format seconds as M:SS."""
    total = int(round(seconds))
    m, s = divmod(total, 60)
    return f"{m}:{s:02d}"


def draw_help_popup(stdscr, lines: Sequence[str]):
    """Draw a centred help popup window and return its window object."""
    h, w = stdscr.getmaxyx()
    pad = 4

    # Compute popup width and height with some padding and bounds.
    inner_width = len(max(lines, key=len)) + 4
    content_w = min(
        max(inner_width, 40),
        max(40, w - 2 * pad),
    )
    content_h = min(
        len(lines) + 4,
        max(12, h - 2 * pad),
    )

    # Centre the popup on screen.
    start_y = max(1, (h - content_h) // 2)
    start_x = max(1, (w - content_w) // 2)

    win = curses.newwin(content_h, content_w, start_y, start_x)

    # Draw border and title.
    border_attr = curses.color_pair(COLOR_POPUP_BORDER)
    win.attron(border_attr)
    win.box()
    win.attroff(border_attr)

    title = " Help "
    try:
        win.addstr(
            0,
            max(1, (content_w - len(title)) // 2),
            title,
            curses.color_pair(COLOR_COL_HEADER),
        )
    except curses.error:
        # Ignore drawing errors for very small terminals.
        pass

    # Draw help text, leaving room for border and title.
    win.attron(curses.color_pair(COLOR_POPUP_TEXT))
    max_lines = content_h - 3
    text_w = max(0, content_w - 4)
    for i, line in enumerate(lines[:max_lines]):
        try:
            win.addstr(2 + i, 2, line[:text_w])
        except curses.error:
            pass
    win.attroff(curses.color_pair(COLOR_POPUP_TEXT))
    win.refresh()
    return win


def display_ui(
    stdscr,
    btrfs_fs: dict[str, list[str]],
    fs_labels: dict[str, str],
    sample_interval_s: float,
    history_seconds: float,
    min_points: int,
) -> None:
    """
    Main curses UI loop.

    Handles input, sampling, aggregation, and rendering of the summary
    table and charts until the user quits.
    """
    # Configure curses input/keyboard behaviour and colours.
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)
    init_colors()

    # Selection and display mode state.
    selected_idx = 0           # index into uuids list
    use_labels = True          # show labels instead of UUIDs
    show_iops_charts = False   # toggle between BW-only and BW+IOPS charts

    # Width mode state ("auto" | "custom").
    width_mode = "auto"
    custom_chart_w = DEFAULT_CHART_W

    # Chart height (dynamic).
    chart_h = DEFAULT_CHART_H

    # History scroll (0 = live; >0 scrolls back in time).
    scroll_offset = 0

    # Pause (screen updates).
    paused = False

    # Help popup state.
    show_help = False
    help_drawn = False
    help_win: Any = None

    # Most recent raw stats per filesystem; used for delta computation.
    prev_stats: dict[str, dict[str, int]] = {}
    # Skip the first delta per filesystem to avoid a huge first jump.
    first_skipped: set[str] = set()

    # History buffers per filesystem. The actual number of points is
    # determined by history_seconds and sampling interval.
    history_len = max(min_points, int(history_seconds / sample_interval_s) + 1)
    history: dict[str, dict[str, deque[float]]] = defaultdict(
        lambda: {
            "read_bw":    deque(maxlen=history_len),
            "write_bw":   deque(maxlen=history_len),
            "read_iops":  deque(maxlen=history_len),
            "write_iops": deque(maxlen=history_len),
        }
    )
    # Timeline of sampling instants.
    timeline: deque[int] = deque(maxlen=history_len)

    # Y-scale mode & manual max for BW and IOPS.
    y_mode_bw, y_manual_bw = "auto", 0.0
    y_mode_iops, y_manual_iops = "auto", 0.0

    def active_group() -> str:
        """Return currently active chart group ('bw' or 'iops')."""
        return "iops" if show_iops_charts else "bw"

    def current_group_window_max(
        fs_uuid: str,
        chart_w_local: int,
        offset_local: int,
        group: str,
    ) -> float:
        """Return the maximum value over the visible window for a group."""
        if group == "bw":
            rw = _window_data(history[fs_uuid]["read_bw"],  chart_w_local, offset_local)
            ww = _window_data(history[fs_uuid]["write_bw"], chart_w_local, offset_local)
            return float(max([0.0] + rw + ww))
        ri = _window_data(history[fs_uuid]["read_iops"],  chart_w_local, offset_local)
        wi = _window_data(history[fs_uuid]["write_iops"], chart_w_local, offset_local)
        return float(max([0.0] + ri + wi))

    def ensure_manual_mode(
        group: str,
        fs_uuid: str,
        chart_w_local: int,
        offset_local: int,
    ) -> None:
        """Switch Y-scale for a group to manual if it is currently auto."""
        nonlocal y_mode_bw, y_manual_bw, y_mode_iops, y_manual_iops
        if group == "bw":
            if y_mode_bw == "auto":
                base = current_group_window_max(fs_uuid, chart_w_local, offset_local, "bw") or 1.0
                y_manual_bw = float(base)
                y_mode_bw = "manual"
        else:
            if y_mode_iops == "auto":
                base = current_group_window_max(fs_uuid, chart_w_local, offset_local, "iops") or 1.0
                y_manual_iops = float(base)
                y_mode_iops = "manual"

    def bump_manual(group: str, factor: float) -> None:
        """Scale manual Y-max for the given group by factor."""
        nonlocal y_manual_bw, y_manual_iops
        if group == "bw":
            y_manual_bw = max(1.0, y_manual_bw * factor)
        else:
            y_manual_iops = max(1.0, y_manual_iops * factor)

    last_sample_time = 0.0
    # Per-interval deltas used for both table and charts.
    last_deltas: dict[str, dict[str, float]] = {}

    try:
        while True:
            ########
            # Input handling
            ########
            while True:
                key = stdscr.getch()
                if key == -1:
                    break

                if key == curses.KEY_UP:
                    selected_idx = max(0, selected_idx - 1)
                elif key == curses.KEY_DOWN:
                    selected_idx += 1
                elif key in (ord("q"), ord("Q")):
                    return
                elif key in (ord("l"), ord("L")):
                    # Toggle between label and UUID display.
                    use_labels = not use_labels
                elif key in (ord("i"), ord("I")):
                    # Toggle additional IOPS charts for the selected filesystem.
                    show_iops_charts = not show_iops_charts
                    scroll_offset = 0
                elif key in (ord("w"), ord("W")):
                    # Toggle auto ↔ custom chart width.
                    width_mode = "custom" if width_mode == "auto" else "auto"
                elif key == curses.KEY_LEFT:
                    # Scroll backward in history.
                    scroll_offset += 5
                elif key == curses.KEY_RIGHT:
                    # Scroll forward towards live.
                    scroll_offset = max(0, scroll_offset - 5)
                elif key in (ord("0"),):
                    # Jump back to live view.
                    scroll_offset = 0
                elif key in (ord("["),):
                    # Decrease custom chart width.
                    width_mode = "custom"
                    custom_chart_w = max(MIN_CHART_W, custom_chart_w - 5)
                elif key in (ord("]"),):
                    # Increase custom chart width.
                    width_mode = "custom"
                    custom_chart_w = custom_chart_w + 5
                elif key in (ord("{"),):
                    # Decrease chart height.
                    chart_h = max(MIN_CHART_H, chart_h - 1)
                elif key in (ord("}"),):
                    # Increase chart height.
                    chart_h = min(MAX_CHART_H, chart_h + 1)
                elif key in (ord("p"), ord("P")):
                    # Pause screen updates (sampling continues).
                    paused = not paused
                elif key in (ord("y"),):  # zoom in (smaller max)
                    g = active_group()
                    fs_uuid = list(btrfs_fs.keys())[selected_idx] if btrfs_fs else None
                    if fs_uuid:
                        est_w = custom_chart_w if width_mode == "custom" else DEFAULT_CHART_W
                        ensure_manual_mode(g, fs_uuid, est_w, scroll_offset)
                        bump_manual(g, 0.90)
                elif key in (ord("Y"),):  # zoom out (larger max)
                    g = active_group()
                    fs_uuid = list(btrfs_fs.keys())[selected_idx] if btrfs_fs else None
                    if fs_uuid:
                        est_w = custom_chart_w if width_mode == "custom" else DEFAULT_CHART_W
                        ensure_manual_mode(g, fs_uuid, est_w, scroll_offset)
                        bump_manual(g, 1.10)
                elif key in (ord("r"), ord("R")):
                    # Reset Y-scale for the active group back to auto.
                    if active_group() == "bw":
                        y_mode_bw, y_manual_bw = "auto", 0.0
                    else:
                        y_mode_iops, y_manual_iops = "auto", 0.0
                elif key in (ord("?"), ord("h")):
                    # Toggle help overlay.
                    show_help = not show_help
                    help_drawn = False

            ########
            # Sampling
            ########
            now = time.time()
            if now - last_sample_time >= sample_interval_s:
                current_stats = get_device_stats(btrfs_fs)
                deltas: dict[str, dict[str, float]] = {}

                for uuid, stats in current_stats.items():
                    prev = prev_stats.get(
                        uuid,
                        {
                            "read_bytes":  0,
                            "write_bytes": 0,
                            "read_ops":    0,
                            "write_ops":   0,
                        },
                    )
                    # Skip first delta per filesystem to avoid a spike.
                    if uuid not in first_skipped:
                        first_skipped.add(uuid)
                        prev_stats[uuid] = stats
                        continue

                    inv = 1.0 / sample_interval_s
                    deltas[uuid] = {
                        "read_bw":    (stats["read_bytes"]  - prev["read_bytes"])  * inv,
                        "write_bw":   (stats["write_bytes"] - prev["write_bytes"]) * inv,
                        "read_iops":  (stats["read_ops"]    - prev["read_ops"])    * inv,
                        "write_iops": (stats["write_ops"]   - prev["write_ops"])   * inv,
                    }

                prev_stats = current_stats
                if deltas:
                    # Append a single timestamp per sampling step.
                    timeline.append(int(now))
                    # Append per-filesystem deltas into the history buffers.
                    for uuid, d in deltas.items():
                        h = history[uuid]
                        h["read_bw"].append(d["read_bw"])
                        h["write_bw"].append(d["write_bw"])
                        h["read_iops"].append(d["read_iops"])
                        h["write_iops"].append(d["write_iops"])
                    last_deltas = deltas

                last_sample_time = now

            ########
            # Clamp selection and scroll offsets
            ########
            uuids = list(btrfs_fs.keys())
            if selected_idx >= len(uuids):
                selected_idx = max(0, len(uuids) - 1)

            if timeline:
                max_offset = max(0, len(timeline) - 1)
                if scroll_offset > max_offset:
                    scroll_offset = max_offset

            ########
            # Help dialogue
            ########
            if show_help:
                if not help_drawn:
                    help_lines = [
                        " ↑ ↓       Select filesystem",
                        " L         Toggle labels ↔ UUID",
                        " i         Toggle IOPS charts",
                        " ← →       Scroll history (±5 cols)",
                        " 0         Jump to live",
                        " w         Auto ↔ Custom width",
                        " [ ]       Custom width −/+ 5 cols (enters Custom)",
                        " { }       Adjust chart HEIGHT −/+ 1 row",
                        " y / Y     Y-scale: zoom in / out (active group), r reset",
                        " p         Pause screen updates",
                        " q         Quit",
                        " ? / h     Toggle help",
                    ]
                    help_win = draw_help_popup(stdscr, help_lines)
                    help_drawn = True
                # When help is visible, skip normal rendering.
                continue
            else:
                # Help just closed: clear the popup and underlying screen.
                if help_win is not None:
                    help_win.clear()
                    help_win.refresh()
                    help_win = None
                    stdscr.clear()

            ########
            # Rendering
            ########
            if not paused:
                stdscr.erase()

            height, width = stdscr.getmaxyx()
            min_height = len(uuids) + 4  # header + rows + separator
            if (height < min_height) or (width < MIN_WIDTH):
                stdscr.addstr(0, 0, f"Terminal too small (need ≥ {min_height}x{MIN_WIDTH}).")
                stdscr.refresh()
                time.sleep(UI_TICK_S)
                continue

            # Header bar.
            stdscr.attron(curses.color_pair(COLOR_HEADER))
            stdscr.addstr(0, 0, "Btrfs Filesystem I/O Monitor")
            stdscr.addstr(0, width - 6 - len(__version__), f"fsmon {__version__}")
            stdscr.attroff(curses.color_pair(COLOR_HEADER))

            # Status bar.
            filled_secs = len(timeline) * sample_interval_s
            if paused:
                left = f"Press ? for help  refresh=PAUSED  history={_mmss(filled_secs)}"
            else:
                left = (
                    f"Press ? for help  refresh={sample_interval_s:.3g}s  "
                    f"history={_mmss(filled_secs)}"
                )
            right = f"{width}x{height}"
            stdscr.attron(curses.color_pair(COLOR_FOOTER))
            left_max = max(0, width - len(right) - 1)
            stdscr.addstr(height - 1, 0, left[:left_max])
            stdscr.addstr(height - 1, width - 1 - len(right), right)
            stdscr.attroff(curses.color_pair(COLOR_FOOTER))

            # Compute label column width based on longest label/UUID.
            max_label_len = (
                max((len(fs_labels[u]) for u in uuids), default=COL_LABEL)
                if use_labels
                else 36
            )
            fixed_cols = COL_READ_BW + COL_WRITE_BW + COL_IOPS
            label_col = min(
                max_label_len,
                max(COL_LABEL, width - fixed_cols - 2),
            )

            # Column header for the summary table.
            stdscr.attron(curses.color_pair(COLOR_COL_HEADER))
            header_line = (
                f"{'Filesystem':<{label_col}}"
                f"{'Read/s':>{COL_READ_BW}}"
                f"{'Write/s':>{COL_WRITE_BW}}"
                f"{'IOPS(R/W)':>{COL_IOPS}}"
            )
            stdscr.addstr(2, 1, header_line[: max(0, width - 2)])
            stdscr.attroff(curses.color_pair(COLOR_COL_HEADER))

            # Filesystem rows.
            for uuid_idx, uuid in enumerate(uuids):
                label = fs_labels[uuid] if use_labels else uuid
                if len(label) > label_col:
                    label_display = label[: label_col - 3] + "..."
                else:
                    label_display = label

                if uuid in last_deltas:
                    d = last_deltas[uuid]
                    read_bw   = d["read_bw"]
                    write_bw  = d["write_bw"]
                    read_iops = d["read_iops"]
                    write_iops = d["write_iops"]
                else:
                    read_bw = write_bw = read_iops = write_iops = 0.0

                iops_str = f"{format_base10(read_iops)}/{format_base10(write_iops)}"
                line = (
                    f"{label_display:<{label_col}}"
                    f"{format_iec(read_bw):>{COL_READ_BW}}"
                    f"{format_iec(write_bw):>{COL_WRITE_BW}}"
                    f"{iops_str:>{COL_IOPS}}"
                )
                fs_list_ypos = 3 + uuid_idx
                truncated = line[: max(0, width - 2)]
                if uuid_idx == selected_idx:
                    # Highlight currently selected filesystem.
                    stdscr.attron(curses.color_pair(COLOR_SELECTED))
                    stdscr.addstr(fs_list_ypos, 0, ">" + truncated)
                    stdscr.attroff(curses.color_pair(COLOR_SELECTED))
                else:
                    stdscr.addstr(fs_list_ypos, 0, " " + truncated)

            # Horizontal separator between table and charts.
            stdscr.attron(curses.color_pair(COLOR_HLINE))
            stdscr.addstr(3 + len(uuids), 0, "-" * (width - 1))
            stdscr.attroff(curses.color_pair(COLOR_HLINE))

            # If paused, skip chart updates but still refresh screen.
            if paused:
                stdscr.refresh()
                time.sleep(UI_TICK_S)
                continue

            # Chart area geometry.
            chart_start = 4 + len(uuids)
            available_w = width - 1
            available_h = height - 1 - chart_start

            if available_h >= chart_h and uuids:
                selected_fs = uuids[selected_idx]
                # Build list of charts (title, data sequence, colour, is_iops?).
                charts: list[tuple[str, Sequence[float], int, bool]] = [
                    ("Read bytes/sec",  history[selected_fs]["read_bw"],  COLOR_CHART_BW_READ,  False),
                    ("Write bytes/sec", history[selected_fs]["write_bw"], COLOR_CHART_BW_WRITE, False),
                ]
                if show_iops_charts:
                    charts += [
                        ("Read IOPS",  history[selected_fs]["read_iops"],  COLOR_CHART_IOPS_READ,  True),
                        ("Write IOPS", history[selected_fs]["write_iops"], COLOR_CHART_IOPS_WRITE, True),
                    ]

                ncharts = len(charts)

                # Compute chart layout (auto or custom width).
                if width_mode == "auto":
                    cols, rows, chart_w = _fit_layout_auto(
                        available_w,
                        available_h,
                        ncharts,
                        chart_h,
                    )
                else:
                    cols, rows, chart_w = _fit_layout_custom(
                        available_w,
                        available_h,
                        ncharts,
                        chart_h,
                        custom_chart_w,
                    )

                if cols == 0:
                    # Not enough room to draw charts at all.
                    stdscr.addstr(
                        chart_start,
                        0,
                        "Not enough space to render charts.",
                    )
                else:
                    # Select Y-scale overrides and their suffix labels.
                    y_bw_override = (
                        y_manual_bw if y_mode_bw == "manual" else None
                    )
                    y_iops_override = (
                        y_manual_iops if y_mode_iops == "manual" else None
                    )
                    bw_suffix = (
                        f"[Max {format_iec(y_manual_bw)}]"
                        if y_bw_override is not None
                        else ""
                    )
                    iops_suffix = (
                        f"[Max {format_base10(y_manual_iops)}]"
                        if y_iops_override is not None
                        else ""
                    )

                    # Draw charts in a grid layout.
                    for i, (title, data_seq, color, use_iops_flag) in enumerate(charts):
                        r = i // cols
                        c = i % cols
                        row_y = chart_start + r * chart_h
                        col_x = c * (chart_w + Y_LABEL_PAD + COL_GAP)
                        display_chart(
                            stdscr,
                            title,
                            data_seq,
                            timeline,
                            row_y,
                            col_x,
                            chart_w,
                            chart_h,
                            color,
                            use_iops=use_iops_flag,
                            offset=scroll_offset,
                            y_override=(
                                y_iops_override
                                if use_iops_flag
                                else y_bw_override
                            ),
                            title_suffix=(iops_suffix if use_iops_flag else bw_suffix),
                            interval_s=sample_interval_s,
                        )

            stdscr.refresh()
            time.sleep(UI_TICK_S)

    except KeyboardInterrupt:
        # Allow graceful exit when user hits Ctrl+c in the terminal.
        stdscr.clear()


def main() -> None:
    """Program entry point: parse arguments, discover filesystems, start UI."""
    args = parse_arguments()

    # Clamp sampling interval into allowed bounds.
    sample_interval_s = max(MIN_INTERVAL_S, min(MAX_INTERVAL_S, float(args.interval)))

    # Parse history duration from CLI. On error, exit with message.
    try:
        history_seconds = parse_duration(args.history)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Keep at least the default window width as minimum history duration.
    history_seconds = max(parse_duration(DEFAULT_HISTORY_STR), history_seconds)

    # Discover Btrfs filesystems and labels from sysfs.
    btrfs_fs, fs_labels = get_btrfs_filesystems()

    # Enter curses main loop.
    curses.wrapper(
        lambda stdscr: display_ui(
            stdscr,
            btrfs_fs,
            fs_labels,
            sample_interval_s,
            history_seconds,
            DEFAULT_MIN_POINTS,
        )
    )


if __name__ == "__main__":
    main()
