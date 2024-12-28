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
__version__     = "0.1.0"

import argparse
import curses
import os
import sys
import time

from collections import defaultdict


####### Configuration Options ########

# Colors
USE_TERM_COLORS = True      # Use terminal's default colour
COLOR_CHART_BW_READ    = 3  # Green
COLOR_CHART_BW_WRITE   = 2  # Red
COLOR_CHART_IOPS_READ  = 7  # Cyan
COLOR_CHART_IOPS_WRITE = 6  # Magenta
COLOR_SELECTED         = 7  # Cyan
COLOR_COL_HEADER       = 4  # Yellow
COLOR_HEADER           = 8  # Terminal's default colour
COLOR_FOOTER           = 8  # Terminal's default colour
COLOR_HLINE            = 4  # Yellow

# Column widths (label, read, write, iops)
COL_LABEL    = 12
COL_READ_BW  = 10
COL_WRITE_BW = 10
COL_IOPS     = 11

# Minimum allowed terminal size
MIN_WIDTH = COL_LABEL + COL_READ_BW + COL_WRITE_BW + COL_IOPS

# Chart size
CHART_HEIGHT = 11
CHART_WIDTH  = 61

# Btrfs labels and member devices are read from sysfs
SYSFS_PATH = "/sys/fs/btrfs"

####### Configuration End ########

def init_colors():
    """
    Initialise colors
    """
    curses.start_color()

    if USE_TERM_COLORS:
        curses.use_default_colors()
        default_bg = -1  # Terminal's default background
    else:
        default_bg = curses.COLOR_BLACK

    # Initialize all 8 default colour pairs
    curses.init_pair(1, curses.COLOR_BLACK, default_bg)   # Black
    curses.init_pair(2, curses.COLOR_RED, default_bg)     # Red
    curses.init_pair(3, curses.COLOR_GREEN, default_bg)   # Green
    curses.init_pair(4, curses.COLOR_YELLOW, default_bg)  # Yellow
    curses.init_pair(5, curses.COLOR_BLUE, default_bg)    # Blue
    curses.init_pair(6, curses.COLOR_MAGENTA, default_bg) # Magenta
    curses.init_pair(7, curses.COLOR_CYAN, default_bg)    # Cyan
    curses.init_pair(8, curses.COLOR_WHITE, default_bg)   # White


def parse_arguments():
    """
    Parse command-line arguments for fsmon.
    """
    parser = argparse.ArgumentParser(
        description=__description__
    )
    parser.add_argument(
        "-v", "--version", action="version", version=f"fsmon {__version__}",
        help="Show program version and exit."
    )
    return parser.parse_args()


def get_btrfs_filesystems():
    """
    Fetch all Btrfs filesystems and their devices.
    """
    btrfs_fs = {}  # Dictionary: UUID -> list of device paths
    labels   = {}  # Dictionary: UUID -> label (or UUID if no label)

    # UUIDs are directory entries in SYSFS_PATH
    try:
        uuids = os.listdir(SYSFS_PATH)
    except FileNotFoundError:
        print(f"Error: '{SYSFS_PATH}' does not exist. Ensure sysfs is mounted.", file=sys.stderr)
        exit(1)
    except PermissionError:
        print(f"Error: Permission denied when accessing '{SYSFS_PATH}'", file=sys.stderr)
        exit(1)
    except OSError as e:
        print(f"Error: Unable to access '{SYSFS_PATH}': {e}", file=sys.stderr)
        exit(1)

    # Process each UUID
    for uuid in uuids:
        devices_path = os.path.join(SYSFS_PATH, uuid, "devices")
        label_path   = os.path.join(SYSFS_PATH, uuid, "label")

        # Get filesystem label and member devices
        if os.path.isdir(devices_path) and os.path.exists(label_path):
            # Read the devices directory
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

            # Get filesystem label
            label = ""
            try:
                with open(label_path, "r") as file:
                    label = file.read().strip()
                    # Decode label as UTF-8 with error handling
                    label = label.encode("utf-8").decode("utf-8", errors="replace")
            except PermissionError:
                print(f"Error: Permission denied when reading '{label_path}'", file=sys.stderr)
                continue
            except OSError as e:
                print(f"Error: Unable to read '{label_path}': {e}", file=sys.stderr)
                continue

            # Use UUID if no label is available
            labels[uuid] = label if label else uuid

    # Check if any filesystems were found
    if not btrfs_fs:
        print(f"Error: No Btrfs filesystems found in '{SYSFS_PATH}'")
        exit(1)

    # Return the list of found filesystems
    return btrfs_fs, labels

def get_device_stats(btrfs_fs):
    """
    Calculate aggregated statistics for each Btrfs filesystem.
    """
    fs_stats    = {}  # Dictionary: UUID -> aggregated statistics
    sector_size = 512
    # Linux device/stat file always use 512-byte sector size.
    # Reference: https://www.kernel.org/doc/Documentation/block/stat.txt

    # Iterate through each UUID and its associated devices
    for uuid, devices in btrfs_fs.items():
        # Initialise counters for aggregated statistics
        read_bytes  = 0
        write_bytes = 0
        read_ops    = 0
        write_ops   = 0

        # Iterate through each device stat file
        for dev_stat_path in devices:
            try:
                # Read the stat file and parse its fields
                with open(dev_stat_path, "r") as file:
                    stats = file.read().split()
                # Ensure the stat file contains the expected fields
                if len(stats) < 7:
                    print(f"Warning: Stat file '{dev_stat_path}' is malformed or incomplete.", file=sys.stderr)
                    continue
                # Update operation counts and byte counters
                read_ops     += int(stats[0])  # Reads completed
                write_ops    += int(stats[4])  # Writes completed
                read_sectors  = int(stats[2])  # Sectors read
                write_sectors = int(stats[6])  # Sectors written
                read_bytes   += read_sectors * sector_size
                write_bytes  += write_sectors * sector_size
            except FileNotFoundError:
                # Skip devices where the stat file is missing
                print(f"Warning: Stat file not found for device at '{dev_stat_path}'", file=sys.stderr)
                continue
            except PermissionError:
                # Skip devices we cannot read
                print(f"Error: Permission denied when reading '{dev_stat_path}'", file=sys.stderr)
                continue
            except ValueError:
                # Skip devices where the stat file contains invalid data
                print(f"Warning: Invalid data in stat file '{dev_stat_path}'", file=sys.stderr)
                continue
            except OSError as e:
                # Abort on other errors
                print(f"Error: Unable to read '{dev_stat_path}': {e}", file=sys.stderr)
                exit(1)

        # Store aggregated statistics for the current UUID
        fs_stats[uuid] = {
            "read_bytes": read_bytes,
            "write_bytes": write_bytes,
            "read_ops": read_ops,
            "write_ops": write_ops
        }
    return fs_stats


def format_iec(value):
    """
    Convert numbers to IEC units: B, KiB, MiB, GiB...
    """
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    for unit in units:
        if value < 1024:
            return f"{value:.1f} {unit}"
        value /= 1024

    return f"{value:.1f} PiB"


def format_base10(value):
    """
    Convert numbers to base-10 units: k, M, G...
    """
    prefixes = {
        'P': 1e15,  # peta
        'T': 1e12,  # tera
        'G': 1e9,   # giga
        'M': 1e6,   # mega
        'k': 1e3    # kilo
    }

    v = float(value)
    for suffix, threshold in prefixes.items():
        if v >= threshold:
            return f"{v / threshold:.1f}{suffix}"

    return f"{v:.1f}"


def display_chart(stdscr, title, data_list, row_start, col_start, CHART_WIDTH, color, use_iops=False):
    """
    Renders a single chart.
    """
    # Store the max value
    max_val = max(max(data_list), 1)

    # Print chart title
    stdscr.addstr(row_start, col_start + 9 + (CHART_WIDTH // 2) - (len(title) //2), title)

    # Number of diagram rows
    y_axis_rows = CHART_HEIGHT - 2
    for i in range(y_axis_rows-1):
        # Set row max value
        threshold = max_val * ((y_axis_rows-1) - i) / (y_axis_rows-1)
        # Construct a complete row
        chart_row = "".join(
            "|" if val > threshold else " " if i == (y_axis_rows-2) else "."
            for val in data_list[-CHART_WIDTH:]
        ).ljust(CHART_WIDTH-1)

        if use_iops:
            y_axis_label = format_base10(threshold)
        else:
            y_axis_label = format_iec(threshold)

        # Draw Y-axis label
        stdscr.addstr(row_start + 1 + i, col_start, f"{y_axis_label:>10} ")

        # Draw chart rows
        if i == (y_axis_rows-2):
            # Underline the bottom row
            stdscr.addstr(row_start + 1 + i, col_start + 11, chart_row, curses.A_UNDERLINE | curses.color_pair(color) )
        else:
            stdscr.addstr(row_start + 1 + i, col_start + 11, chart_row, curses.color_pair(color))
    # Draw X-axis scale
    x_axis = " ".join(f"{x:>4}" for x in range(CHART_WIDTH-1, -1, -5))
    stdscr.addstr(row_start + y_axis_rows, col_start + 8, x_axis)


def display_ui(stdscr, btrfs_fs, fs_labels):
    """
    Main UI.
    """
    curses.curs_set(0)
    stdscr.nodelay(1)
    init_colors()

    selected_idx = 0
    use_labels = True  # Show labels as default
    show_iops_charts = False  # Toggles display of iops charts

    prev_stats = {}
    first_skipped = set()

    history = defaultdict(
        lambda: {
            "read_bw": [0] * CHART_WIDTH,
            "write_bw": [0] * CHART_WIDTH,
            "read_iops": [0] * CHART_WIDTH,
            "write_iops": [0] * CHART_WIDTH,
        }
    )

###
# Begin main UI loop
###
    try:
        # Main UI loop
        while True:
            # Check for keyboard input
            key = stdscr.getch()
            if key == curses.KEY_UP:
                selected_idx = max(0, selected_idx - 1)
            elif key == curses.KEY_DOWN:
                selected_idx += 1
                deltas = {}
            elif key in [ord("q"), ord("Q")]:
                break
            elif key in [ord("l"), ord("L")]:
                use_labels = not use_labels
            elif key in [ord("i"), ord("I")]:
                show_iops_charts = not show_iops_charts

            # Get the current statistics for all Btrfs filesystems
            current_stats = get_device_stats(btrfs_fs)

            # Calculate deltas
            deltas = {}
            for uuid, stats in current_stats.items():
                # Get the previous stats for this UUID; default to 0 for all fields if not found
                prev = prev_stats.get(
                    uuid,
                    {"read_bytes": 0, "write_bytes": 0, "read_ops": 0, "write_ops": 0},
                )
                # Skip the first iteration for this UUID to avoid incorrect deltas
                # This ensures we have a baseline before calculating differences
                if uuid not in first_skipped:
                    first_skipped.add(uuid)
                    prev_stats[uuid] = stats
                    continue

                # Calculate the delta between current and previous stats
                deltas[uuid] = {
                    "read_bw": stats["read_bytes"] - prev["read_bytes"],
                    "write_bw": stats["write_bytes"] - prev["write_bytes"],
                    "read_iops": stats["read_ops"] - prev["read_ops"],
                    "write_iops": stats["write_ops"] - prev["write_ops"],
                }
            # Update the previous stats to the current stats for the next iteration
            prev_stats = current_stats

            # List of UUIDs
            uuids = list(btrfs_fs.keys())

            # Clear screen
            stdscr.erase()

            # Prevent drawing UI if terminal is too small
            height, width = stdscr.getmaxyx()
            MIN_HEIGHT = len(uuids) + 4  # Enough to show list without charts
            if (height < MIN_HEIGHT) or (width < MIN_WIDTH):
                stdscr.addstr(
                    0, 0, f"Terminal too small (need ≥ {MIN_HEIGHT}x{MIN_WIDTH})."
                )
                time.sleep(1)
                continue

            # Print header and footer
            stdscr.attron(curses.color_pair(COLOR_HEADER))
            stdscr.addstr(0, 0, "Btrfs Filesystem I/O Monitor")
            stdscr.addstr(0, width - 6 - len(__version__), f"fsmon {__version__}")
            stdscr.attroff(curses.color_pair(COLOR_HEADER))
            stdscr.attron(curses.color_pair(COLOR_FOOTER))
            stdscr.addstr(height - 1, 0, "Keys: q=quit, L=labels, i=iops")
            stdscr.addstr(height - 1, width - len(f"{width}x{height}") - 1, f"{width}x{height}")
            stdscr.attroff(curses.color_pair(COLOR_FOOTER))

            # Determine the maximum label length
            max_label_len = max(
                    len(fs_labels[u])
                    for u in uuids
            )

            # Length of fixed columns combined
            fixed_cols = COL_READ_BW + COL_WRITE_BW + COL_IOPS

            # Calculate label column length and set
            # to COL_LABEL length if label is too short
            label_col = min(
                max_label_len,
                max(
                    COL_LABEL,
                    width - fixed_cols - 2
                )
            )
            # Define header row text
            col_header = (
                f"{'Filesystem':<{label_col}}"
                f"{'Read/s':>{COL_READ_BW}}"
                f"{'Write/s':>{COL_WRITE_BW}}"
                f"{'IOPS(R/W)':>{COL_IOPS}}"
            )
            # Write out header text
            stdscr.attron(curses.color_pair(COLOR_COL_HEADER))
            stdscr.addstr(2, 1, col_header)
            stdscr.attroff(curses.color_pair(COLOR_COL_HEADER))

            # Number of items in the filsystem list
            #list_uuids = uuids[:max_list_area]
            list_uuids = uuids

            # clamp selection
            if selected_idx >= len(list_uuids):
                selected_idx = max(0, len(list_uuids) - 1)

            # Loop through each filesystem and print out their stats in columns
            for idx, uuid in enumerate(list_uuids):
                label = fs_labels[uuid] if use_labels else uuid
                if len(label) > label_col:
                    label_display = label[: label_col - 3] + "..."
                else:
                    label_display = label

                if uuid in deltas:
                    read_bw = deltas[uuid]["read_bw"]
                    write_bw = deltas[uuid]["write_bw"]
                    read_iops = deltas[uuid]["read_iops"]
                    write_iops = deltas[uuid]["write_iops"]
                # No stats available, initialise to 0
                else:
                    read_bw = write_bw = read_iops = write_iops = 0

                # Update history for all filesystems:
                h = history[uuid]
                h["read_bw"].append(read_bw)
                h["write_bw"].append(write_bw)
                h["read_iops"].append(read_iops)
                h["write_iops"].append(write_iops)
                for k in ["read_bw", "write_bw", "read_iops", "write_iops"]:
                    h[k] = h[k][-CHART_WIDTH:]

                iops_str = f"{read_iops}/{write_iops}"
                line = (
                    f"{label_display:<{label_col}}"
                    f"{format_iec(read_bw):>{COL_READ_BW}}"
                    f"{format_iec(write_bw):>{COL_WRITE_BW}}"
                    f"{iops_str:>{COL_IOPS}}"
                )
                fs_list_ypos = 3 + idx  # Start filesystem list at 3rd row
                if idx == selected_idx:
                    stdscr.attron(curses.color_pair(COLOR_SELECTED))
                    stdscr.addstr(fs_list_ypos, 0, ">" + line)
                    stdscr.attroff(curses.color_pair(COLOR_SELECTED))
                else:
                    stdscr.addstr(fs_list_ypos, 0, " " + line)

            # Draw a horizontal line
            stdscr.attron(curses.color_pair(COLOR_HLINE))
            stdscr.addstr(3 + len(list_uuids), 0, "—" * width)
            stdscr.attroff(curses.color_pair(COLOR_HLINE))

            # Chart details
            chart_start = 4 + len(list_uuids)
            available_for_charts_w = width - 1
            available_for_charts_h = height - 1 - chart_start

            # Selected filesystem's stats should be rendered as charts
            selected_fs = list_uuids[selected_idx]

            # -- Display logic:
            #  1) If IOPS is *not* toggled
            #  2) If IOPS is toggled
            if not show_iops_charts:
                # Show BW charts side-by-side
                if available_for_charts_w >= ((CHART_WIDTH + 11) * 2) and available_for_charts_h >= CHART_HEIGHT:
                    display_chart(
                        stdscr,
                        "Read bytes/sec",
                        history[selected_fs]["read_bw"],
                        chart_start,
                        0,
                        CHART_WIDTH,
                        COLOR_CHART_BW_READ,
                        use_iops=False,
                    )
                    display_chart(
                        stdscr,
                        "Write bytes/sec",
                        history[selected_fs]["write_bw"],
                        chart_start,
                        CHART_WIDTH + 12,
                        CHART_WIDTH,
                        COLOR_CHART_BW_WRITE,
                        use_iops=False,
                    )
                # Show BW charts stacked
                elif available_for_charts_w >= (CHART_WIDTH + 11) and available_for_charts_h >= (CHART_HEIGHT * 2):
                    display_chart(
                        stdscr,
                        "Read bytes/sec",
                        history[selected_fs]["read_bw"],
                        chart_start,
                        0,
                        CHART_WIDTH,
                        COLOR_CHART_BW_READ,
                        use_iops=False,
                    )
                    display_chart(
                        stdscr,
                        "Write bytes/sec",
                        history[selected_fs]["write_bw"],
                        chart_start + CHART_HEIGHT,
                        0,
                        CHART_WIDTH,
                        COLOR_CHART_BW_WRITE,
                        use_iops=False,
                    )
            # Show BW and/or IOPS charts
            else:
                # Show BW + IOPS charts side-by-side
                if available_for_charts_w >= ((CHART_WIDTH + 11) * 4):
                    display_chart(
                        stdscr,
                        "Read bytes/sec",
                        history[selected_fs]["read_bw"],
                        chart_start,
                        0,
                        CHART_WIDTH,
                        COLOR_CHART_BW_READ,
                        use_iops=False,
                    )
                    display_chart(
                        stdscr,
                        "Write bytes/sec",
                        history[selected_fs]["write_bw"],
                        chart_start,
                        (CHART_WIDTH + 12),
                        CHART_WIDTH,
                        COLOR_CHART_BW_WRITE,
                        use_iops=False,
                    )
                    display_chart(
                        stdscr,
                        "Read IOPS",
                        history[selected_fs]["read_iops"],
                        chart_start,
                        (CHART_WIDTH + 12) * 2,
                        CHART_WIDTH,
                        COLOR_CHART_IOPS_READ,
                        use_iops=True,
                    )
                    display_chart(
                        stdscr,
                        "Write IOPS",
                        history[selected_fs]["write_iops"],
                        chart_start,
                        (CHART_WIDTH + 12) *3,
                        CHART_WIDTH,
                        COLOR_CHART_IOPS_WRITE,
                        use_iops=True,
                    )
                # Show 2x2 side-by-side
                elif available_for_charts_w >= ((CHART_WIDTH + 11) * 2) and available_for_charts_h >= (CHART_HEIGHT * 2):
                    display_chart(
                        stdscr,
                        "Read bytes/sec",
                        history[selected_fs]["read_bw"],
                        chart_start,
                        0,
                        CHART_WIDTH,
                        COLOR_CHART_BW_READ,
                        use_iops=False,
                    )
                    display_chart(
                        stdscr,
                        "Write bytes/sec",
                        history[selected_fs]["write_bw"],
                        chart_start + CHART_HEIGHT,
                        0,
                        CHART_WIDTH,
                        COLOR_CHART_BW_WRITE,
                        use_iops=False,
                    )
                    display_chart(
                        stdscr,
                        "Read IOPS",
                        history[selected_fs]["read_iops"],
                        chart_start,
                        CHART_WIDTH + 12,
                        CHART_WIDTH,
                        COLOR_CHART_IOPS_READ,
                        use_iops=True,
                    )
                    display_chart(
                        stdscr,
                        "Write IOPS",
                        history[selected_fs]["write_iops"],
                        chart_start + CHART_HEIGHT,
                        CHART_WIDTH + 12,
                        CHART_WIDTH,
                        COLOR_CHART_IOPS_WRITE,
                        use_iops=True,
                    )
                # Show 4x1 charts stacked.
                elif available_for_charts_w >= (CHART_WIDTH + 11) and available_for_charts_h >= (CHART_HEIGHT * 4):
                    display_chart(
                        stdscr,
                        "Read bytes/sec",
                        history[selected_fs]["read_bw"],
                        chart_start,
                        0,
                        CHART_WIDTH,
                        COLOR_CHART_BW_READ,
                        use_iops=False,
                    )
                    display_chart(
                        stdscr,
                        "Write bytes/sec",
                        history[selected_fs]["write_bw"],
                        chart_start + CHART_HEIGHT,
                        0,
                        CHART_WIDTH,
                        COLOR_CHART_BW_WRITE,
                        use_iops=False,
                    )
                    display_chart(
                        stdscr,
                        "Read IOPS",
                        history[selected_fs]["read_iops"],
                        chart_start + CHART_HEIGHT * 2,
                        0,
                        CHART_WIDTH,
                        COLOR_CHART_IOPS_READ,
                        use_iops=True,
                    )
                    display_chart(
                        stdscr,
                        "Write IOPS",
                        history[selected_fs]["write_iops"],
                        chart_start + CHART_HEIGHT * 3,
                        0,
                        CHART_WIDTH,
                        COLOR_CHART_IOPS_WRITE,
                        use_iops=True,
                    )
                # Show 2x1 charts stacked
                elif available_for_charts_w >= (CHART_WIDTH + 11) and available_for_charts_h >= (CHART_HEIGHT * 2):
                    display_chart(
                        stdscr,
                        "Read IOPS",
                        history[selected_fs]["read_iops"],
                        chart_start,
                        0,
                        CHART_WIDTH,
                        COLOR_CHART_IOPS_READ,
                        use_iops=True,
                    )
                    display_chart(
                        stdscr,
                        "Write IOPS",
                        history[selected_fs]["write_iops"],
                        chart_start + CHART_HEIGHT,
                        0,
                        CHART_WIDTH,
                        COLOR_CHART_IOPS_WRITE,
                        use_iops=True,
                    )
                # Show 1x2 charts side-by-side
                elif available_for_charts_h >= CHART_HEIGHT and available_for_charts_w >= (CHART_WIDTH * 2):
                    display_chart(
                        stdscr,
                        "Read IOPS",
                        history[selected_fs]["read_iops"],
                        chart_start,
                        0,
                        CHART_WIDTH,
                        COLOR_CHART_IOPS_READ,
                        use_iops=True,
                    )
                    display_chart(
                        stdscr,
                        "Write IOPS",
                        history[selected_fs]["write_iops"],
                        chart_start,
                        CHART_WIDTH + 11,
                        CHART_WIDTH,
                        COLOR_CHART_IOPS_WRITE,
                        use_iops=True,
                    )
            stdscr.refresh()
            time.sleep(1)
    except KeyboardInterrupt:
        # Clear screen and exit on Ctrl-C
        stdscr.clear()
###
# End main UI loop
###

def main():
    # Parse CLI arguments
    args = parse_arguments()

    # Get a list of Btrfs filesystems
    btrfs_fs, fs_labels = get_btrfs_filesystems()

    # Display main UI
    curses.wrapper(lambda stdscr: display_ui(stdscr, btrfs_fs, fs_labels))


if __name__ == "__main__":
    main()
