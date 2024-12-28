# fsmon: A Real-Time Btrfs I/O Monitor

`fsmon` monitors the I/O activity of Btrfs filesystems in real time, displaying bandwidth and IOPS statistics for detected filesystems and their devices.

It combines I/O statistics from all member devices of each Btrfs filesystem, providing a unified view of the filesystem's overall activity.

## Features

- Real-time monitoring of read/write bandwidth and IOPS.
- Visual charts for I/O statistics.
- Dynamic terminal size handling.
- Lightweight and efficient, leveraging sysfs for minimal overhead.

## Requirements

- Python 3.6 or higher.
- Btrfs filesystems mounted on the system.
- Sufficient permissions to access `/sys/fs/btrfs` and related devices in `/sys/block`.

## Usage

- `-h` or `--help`: Display usage information.
- `-v` or `--version`: Display version.

## License

This program is licensed under the [GNU General Public License v3.0 or later](https://www.gnu.org/licenses/gpl-3.0.html).