# capview

Minimal low-latency v4l2 capture card viewer. No demuxer, no decoder, no frame
queue, no threading. Just a tight DQBUF→texture→present loop on top of SDL2.

Rust port. Same architecture, same latency characteristics. Memory safety where
it matters (frame handling, protocol dispatch), raw `ioctl` where it doesn't
(the kernel ABI isn't going to segfault you, but your clipboard protocol handler
might).

## Build

Needs Rust 1.75+, SDL2, wayland-client, and zlib dev headers.

```sh
# Debian/Ubuntu
sudo apt install libsdl2-dev libwayland-dev zlib1g-dev pkg-config

# Arch
sudo pacman -S sdl2 wayland zlib

# Fedora
sudo dnf install SDL2-devel wayland-devel zlib-devel

cargo build --release
```

Binary lands in `target/release/capview`.

```sh
# Or install system-wide
make install
```

## Docker build (single binary)

`build.sh` builds inside Docker and extracts a single binary — no local Rust
toolchain needed:

```sh
./build.sh                    # binary at /tmp/capview-build/capview
./build.sh ~/bin              # binary at ~/bin/capview
make docker OUTPUT_DIR=~/bin  # same thing via make
```

The binary is dynamically linked against SDL2 and wayland-client (same as the
C version), so the target host needs those libs installed.

## Usage

```sh
# Elgato HD60X at 1080p60 NV12
capview --device /dev/video0 --width 1920 --height 1080 --fps 60

# YUYV webcam
capview --device /dev/video2 --format YUYV --fps 30
```

## Keys

- `q` / `Esc` — quit
- `s` — save screenshot (PNG to ~/Pictures)
- `c` — copy screenshot to clipboard (Wayland native)

## CLI Options

```
-d, --device <PATH>     v4l2 device (default: /dev/video0)
-W, --width <NUM>       capture width (default: 1920)
-H, --height <NUM>      capture height (default: 1080)
-f, --fps <NUM>         target framerate (default: 60)
-F, --format <FMT>      pixel format: NV12, YUYV, UYVY (default: NV12)
-q, --quiet             suppress all output
    --fork              fork to background, detach from terminal
```

## Config file

Auto-created at `~/.config/capview/capview.conf` on first run (respects `$XDG_CONFIG_HOME`). CLI always overrides.

```ini
# device     = /dev/video0
# width      = 1920
# height     = 1080
# fps        = 60
# format     = nv12
# buffers    = 2
# window     = 960x540
vsync      = false
smooth     = false
fullscreen = false
quiet      = false
daemonize  = false
```

## Clipboard

Screenshot-to-clipboard uses native Wayland protocols (no wl-copy needed on
modern compositors):

1. **ext-data-control-v1** — standardized protocol, KDE Plasma 6.4+, wlroots 0.18+
2. **wl-copy fallback** — shells out to wl-copy for older compositors (Sway, Hyprland)

## Pixel formats

- **NV12** — Y + interleaved UV planes. What the HD60X uses at 1080p120.
- **YUYV** — packed 4:2:2. Most UVC devices at lower framerates.
- **UYVY** — alternate packed 4:2:2 byte order.

Check your device: `v4l2-ctl -d /dev/video0 --list-formats-ext`
