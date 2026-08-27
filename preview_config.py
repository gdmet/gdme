#!/usr/bin/env python3
"""Show an in-memory preview of the current animation configuration."""

from __future__ import annotations

import math
import os
import subprocess
import sys
import tkinter as tk
from datetime import datetime, timedelta
from pathlib import Path
from tkinter import messagebox


ROOT = Path(__file__).resolve().parent
RUNTIME_MARKER = "TYPHOON_PREVIEW_VENV_RUNTIME"


def relaunch_with_project_runtime() -> bool:
    """Relaunch with the project virtual environment when double-clicked."""
    if os.environ.get(RUNTIME_MARKER) == "1":
        return False
    venv_python = ROOT / ".venv" / "Scripts" / "python.exe"
    if not venv_python.exists():
        return False

    try:
        already_bundled = Path(sys.executable).resolve() == venv_python.resolve()
    except OSError:
        already_bundled = False
    if already_bundled:
        return False

    environment = os.environ.copy()
    environment[RUNTIME_MARKER] = "1"
    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(
        [str(venv_python), str(Path(__file__).resolve())],
        cwd=str(ROOT),
        env=environment,
        creationflags=creation_flags,
    )
    return True


def preview_moment(animation, cfg: dict) -> datetime:
    """Use the real season start so the timeline preview shows a useful date."""
    starts: list[datetime] = []
    for path in sorted(animation.TRACKS_DIR.glob("*.dat")):
        points = animation.parse_btk(path, cfg)
        if points:
            starts.append(points[0].time)
    if starts:
        buffer_hours = float(cfg.get("timeline_edge_buffer_hours", 12.0))
        moving_preroll_hours = (
            max(0.0, float(cfg.get("intro_seconds", 0.0)))
            * animation.simulation_hours_per_second(cfg)
        )
        return min(starts) - timedelta(
            hours=buffer_hours + moving_preroll_hours
        )
    return datetime(2000, 6, 15, 0)


def make_background(animation, cfg: dict, width: int, height: int):
    Image = animation.Image
    map_path = animation.required_asset_path(
        cfg, "map_file", animation.IMAGE_ASSET_EXTENSIONS
    )
    with Image.open(map_path) as source:
        source = source.convert("RGB")
        # Match IntroReader: preserve aspect ratio, cover the whole frame and
        # crop the overflow equally around the fixed image centre.
        scale = max(width / source.width, height / source.height)
        scaled_width = max(width, int(math.ceil(source.width * scale)))
        scaled_height = max(height, int(math.ceil(source.height * scale)))
        source = source.resize(
            (scaled_width, scaled_height), Image.Resampling.LANCZOS
        )
        crop_x = (scaled_width - width) // 2
        crop_y = (scaled_height - height) // 2
        return source.crop((crop_x, crop_y, crop_x + width, crop_y + height))


def read_video_frame(animation, path: Path, seconds: float):
    """Decode one frame at or immediately after seconds without a disk cache."""
    av = animation.av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        duration = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None
            else seconds + 1.0
        )
        rate = float(stream.average_rate)
        target = min(max(0.0, seconds), max(0.0, duration - 1.0 / rate))
        seek_time = max(0.0, target - 0.5)
        container.seek(
            int(seek_time / float(stream.time_base)),
            stream=stream,
            backward=True,
            any_frame=False,
        )
        chosen = None
        for frame in container.decode(stream):
            chosen = frame
            frame_time = (
                float(frame.pts * frame.time_base)
                if frame.pts is not None
                else 0.0
            )
            if frame_time + 0.5 / rate >= target:
                break
        if chosen is None:
            raise RuntimeError(f"无法读取视频帧：{path.name}")
        return chosen.to_image().convert("RGB")


def make_ts_icon(animation, cfg: dict):
    np = animation.np
    Image = animation.Image
    materials = animation.discover_materials()
    path = materials.get("ts")
    if path is None:
        raise FileNotFoundError("在 tc_icons 中找不到 TS.mp4 素材")

    image = read_video_frame(animation, path, 0.0)
    source_width, source_height = image.size
    scales = cfg.get("icon_state_scales", {})
    if "ts" in scales:
        target_size = max(
            1,
            int(math.ceil(max(source_width, source_height) * float(scales["ts"]))),
        )
    else:
        target_size = max(1, int(cfg.get("icon_size", 101)))
    target_size = max(
        1,
        int(
            round(
                target_size
                * animation.magnification_factor(cfg, "tc_icon_mag")
            )
        ),
    )

    if not cfg.get("icon_preserve_source_canvas", True):
        rgb = np.asarray(image, dtype=np.uint8)
        threshold = int(cfg.get("icon_glow_crop_threshold", 3))
        ys, xs = np.nonzero(rgb.max(axis=2) > threshold)
        if xs.size:
            side = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
            side = max(1, int(math.ceil(side * 1.12)))
            center_x = (int(xs.min()) + int(xs.max())) // 2
            center_y = (int(ys.min()) + int(ys.max())) // 2
            left = max(0, min(source_width - side, center_x - side // 2))
            top = max(0, min(source_height - side, center_y - side // 2))
            image = image.crop(
                (left, top, min(source_width, left + side), min(source_height, top + side))
            )

    image = image.resize((target_size, target_size), Image.Resampling.LANCZOS)
    rgb = np.asarray(image, dtype=np.uint8)
    mode = str(cfg.get("icon_composite_mode", "screen")).lower()
    if mode == "screen":
        alpha = np.ones((target_size, target_size), dtype=np.uint8)
    else:
        rgb, alpha = animation.black_key(
            rgb,
            int(cfg.get("black_key_low", 82)),
            int(cfg.get("black_key_high", 118)),
        )
    rgb, alpha = animation.hue_shift_icon((rgb, alpha), "ts", 35.0)
    return rgb, alpha, mode


def add_preview_track(animation, canvas, cfg: dict):
    """Draw a LOW -> TD -> TS track arriving at the icon from the southeast."""
    Image = animation.Image
    ImageColor = animation.ImageColor
    ImageDraw = animation.ImageDraw
    width, height = canvas.size
    p0 = (width * 0.82, height * 0.82)
    p1 = (width * 0.74, height * 0.65)
    p2 = (width * 0.61, height * 0.70)
    p3 = (width * 0.50, height * 0.50)
    points = []
    for index in range(73):
        t = index / 72.0
        u = 1.0 - t
        points.append(
            (
                u**3 * p0[0]
                + 3.0 * u * u * t * p1[0]
                + 3.0 * u * t * t * p2[0]
                + t**3 * p3[0],
                u**3 * p0[1]
                + 3.0 * u * u * t * p1[1]
                + 3.0 * u * t * t * p2[1]
                + t**3 * p3[1],
            )
        )
    groups = [
        ("low", points[:25]),
        ("td", points[24:49]),
        ("ts", points[48:]),
    ]
    if cfg.get("track_antialias", True):
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        layer = animation.draw_track_groups_native_aa(layer, groups, cfg)
        return Image.alpha_composite(
            canvas.convert("RGBA"), layer
        ).convert("RGB")

    line_width = max(
        1,
        int(
            round(
                float(cfg.get("track_width", 4))
                * animation.magnification_factor(cfg, "track_width_mag")
            )
        ),
    )
    layer = Image.new(
        "RGBA",
        (width, height),
        (0, 0, 0, 0),
    )
    draw = ImageDraw.Draw(layer)
    radius = line_width / 2.0
    configured_colours = cfg.get("track_colors", {})
    for state, group_points in groups:
        style = animation.track_style(state, cfg)
        colour_name = configured_colours.get(
            state, animation.STATE_COLOURS.get(state, "#888888")
        )
        colour = ImageColor.getrgb(colour_name) + (
            int(round(255 * style["opacity"])),
        )
        if style["pattern"] == "dotted":
            dot_radius = line_width * style["dot_radius_widths"]
            for x, y in animation.dotted_polyline(
                group_points, line_width * style["spacing_widths"]
            ):
                draw.ellipse(
                    (x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius),
                    fill=colour,
                )
        elif style["pattern"] == "dashed":
            for dash in animation.dashed_polyline(
                group_points,
                line_width * style["dash_widths"],
                line_width * style["gap_widths"],
            ):
                draw.line(dash, fill=colour, width=line_width, joint="curve")
                for x, y in (dash[0], dash[-1]):
                    draw.ellipse(
                        (x - radius, y - radius, x + radius, y + radius),
                        fill=colour,
                    )
        else:
            draw.line(group_points, fill=colour, width=line_width, joint="curve")
            for x, y in (group_points[0], group_points[-1]):
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius),
                    fill=colour,
                )
    return Image.alpha_composite(canvas.convert("RGBA"), layer).convert("RGB")


def add_test_icon_and_label(animation, canvas, cfg: dict):
    Image = animation.Image
    ImageDraw = animation.ImageDraw
    ImageFilter = animation.ImageFilter
    np = animation.np
    width, height = canvas.size
    center = (width / 2.0, height / 2.0)
    icon_rgb, icon_alpha, mode = make_ts_icon(animation, cfg)
    pixels = np.asarray(canvas).copy()
    animation.paste_rgba(pixels, icon_rgb, icon_alpha, center, 1.0, mode)
    canvas = Image.fromarray(pixels, "RGB")

    display_size = max(icon_rgb.shape[0], icon_rgb.shape[1])
    label_position = (
        center[0] + display_size * float(cfg.get("label_offset_x_icons", 0.56)),
        center[1] + display_size * float(cfg.get("label_offset_y_icons", -0.58)),
    )
    icon_factor = animation.magnification_factor(cfg, "tc_icon_mag")
    font = animation.load_font(
        max(1, int(round(float(cfg.get("label_size", 28)) * icon_factor))),
        bold=True,
        font_file=cfg.get("font_file"),
    )
    stroke = max(0, int(cfg.get("label_stroke_width", 3)))
    shadow_opacity = max(
        0.0, min(1.0, float(cfg.get("label_shadow_opacity", 0.70)))
    )
    shadow_alpha = int(round(255 * shadow_opacity))
    shadow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer)
    shadow_draw.text(
        (
            label_position[0] + float(cfg.get("label_shadow_offset_x", 3)),
            label_position[1] + float(cfg.get("label_shadow_offset_y", 3)),
        ),
        "TEST",
        font=font,
        fill=(0, 0, 0, shadow_alpha),
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, shadow_alpha),
    )
    blur = max(0.0, float(cfg.get("label_shadow_blur", 3.0)))
    if blur:
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(blur))

    label_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(label_layer).text(
        label_position,
        "TEST",
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=stroke,
        stroke_fill=(0, 0, 0, 255),
    )
    canvas = Image.alpha_composite(canvas.convert("RGBA"), shadow_layer)
    return Image.alpha_composite(canvas, label_layer).convert("RGB")


def add_timeline(animation, canvas, cfg: dict, moment: datetime):
    path = animation.required_asset_path(
        cfg, "timeline_video_file", animation.VIDEO_ASSET_EXTENSIONS
    )
    np = animation.np
    Image = animation.Image
    image = read_video_frame(
        animation, path, animation.timeline_source_seconds(moment, cfg)
    )
    scale = max(0.01, float(cfg.get("timeline_scale", 0.25)))
    target_width = max(1, int(round(image.width * scale)))
    target_height = max(1, int(round(image.height * scale)))
    image = image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    rgb_f = np.asarray(image, dtype=np.float32)
    luminance = (
        rgb_f[..., 0] * 0.2126
        + rgb_f[..., 1] * 0.7152
        + rgb_f[..., 2] * 0.0722
    )
    floor = max(0.0, float(cfg.get("timeline_black_floor", 16)))
    if floor:
        key = np.clip((luminance - floor) / floor, 0.0, 1.0)
        key = key * key * (3.0 - 2.0 * key)
        rgb_f *= key[..., None]
    timeline_rgb = np.clip(rgb_f, 0, 255).astype(np.uint8)

    pixels = np.asarray(canvas).copy()
    margin_x = int(cfg.get("timeline_margin_x", 0))
    margin_y = int(cfg.get("timeline_margin_y", 0))
    animation.paste_rgba(
        pixels,
        timeline_rgb,
        np.ones((target_height, target_width), dtype=np.uint8),
        (
            margin_x + target_width / 2.0,
            canvas.height - margin_y - target_height / 2.0,
        ),
        1.0,
        "screen",
    )
    return Image.fromarray(pixels, "RGB")


def build_preview(animation):
    cfg = animation.load_config(ROOT / "config.py")
    font_path = animation.required_asset_path(
        cfg, "font_file", animation.FONT_ASSET_EXTENSIONS
    )
    try:
        animation.ImageFont.truetype(str(font_path), 12)
    except OSError as exc:
        raise RuntimeError(f"无法读取必需字体：{font_path}") from exc
    cfg["font_file"] = str(font_path)
    resolution = int(cfg.get("resolution", 720))
    if resolution not in {720, 1080}:
        raise ValueError("config.py 的 resolution 只能是 720 或 1080")
    width, height = resolution * 16 // 9, resolution
    cfg["width"], cfg["height"] = width, height
    cfg["_map_zoom"] = 1.0
    canvas = make_background(animation, cfg, width, height)
    canvas = add_preview_track(animation, canvas, cfg)
    canvas = add_timeline(animation, canvas, cfg, preview_moment(animation, cfg))
    canvas = animation.draw_ace_bar(
        canvas, 100.0, cfg, cfg.get("font_file"), {}
    )
    return add_test_icon_and_label(animation, canvas, cfg)


def main() -> None:
    if relaunch_with_project_runtime():
        return
    root = tk.Tk()
    root.withdraw()
    try:
        import animation
        from PIL import ImageTk

        preview = build_preview(animation)
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        available_width = max(320, screen_width - 80)
        available_height = max(180, screen_height - 120)
        display_scale = min(
            1.0,
            available_width / preview.width,
            available_height / preview.height,
        )
        if display_scale < 1.0:
            display = preview.resize(
                (
                    max(1, int(round(preview.width * display_scale))),
                    max(1, int(round(preview.height * display_scale))),
                ),
                animation.Image.Resampling.LANCZOS,
            )
        else:
            display = preview

        photo = ImageTk.PhotoImage(display, master=root)
        root.title("Typhoon animation config preview")
        root.resizable(False, False)
        label = tk.Label(root, image=photo, borderwidth=0, highlightthickness=0)
        label.image = photo
        label.pack()
        root.bind("<Escape>", lambda _event: root.destroy())
        root.deiconify()
        root.mainloop()
    except BaseException as exc:
        messagebox.showerror("Preview failed", str(exc), parent=root)
        root.destroy()


if __name__ == "__main__":
    main()
