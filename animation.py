#!/usr/bin/env python3
"""Render an ATCF best-track file as a compact H.264 track animation.

The renderer streams frames directly into the encoder; it never writes an
image sequence.  Every storm owns an independent phase clock.  When its
status changes, both source clips are sampled at the same phase and blended
in premultiplied colour space, which avoids the 75%-opacity flash caused by
stacking two half-transparent layers.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import math
import os
import random
import runpy
import struct
import sys
import time
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta
from fractions import Fraction
from functools import reduce
from math import gcd
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GENERATION_CACHE = ROOT / "generation_cache"
CACHE_SESSION = f"{os.getpid()}_{time.time_ns():x}"
TRACKS_DIR = ROOT / "tc_tracks"
MUSIC_DIR = ROOT / "music"
VIDEO_ASSET_EXTENSIONS = {".mp4", ".mov", ".avi"}
IMAGE_ASSET_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
FONT_ASSET_EXTENSIONS = {".ttf", ".otf", ".ttc"}


def cleanup_stale_preprocess_cache() -> None:
    """Remove unused memmaps while leaving files locked by another run alone."""
    if not GENERATION_CACHE.is_dir():
        return
    for pattern in ("tc_*.npy", "landfall_*.npy"):
        for path in GENERATION_CACHE.glob(pattern):
            try:
                path.unlink()
            except (FileNotFoundError, PermissionError, OSError):
                # Windows refuses to unlink a memmap owned by a live renderer.
                # That run has its own session suffix, so it is safe to skip.
                pass

try:
    import av
    import cv2
    import numpy as np
    from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFilter, ImageFont
except ImportError as exc:
    raise SystemExit(
        "缺少运行环境。请先双击 env_initialize.bat。\n"
        f"原始错误：{exc}"
    ) from exc


SESSION_MEMMAPS: list[np.memmap] = []


def open_session_memmap(
    path: Path,
    *,
    dtype: object,
    shape: tuple[int, ...],
) -> np.memmap:
    """Create and register a non-reusable preprocessing memmap."""
    array = np.lib.format.open_memmap(
        path, mode="w+", dtype=dtype, shape=shape
    )
    SESSION_MEMMAPS.append(array)
    return array


def cleanup_session_preprocess_cache() -> None:
    """Close and remove this run's TC/landfall preprocessing files."""
    while SESSION_MEMMAPS:
        array = SESSION_MEMMAPS.pop()
        try:
            array.flush()
        except (OSError, ValueError):
            pass
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            try:
                mapping.close()
            except (OSError, ValueError):
                pass

    removed = 0
    for path in GENERATION_CACHE.glob(f"*_{CACHE_SESSION}_*.npy"):
        try:
            path.unlink()
            removed += 1
        except FileNotFoundError:
            pass
        except (PermissionError, OSError) as exc:
            print(f"警告：无法删除临时缓存 {path.name}：{exc}", file=sys.stderr)
    if removed:
        print(f"已清理临时预处理缓存：{removed} 个文件")


TC_ICON_STATES = (
    "low", "td", "ts", "c1", "c2", "c3", "c4", "c5", "sd", "ss", "ex"
)


STATE_COLOURS = {
    "low": "#C0EEEE",
    "td": "#0080FF",
    "ts": "#00FF00",
    "c1": "#e9d72d",
    "c2": "#f0cc32",
    "c3": "#efa43a",
    "c4": "#ed5c39",
    "c5": "#e9388d",
    "sd": "#7662dc",
    "ss": "#45c895",
    "ex": "#777777",
}


@dataclass
class TrackPoint:
    time: datetime
    lat: float
    lon: float
    wind: int
    nature: str
    state: str
    raw_name: str
    label: str


@dataclass
class ClipInfo:
    state: str
    path: Path
    width: int
    height: int
    fps: Fraction
    frames: int
    duration: Fraction


@dataclass
class PreparedClip:
    info: ClipInfo
    rgb: np.ndarray
    alpha: np.ndarray


@dataclass
class TimelineClipInfo:
    path: Path
    width: int
    height: int
    fps: Fraction
    frames: int
    duration: float
    calendar_start: float = 0.0


@dataclass
class SeasonTrack:
    path: Path
    points: list[TrackPoint]
    phase_offset: float = 0.0


class AceSeries:
    """Deduplicated six-hourly ACE reports across every season track file."""

    TROPICAL_NATURES = {"TD", "TS", "HU", "TY", "ST"}
    SUBTROPICAL_NATURES = {"SD", "SS"}

    def __init__(self, dat_paths: list[Path], include_subtropical: bool):
        increments: dict[datetime, float] = {}
        seen: set[tuple[str, str, datetime]] = set()
        eligible = set(self.TROPICAL_NATURES)
        if include_subtropical:
            eligible.update(self.SUBTROPICAL_NATURES)

        for path in dat_paths:
            with path.open("r", encoding="utf-8-sig", errors="replace") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    fields = [item.strip() for item in line.split(",")]
                    if len(fields) <= 10:
                        continue
                    raw_time = fields[2]
                    # Only exact synoptic reports count. Values such as 03Z or
                    # 0540Z are rejected before parsing.
                    if len(raw_time) != 10 or not raw_time.isdigit():
                        continue
                    try:
                        moment = datetime.strptime(raw_time, "%Y%m%d%H")
                    except ValueError:
                        continue
                    if moment.hour not in {0, 6, 12, 18}:
                        continue
                    report_key = (
                        fields[0].upper(), fields[1].upper(), moment
                    )
                    if report_key in seen:
                        continue
                    # Mark the timestamp before validating intensity/nature:
                    # a contradictory duplicate must never replace its first row.
                    seen.add(report_key)
                    try:
                        wind = int(fields[8] or 0)
                    except ValueError:
                        continue
                    nature = fields[10].upper()
                    if nature not in eligible or wind < 35:
                        continue
                    increments[moment] = (
                        increments.get(moment, 0.0) + wind * wind / 10000.0
                    )

        self.times = sorted(increments)
        self.cumulative: list[float] = []
        total = 0.0
        for moment in self.times:
            total += increments[moment]
            self.cumulative.append(total)
        self.total = total

    def cumulative_at_report(self, moment: datetime) -> float:
        index = bisect_right(self.times, moment) - 1
        return self.cumulative[index] if index >= 0 else 0.0

    def value_at(self, moment: datetime) -> float:
        """Continuously interpolate between consecutive six-hour refreshes."""
        slot = moment.replace(
            hour=(moment.hour // 6) * 6,
            minute=0,
            second=0,
            microsecond=0,
        )
        next_slot = slot + timedelta(hours=6)
        current = self.cumulative_at_report(slot)
        following = self.cumulative_at_report(next_slot)
        fraction = (moment - slot).total_seconds() / (6.0 * 3600.0)
        return current + (following - current) * max(0.0, min(1.0, fraction))


@dataclass
class TimelineSegment:
    start: datetime
    end: datetime
    speed: float
    video_start: float
    video_duration: float


@dataclass
class CameraRect:
    west: float
    east: float
    south: float
    north: float

    def interpolate(self, other: "CameraRect", weight: float) -> "CameraRect":
        weight = smoothstep(weight)
        return CameraRect(
            self.west + (other.west - self.west) * weight,
            self.east + (other.east - self.east) * weight,
            self.south + (other.south - self.south) * weight,
            self.north + (other.north - self.north) * weight,
        )


@dataclass
class Landfall:
    time: datetime
    lat: float
    lon: float
    state: str
    intensity: int


class CoastlineIndex:
    """Minimal indexed reader for PolyLine/Polygon ESRI Shapefiles."""

    def __init__(self, path: Path, cfg: dict):
        self.cell_size = max(
            0.1, float(cfg.get("coastline_grid_degrees", 1.0))
        )
        self.segments: list[tuple[float, float, float, float]] = []
        self.grid: dict[tuple[int, int], list[int]] = {}
        west = float(cfg["longitude_left"]) - 2.0
        east = float(cfg["longitude_right"]) + 2.0
        south = float(cfg["latitude_bottom"]) - 2.0
        north = float(cfg["latitude_top"]) + 2.0
        with path.open("rb") as handle:
            header = handle.read(100)
            if len(header) != 100 or struct.unpack(">i", header[:4])[0] != 9994:
                raise SystemExit(f"无法识别海岸线 Shapefile：{path.name}")
            while True:
                record_header = handle.read(8)
                if not record_header:
                    break
                if len(record_header) != 8:
                    raise SystemExit(f"海岸线 Shapefile 已损坏：{path.name}")
                _, length_words = struct.unpack(">2i", record_header)
                content = handle.read(length_words * 2)
                if len(content) != length_words * 2:
                    raise SystemExit(f"海岸线 Shapefile 已截断：{path.name}")
                shape_type = struct.unpack("<i", content[:4])[0]
                if shape_type == 0:
                    continue
                if shape_type not in {3, 5}:
                    raise SystemExit(
                        f"海岸线图层必须是 PolyLine/Polygon：{path.name}"
                    )
                part_count, point_count = struct.unpack("<2i", content[36:44])
                parts_end = 44 + part_count * 4
                part_starts = list(
                    struct.unpack(
                        "<" + "i" * part_count,
                        content[44:parts_end],
                    )
                ) + [point_count]
                points = [
                    struct.unpack(
                        "<2d", content[parts_end + i * 16:parts_end + (i + 1) * 16]
                    )
                    for i in range(point_count)
                ]
                for start, end in zip(part_starts, part_starts[1:]):
                    for first, second in zip(points[start:end - 1], points[start + 1:end]):
                        x1, y1 = first
                        x2, y2 = second
                        if (
                            max(x1, x2) < west
                            or min(x1, x2) > east
                            or max(y1, y2) < south
                            or min(y1, y2) > north
                        ):
                            continue
                        self._add_segment(x1, y1, x2, y2)

    def _cell(self, value: float) -> int:
        return math.floor(value / self.cell_size)

    def _add_segment(self, x1: float, y1: float, x2: float, y2: float) -> None:
        index = len(self.segments)
        self.segments.append((x1, y1, x2, y2))
        for cell_x in range(self._cell(min(x1, x2)), self._cell(max(x1, x2)) + 1):
            for cell_y in range(self._cell(min(y1, y2)), self._cell(max(y1, y2)) + 1):
                self.grid.setdefault((cell_x, cell_y), []).append(index)

    def candidates(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> list[tuple[float, float, float, float]]:
        indexes: set[int] = set()
        for cell_x in range(self._cell(min(x1, x2)), self._cell(max(x1, x2)) + 1):
            for cell_y in range(self._cell(min(y1, y2)), self._cell(max(y1, y2)) + 1):
                indexes.update(self.grid.get((cell_x, cell_y), ()))
        return [self.segments[index] for index in indexes]

    def landfall_intersection(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> tuple[float, float, float] | None:
        """Return the first ocean-to-land intersection along a path chord."""
        path_dx, path_dy = x2 - x1, y2 - y1
        intersections: list[tuple[float, float, float]] = []
        for coast_x1, coast_y1, coast_x2, coast_y2 in self.candidates(
            x1, y1, x2, y2
        ):
            coast_dx = coast_x2 - coast_x1
            coast_dy = coast_y2 - coast_y1
            denominator = path_dx * coast_dy - path_dy * coast_dx
            if abs(denominator) < 1e-12:
                continue
            offset_x, offset_y = coast_x1 - x1, coast_y1 - y1
            path_fraction = (offset_x * coast_dy - offset_y * coast_dx) / denominator
            coast_fraction = (offset_x * path_dy - offset_y * path_dx) / denominator
            if not (0.0 <= path_fraction <= 1.0 and 0.0 <= coast_fraction <= 1.0):
                continue
            # Natural Earth coastline direction keeps land on its left.  A
            # negative-to-positive side change is therefore ocean to land.
            side_before = coast_dx * (y1 - coast_y1) - coast_dy * (x1 - coast_x1)
            side_after = coast_dx * (y2 - coast_y1) - coast_dy * (x2 - coast_x1)
            epsilon = 1e-12
            if side_before <= epsilon and side_after > epsilon:
                intersections.append(
                    (
                        path_fraction,
                        x1 + path_fraction * path_dx,
                        y1 + path_fraction * path_dy,
                    )
                )
        return min(intersections, default=None, key=lambda item: item[0])


def simulation_hours_per_second(cfg: dict) -> float:
    seconds_per_day = float(cfg.get("timeline_sec_per_day", 4.0))
    if seconds_per_day <= 0.0:
        raise SystemExit("timeline_sec_per_day 必须大于 0")
    return 24.0 / seconds_per_day


class SeasonTimeline:
    def __init__(
        self,
        tracks: list[SeasonTrack],
        cfg: dict,
        final_activity_end: datetime | None = None,
    ):
        self.base_hours_per_second = simulation_hours_per_second(cfg)
        edge_buffer = timedelta(
            hours=float(cfg.get("timeline_edge_buffer_hours", 12.0))
        )
        # The former map intro was a frozen hold at T-12h. Move the timeline
        # start farther back by the same real duration so it advances from the
        # first map frame while the first TC still appears at exactly T.
        moving_preroll = timedelta(
            hours=max(0.0, float(cfg.get("intro_seconds", 0.0)))
            * self.base_hours_per_second
        )
        self.start = (
            min(track.points[0].time for track in tracks)
            - edge_buffer
            - moving_preroll
        )
        last_track_end = max(track.points[-1].time for track in tracks)
        self.end = max(last_track_end, final_activity_end or last_track_end) + edge_buffer
        intervals = sorted(
            ((track.points[0].time, track.points[-1].time) for track in tracks),
            key=lambda item: item[0],
        )
        merged: list[list[datetime]] = []
        for start, end in intervals:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)

        threshold = timedelta(days=float(cfg["fast_forward_gap_days"]))
        buffer = timedelta(hours=float(cfg["fast_forward_buffer_hours"]))
        multiplier = float(cfg["fast_forward_multiplier"])
        fast_ranges: list[tuple[datetime, datetime]] = []
        for previous, following in zip(merged, merged[1:]):
            if following[0] - previous[1] > threshold:
                fast_start = previous[1] + buffer
                fast_end = following[0] - buffer
                if fast_end > fast_start:
                    fast_ranges.append((fast_start, fast_end))

        pieces: list[tuple[datetime, datetime, float]] = []
        cursor = self.start
        for fast_start, fast_end in fast_ranges:
            if fast_start > cursor:
                pieces.append((cursor, fast_start, 1.0))
            pieces.append((fast_start, fast_end, multiplier))
            cursor = fast_end
        if cursor < self.end:
            pieces.append((cursor, self.end, 1.0))

        self.segments: list[TimelineSegment] = []
        video_cursor = 0.0
        for start, end, speed in pieces:
            hours = (end - start).total_seconds() / 3600.0
            duration = hours / (self.base_hours_per_second * speed)
            self.segments.append(TimelineSegment(start, end, speed, video_cursor, duration))
            video_cursor += duration
        self.video_duration = video_cursor

    def moment_at(self, video_offset: float) -> tuple[datetime, float]:
        video_offset = min(max(video_offset, 0.0), self.video_duration)
        for segment in self.segments:
            if video_offset <= segment.video_start + segment.video_duration + 1e-9:
                elapsed = max(0.0, video_offset - segment.video_start)
                hours = elapsed * self.base_hours_per_second * segment.speed
                return min(segment.end, segment.start + timedelta(hours=hours)), segment.speed
        return self.end, 1.0

    def video_offset_for(self, moment: datetime) -> float:
        moment = min(max(moment, self.start), self.end)
        for segment in self.segments:
            if moment <= segment.end:
                hours = (moment - segment.start).total_seconds() / 3600.0
                return segment.video_start + hours / (self.base_hours_per_second * segment.speed)
        return self.video_duration


class CameraController:
    def __init__(
        self,
        timeline: SeasonTimeline,
        tracks: list[SeasonTrack],
        cfg: dict,
        aspect_ratio: float,
    ):
        self.timeline = timeline
        self.targets: dict[int, CameraRect] = {}
        for index, segment in enumerate(timeline.segments):
            if segment.speed == 1.0:
                self.targets[index] = camera_for_active_period(
                    segment, tracks, cfg, aspect_ratio
                )

    def at(self, video_offset: float) -> CameraRect:
        segments = self.timeline.segments
        current_index = len(segments) - 1
        for index, segment in enumerate(segments):
            if video_offset <= segment.video_start + segment.video_duration + 1e-9:
                current_index = index
                break
        segment = segments[current_index]
        if current_index in self.targets:
            return self.targets[current_index]

        previous_index = next(
            (index for index in range(current_index - 1, -1, -1) if index in self.targets),
            None,
        )
        next_index = next(
            (index for index in range(current_index + 1, len(segments)) if index in self.targets),
            None,
        )
        if previous_index is None:
            return self.targets[next_index]
        if next_index is None:
            return self.targets[previous_index]
        progress = (
            (video_offset - segment.video_start) / max(1e-9, segment.video_duration)
        )
        return self.targets[previous_index].interpolate(self.targets[next_index], progress)


def camera_for_active_period(
    segment: TimelineSegment,
    tracks: list[SeasonTrack],
    cfg: dict,
    aspect_ratio: float,
) -> CameraRect:
    active_points: list[TrackPoint] = []
    for track in tracks:
        if track.points[-1].time < segment.start or track.points[0].time > segment.end:
            continue
        selected = [
            point for point in track.points
            if segment.start <= point.time <= segment.end
        ]
        active_points.extend(selected or track.points)

    if not active_points:
        return CameraRect(
            float(cfg["longitude_left"]), float(cfg["longitude_right"]),
            float(cfg["latitude_bottom"]), float(cfg["latitude_top"]),
        )

    map_west, map_east = float(cfg["longitude_left"]), float(cfg["longitude_right"])
    map_south, map_north = float(cfg["latitude_bottom"]), float(cfg["latitude_top"])
    min_span = max(0.01, float(cfg.get("auto_zoom_min_span_degrees", 2.0)))
    buffer = max(0.0, float(cfg.get("auto_zoom_buffer", 0.20)))
    west = min(point.lon for point in active_points)
    east = max(point.lon for point in active_points)
    south = min(point.lat for point in active_points)
    north = max(point.lat for point in active_points)
    data_width = max(east - west, min_span)
    data_height = max(north - south, min_span)
    target_width = data_width * (1.0 + 2.0 * buffer)
    target_height = data_height * (1.0 + 2.0 * buffer)
    if target_width / target_height < aspect_ratio:
        target_width = target_height * aspect_ratio
    else:
        target_height = target_width / aspect_ratio

    # Keep a 16:9 crop inside the available map.  Near an edge, the crop is
    # shifted until it touches that edge instead of adding unavailable buffer.
    map_width, map_height = map_east - map_west, map_north - map_south
    max_scale = max(1.0, float(cfg.get("auto_zoom_max_scale", 3.0)))
    minimum_width = max(
        map_width / max_scale,
        (map_height / max_scale) * aspect_ratio,
    )
    if target_width < minimum_width:
        target_width = minimum_width
        target_height = target_width / aspect_ratio
    scale = min(1.0, map_width / target_width, map_height / target_height)
    target_width *= scale
    target_height *= scale
    center_lon = (west + east) / 2.0
    center_lat = (south + north) / 2.0
    crop_west = center_lon - target_width / 2.0
    crop_south = center_lat - target_height / 2.0
    crop_west = min(max(crop_west, map_west), map_east - target_width)
    crop_south = min(max(crop_south, map_south), map_north - target_height)
    return CameraRect(
        crop_west,
        crop_west + target_width,
        crop_south,
        crop_south + target_height,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="制作 ATCF/BTK 热带气旋路径动画")
    parser.add_argument(
        "--dat", type=Path, action="append",
        help="指定 tc_tracks 中的一个 BTK .dat；可重复使用，省略时读取其中全部 .dat",
    )
    parser.add_argument("--output", type=Path, help="覆盖 config.py 中的输出路径")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"找不到必需的配置文件：{path}")
    try:
        namespace = runpy.run_path(str(path))
    except Exception as exc:
        raise SystemExit(f"无法读取配置文件 {path.name}：{exc}") from exc
    configured = namespace.get("CONFIG")
    if not isinstance(configured, dict):
        raise SystemExit(f"{path.name} 必须定义一个 CONFIG 字典")
    return dict(configured)


def configured_asset_path(
    cfg: dict,
    key: str,
    allowed_extensions: set[str],
) -> Path:
    if key not in cfg or not str(cfg[key]).strip():
        raise SystemExit(f"config.py 缺少必需配置：{key}")
    asset_path = ROOT / str(cfg[key])
    if asset_path.suffix.lower() not in allowed_extensions:
        choices = "、".join(sorted(allowed_extensions))
        raise SystemExit(f"{key} 只支持这些格式：{choices}")
    return asset_path


def required_asset_path(
    cfg: dict,
    key: str,
    allowed_extensions: set[str],
) -> Path:
    path = configured_asset_path(cfg, key, allowed_extensions)
    if not path.is_file():
        raise SystemExit(f"找不到必需素材 {key}：{path}")
    return path


def optional_asset_path(
    cfg: dict,
    key: str,
    allowed_extensions: set[str],
) -> Path | None:
    if key not in cfg or not str(cfg[key]).strip():
        return None
    return configured_asset_path(cfg, key, allowed_extensions)


def find_dats(requested: list[Path] | None) -> list[Path]:
    tracks_root = TRACKS_DIR.resolve()
    if requested:
        paths: list[Path] = []
        for path in requested:
            candidate = (
                path.resolve()
                if path.is_absolute()
                else (TRACKS_DIR / path).resolve()
            )
            try:
                candidate.relative_to(tracks_root)
            except ValueError as exc:
                raise SystemExit(
                    f"BTK 文件只能从 tc_tracks 读取：{path}"
                ) from exc
            paths.append(candidate)
        missing = [path for path in paths if not path.exists()]
        if missing:
            raise SystemExit(f"找不到 BTK 文件：{missing[0]}")
        return paths
    files = sorted(TRACKS_DIR.glob("*.dat"))
    if not files:
        raise SystemExit("tc_tracks 中没有 .dat 文件")
    return files


def parse_coordinate(value: str) -> float:
    value = value.strip().upper()
    if len(value) < 2 or value[-1] not in "NSEW":
        raise ValueError(f"无法识别经纬度：{value!r}")
    number = float(value[:-1]) / 10.0
    return -number if value[-1] in "SW" else number


def state_for(nature: str, wind: int) -> str:
    nature = nature.strip().upper()
    if nature in {"HU", "TY", "ST"}:
        if wind >= 137:
            return "c5"
        if wind >= 113:
            return "c4"
        if wind >= 96:
            return "c3"
        if wind >= 83:
            return "c2"
        if wind >= 64:
            return "c1"
        # An inconsistent HU/TY record is safer as TS than as a false Cat 1.
        return "ts"
    return {
        "TD": "td",
        "TS": "ts",
        "SD": "sd",
        "SS": "ss",
        "EX": "ex",
        "ET": "ex",
        "PT": "ex",
        "DB": "low",
        "LO": "low",
        "WV": "low",
    }.get(nature, "low")


NUMBER_NAMES = {
    "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT",
    "NINE", "TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN",
}


def display_name(raw: str) -> str:
    raw = raw.strip()
    upper = raw.upper()
    if not raw or upper in {"UNNAMED", "NONAME", "NONE"}:
        return ""
    if upper == "INVEST" or upper.startswith("GENESIS"):
        return ""
    return raw


def parse_btk(path: Path, cfg: dict) -> list[TrackPoint]:
    by_time: dict[datetime, TrackPoint] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            fields = [item.strip() for item in line.split(",")]
            try:
                moment = datetime.strptime(fields[2], "%Y%m%d%H")
                lat = parse_coordinate(fields[6])
                lon = parse_coordinate(fields[7])
                wind = int(fields[8] or 0)
                nature = fields[10].upper()
                raw_name = fields[27] if len(fields) > 27 else ""
            except (IndexError, ValueError) as exc:
                print(f"警告：跳过第 {line_no} 行：{exc}", file=sys.stderr)
                continue
            candidate = TrackPoint(
                moment, lat, lon, wind, nature, state_for(nature, wind),
                raw_name, display_name(raw_name),
            )
            previous = by_time.get(moment)
            if previous is None or (not previous.label and candidate.label):
                by_time[moment] = candidate

    points = sorted(by_time.values(), key=lambda point: point.time)
    if not points:
        raise SystemExit(f"{path.name} 中没有可用的 BEST TRACK 记录")

    if cfg.get("promote_name_at_ts", True):
        max_ahead = timedelta(hours=float(cfg.get("name_lookahead_hours", 6)))
        for i, point in enumerate(points):
            if point.state not in {"ts", "c1", "c2", "c3", "c4", "c5"}:
                continue
            if point.raw_name.strip().upper() not in NUMBER_NAMES:
                continue
            for future in points[i + 1:]:
                if future.time - point.time > max_ahead:
                    break
                future_name = future.raw_name.strip()
                future_upper = future_name.upper()
                if future_upper and future_upper not in NUMBER_NAMES and future_upper != "INVEST":
                    point.label = display_name(future_name)
                    break

    # Missing names in intermediate/non-synoptic reports mean "unchanged",
    # not "remove the label". Only a new non-empty label replaces the current
    # one; final disappearance is handled by the storm's normal fade-out.
    current_label = ""
    for point in points:
        if point.label:
            current_label = point.label
        else:
            point.label = current_label
    return points


def discover_materials() -> dict[str, Path]:
    material_dir = ROOT / "tc_icons"
    files = {path.stem.lower(): path for path in material_dir.glob("*.mp4")}
    return {state: files[state] for state in TC_ICON_STATES if state in files}


def discover_landfall_materials() -> dict[str, Path]:
    material_dir = ROOT / "landfall_icons"
    files = {path.stem.lower(): path for path in material_dir.glob("*.mp4")}
    return {
        state: files[state]
        for state in ("td", "ts", "c1", "c2", "c3", "c4", "c5")
        if state in files
    }


def probe_clip(state: str, path: Path, cfg: dict) -> ClipInfo:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = Fraction(stream.average_rate)
        frames = int(stream.frames or 0)
        if not frames:
            frames = sum(1 for _ in container.decode(stream))
        # Normalize near-integer frame rates to avoid gcd_fraction collapsing
        # to a microsecond-scale value when materials have slightly different
        # reported rates (e.g. 30 vs 269500000/8983333).  This must happen
        # *before* the trailing-frame normalisation so both steps see the same
        # integer rate.
        integer_rate = max(1, round(float(rate)))
        rate_close = abs(float(rate) - integer_rate) <= float(
            cfg.get("normalize_fps_tolerance", 0.05)
        )
        if rate_close and rate != Fraction(integer_rate, 1):
            print(
                f"帧率规范化：{path.name} {float(rate):g} -> {integer_rate} fps"
            )
            rate = Fraction(integer_rate, 1)
        raw_duration = Fraction(frames, 1) / rate
        if cfg.get("normalize_single_trailing_frame", True):
            integer_seconds = max(1, round(float(raw_duration)))
            expected_frames = integer_rate * integer_seconds
            duration_close = abs(float(raw_duration) - integer_seconds) <= float(
                cfg.get("normalize_duration_tolerance", 0.10)
            )
            # Accept both one extra and one missing frame (e.g. 539 vs 540 at
            # 30 fps / 18 s).  Some exporters drop the last frame or add a
            # duplicate; either way we want all materials to share the same
            # integer-second duration so gcd_fraction produces a sane loop
            # period instead of collapsing to a microsecond value.
            if rate_close and duration_close and abs(frames - expected_frames) == 1:
                direction = "补齐" if frames < expected_frames else "裁剪"
                print(
                    f"素材尾帧{direction}：{path.name} {frames} -> {expected_frames} 帧，"
                    f"{float(rate):g} fps / {integer_seconds} 秒"
                )
                frames = expected_frames
                rate = Fraction(integer_rate, 1)
        duration = Fraction(frames, 1) / rate
        return ClipInfo(state, path, stream.width, stream.height, rate, frames, duration)


def probe_timeline_clip(path: Path) -> TimelineClipInfo:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        rate = Fraction(stream.average_rate)
        frames = int(stream.frames or 0)
        duration = (
            frames / float(rate)
            if frames
            else float(stream.duration * stream.time_base)
        )
        return TimelineClipInfo(
            path, stream.width, stream.height, rate, frames, duration
        )


def timeline_source_seconds(moment: datetime, cfg: dict) -> float:
    """Map a simulation date to its position in the reusable calendar clip."""
    leap = bool(cfg.get("timeline_is_leap_year", True))
    month_days = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    day = moment.day
    if moment.month == 2 and day == 29 and not leap:
        # A non-leap timeline has no Feb 29 slot; map it to the Mar 1 boundary.
        day_index = sum(month_days[:2])
        day_fraction = 0.0
    else:
        day_index = sum(month_days[: moment.month - 1]) + day - 1
        day_fraction = (
            moment.hour * 3600
            + moment.minute * 60
            + moment.second
            + moment.microsecond / 1_000_000
        ) / 86400.0
    return (day_index + day_fraction) * float(cfg.get("timeline_sec_per_day", 4.0))


def ensure_timeline_proxy(
    path: Path,
    cfg: dict,
    calendar_start: float,
    calendar_end: float,
) -> TimelineClipInfo:
    """Crop the used dates and resize once to the exact on-screen size."""
    source = probe_timeline_clip(path)
    scale = max(0.01, float(cfg.get("timeline_scale", 0.25)))
    target_width = max(2, int(round(source.width * scale / 2.0)) * 2)
    target_height = max(2, int(round(source.height * scale / 2.0)) * 2)

    available_frames = max(
        1, source.frames or int(math.floor(source.duration * float(source.fps)))
    )
    if calendar_end < calendar_start:
        # A season crossing New Year needs both ends of the calendar asset.
        calendar_start, calendar_end = 0.0, source.duration
    first_frame = max(
        0,
        min(available_frames - 1, int(math.floor(calendar_start * float(source.fps)))),
    )
    last_frame = max(
        first_frame,
        min(available_frames - 1, int(math.ceil(calendar_end * float(source.fps)))),
    )
    trim_start = first_frame / float(source.fps)
    trim_end = last_frame / float(source.fps)
    expected_frames = last_frame - first_frame + 1

    crf = int(cfg.get("timeline_cache_crf", 20))
    GENERATION_CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = GENERATION_CACHE / (
        f"timeline_cache_{target_width}x{target_height}_"
        f"f{first_frame}-{last_frame}_crf{crf}.mp4"
    )
    if cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
        cached = probe_timeline_clip(cache_path)
        if (
            (cached.width, cached.height) == (target_width, target_height)
            and abs(cached.frames - expected_frames) <= 1
        ):
            cached.calendar_start = trim_start
            return cached

    temporary = cache_path.with_suffix(".tmp.mp4")
    print(
        f"Preprocessing timeline {path.name}: {source.width}x{source.height} -> "
        f"{target_width}x{target_height}, source {trim_start:g}s-{trim_end:g}s"
    )
    with av.open(str(path)) as input_container, av.open(
        str(temporary), mode="w", options={"movflags": "+faststart"}
    ) as output_container:
        input_stream = input_container.streams.video[0]
        try:
            output_stream = output_container.add_stream("libx264", rate=source.fps)
        except Exception:
            output_stream = output_container.add_stream("h264", rate=source.fps)
        output_stream.width = target_width
        output_stream.height = target_height
        output_stream.pix_fmt = "yuv420p"
        output_stream.options = {"crf": str(crf), "preset": "veryfast"}
        time_base = Fraction(source.fps.denominator, source.fps.numerator)
        progress_interval = max(1, round(float(source.fps)))
        processed_frames = 0
        seek_start = max(0.0, trim_start - 1.0)
        input_container.seek(
            int(seek_start / float(input_stream.time_base)),
            stream=input_stream,
            backward=True,
            any_frame=False,
        )
        for frame in input_container.decode(input_stream):
            frame_time = (
                float(frame.pts * frame.time_base)
                if frame.pts is not None
                else 0.0
            )
            if frame_time + 0.5 / float(source.fps) < trim_start:
                continue
            if frame_time - 0.5 / float(source.fps) > trim_end:
                break
            resized = frame.reformat(
                width=target_width, height=target_height, format="yuv420p"
            )
            resized.pts = processed_frames
            resized.time_base = time_base
            for packet in output_stream.encode(resized):
                output_container.mux(packet)
            processed_frames += 1
            if (
                processed_frames == 1
                or processed_frames % progress_interval == 0
                or processed_frames == expected_frames
            ):
                percent = min(100.0, processed_frames * 100.0 / expected_frames)
                progress = (
                    f"\rTimeline preprocessing: {processed_frames}/{expected_frames} "
                    f"frames ({percent:5.1f}%)"
                )
                print(progress, end="", flush=True)
            if processed_frames >= expected_frames:
                break
        for packet in output_stream.encode():
            output_container.mux(packet)
        if processed_frames:
            print()
    temporary.replace(cache_path)
    cached = probe_timeline_clip(cache_path)
    cached.calendar_start = trim_start
    return cached


class TimelineReader:
    """Seek once, then stream timeline frames as the simulation advances."""

    def __init__(self, info: TimelineClipInfo, cfg: dict):
        self.info = info
        self.cfg = cfg
        self.container = av.open(str(info.path))
        self.stream = self.container.streams.video[0]
        self.decoder = None
        self.current_time = -1.0
        self.last_requested = -1.0
        self.last_rgb: np.ndarray | None = None
        self.last_keyed: np.ndarray | None = None

    def close(self) -> None:
        self.container.close()

    def _seek(self, seconds: float) -> None:
        start = max(0.0, seconds - 0.5)
        timestamp = int(start / float(self.stream.time_base))
        self.container.seek(timestamp, stream=self.stream, backward=True, any_frame=False)
        self.decoder = self.container.decode(self.stream)
        self.current_time = -1.0

    def frame_at(self, seconds: float) -> np.ndarray:
        seconds -= self.info.calendar_start
        frame_step = 1.0 / float(self.info.fps)
        seconds = min(max(0.0, seconds), max(0.0, self.info.duration - frame_step))
        # Quantize to the source frame clock.  At 60 fps output a 30 fps
        # timeline frame is reused twice instead of decoding ahead twice.
        seconds = math.floor(seconds / frame_step + 1e-9) * frame_step
        if (
            self.last_keyed is not None
            and abs(seconds - self.last_requested) < frame_step * 1e-6
        ):
            return self.last_keyed
        threshold = float(self.cfg.get("timeline_seek_threshold_seconds", 2.0))
        if (
            self.decoder is None
            or seconds + frame_step < self.current_time
            or (
                self.last_requested >= 0.0
                and seconds - self.last_requested > threshold
            )
        ):
            self._seek(seconds)

        assert self.decoder is not None
        for frame in self.decoder:
            frame_time = float(frame.pts * frame.time_base) if frame.pts is not None else 0.0
            self.current_time = frame_time
            rgb = frame.to_ndarray(format="rgb24")
            self.last_rgb = rgb
            if frame_time + frame_step * 0.5 >= seconds:
                break
        if self.last_rgb is None:
            raise RuntimeError(f"Unable to decode timeline frame at {seconds:g}s")
        self.last_requested = seconds

        # Key dark pixels by luminance, not by maximum RGB channel.  This also
        # removes blue-tinted compression noise around a nominally black matte.
        rgb_f = self.last_rgb.astype(np.float32)
        luminance = (
            rgb_f[..., 0] * 0.2126
            + rgb_f[..., 1] * 0.7152
            + rgb_f[..., 2] * 0.0722
        )
        floor = max(0.0, float(self.cfg.get("timeline_black_floor", 16)))
        if floor > 0.0:
            key = np.clip((luminance - floor) / floor, 0.0, 1.0)
            key = key * key * (3.0 - 2.0 * key)
            rgb_f *= key[..., None]
        self.last_keyed = np.clip(rgb_f, 0, 255).astype(np.uint8)
        return self.last_keyed


class IntroReader:
    """Sequentially decode the configured intro and center-crop to cover."""

    def __init__(self, path: Path, width: int, height: int, duration: float):
        self.info = probe_timeline_clip(path)
        if self.info.duration + 1e-6 < duration:
            raise SystemExit(
                f"{path.name} 只有 {self.info.duration:g} 秒，片头需要 {duration:g} 秒"
            )
        self.width = width
        self.height = height
        scale = max(width / self.info.width, height / self.info.height)
        self.scaled_width = max(width, int(math.ceil(self.info.width * scale)))
        self.scaled_height = max(height, int(math.ceil(self.info.height * scale)))
        self.crop_x = (self.scaled_width - width) // 2
        self.crop_y = (self.scaled_height - height) // 2
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        self.decoder = self.container.decode(self.stream)
        self.last_requested = -1.0
        self.last_frame: np.ndarray | None = None

    def close(self) -> None:
        self.container.close()

    def frame_at(self, seconds: float) -> np.ndarray:
        frame_step = 1.0 / float(self.info.fps)
        seconds = math.floor(max(0.0, seconds) / frame_step + 1e-9) * frame_step
        if (
            self.last_frame is not None
            and abs(seconds - self.last_requested) < frame_step * 1e-6
        ):
            return self.last_frame
        for frame in self.decoder:
            frame_time = (
                float(frame.pts * frame.time_base)
                if frame.pts is not None
                else self.last_requested + frame_step
            )
            if frame_time + frame_step * 0.5 < seconds:
                continue
            scaled = frame.reformat(
                width=self.scaled_width,
                height=self.scaled_height,
                format="rgb24",
            ).to_ndarray()
            self.last_frame = scaled[
                self.crop_y:self.crop_y + self.height,
                self.crop_x:self.crop_x + self.width,
            ].copy()
            self.last_requested = seconds
            return self.last_frame
        if self.last_frame is None:
            raise RuntimeError(f"无法解码 {self.info.path.name} 的 {seconds:g} 秒")
        return self.last_frame


def ease_out_back(value: float) -> float:
    value = max(0.0, min(1.0, value))
    overshoot = 1.70158
    shifted = value - 1.0
    return 1.0 + (overshoot + 1.0) * shifted**3 + overshoot * shifted**2


def shifted_mask(mask: Image.Image, offset_x: int, offset_y: int) -> Image.Image:
    shifted = Image.new("L", mask.size, 0)
    shifted.paste(mask, (offset_x, offset_y))
    return shifted


def render_intro_title(
    canvas: Image.Image,
    seconds: float,
    opacity: float,
    cfg: dict,
    font_file: str | None,
    font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont],
) -> Image.Image:
    text = str(cfg["intro_title"])
    start = float(cfg["intro_title_pop_start"])
    duration = max(0.001, float(cfg["intro_title_pop_duration"]))
    progress = max(0.0, min(1.0, (seconds - start) / duration))
    if progress <= 0.0 or opacity <= 0.0 or not text:
        return canvas

    start_scale = max(0.01, float(cfg["intro_title_start_scale"]))
    scale = start_scale + (1.0 - start_scale) * ease_out_back(progress)
    font_size = max(1, int(round(float(cfg["intro_title_size"]) * scale)))
    if font_size not in font_cache:
        font_cache[font_size] = load_font(
            font_size, bold=True, font_file=font_file
        )
    font = font_cache[font_size]
    stroke = max(0, int(round(float(cfg["intro_title_stroke_width"]) * scale)))
    spacing = int(round(float(cfg["intro_title_line_spacing"]) * scale))
    lines = text.splitlines()

    measure = ImageDraw.Draw(Image.new("L", (1, 1), 0))
    metrics: list[tuple[str, tuple[int, int, int, int]]] = [
        (line, measure.textbbox((0, 0), line, font=font, stroke_width=stroke))
        for line in lines
    ]
    line_heights = [max(1, box[3] - box[1]) for _, box in metrics]
    total_height = sum(line_heights) + spacing * max(0, len(lines) - 1)
    top = canvas.height * float(cfg["intro_title_center_y"]) - total_height / 2.0

    fill_mask = Image.new("L", canvas.size, 0)
    fill_draw = ImageDraw.Draw(fill_mask)
    stroke_mask = Image.new("L", canvas.size, 0)
    stroke_draw = ImageDraw.Draw(stroke_mask)
    positions: list[tuple[str, float, float]] = []
    cursor_y = top
    for (line, box), line_height in zip(metrics, line_heights):
        x = canvas.width / 2.0 - (box[0] + box[2]) / 2.0
        y = cursor_y - box[1]
        positions.append((line, x, y))
        fill_draw.text((x, y), line, font=font, fill=255)
        stroke_draw.text(
            (x, y), line, font=font, fill=255,
            stroke_width=stroke, stroke_fill=255,
        )
        cursor_y += line_height + spacing

    title_opacity = smoothstep(progress) * max(0.0, min(1.0, opacity))
    shadow_alpha = float(cfg["intro_title_shadow_opacity"]) * title_opacity
    shadow_mask = shifted_mask(
        stroke_mask,
        int(round(float(cfg["intro_title_shadow_offset_x"]) * scale)),
        int(round(float(cfg["intro_title_shadow_offset_y"]) * scale)),
    ).filter(
        ImageFilter.GaussianBlur(
            max(0.0, float(cfg["intro_title_shadow_blur"]) * scale)
        )
    )
    shadow_mask = shadow_mask.point(
        lambda value: int(value * max(0.0, min(1.0, shadow_alpha)))
    )
    shadow_colour = ImageColor.getrgb(str(cfg["intro_title_shadow_color"]))
    shadow_layer = Image.new("RGBA", canvas.size, shadow_colour + (0,))
    shadow_layer.putalpha(shadow_mask)
    result = Image.alpha_composite(canvas.convert("RGBA"), shadow_layer)

    title_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    title_draw = ImageDraw.Draw(title_layer)
    fill_colour = ImageColor.getrgb(str(cfg["intro_title_color"]))
    stroke_colour = ImageColor.getrgb(str(cfg["intro_title_stroke_color"]))
    alpha = int(round(255 * title_opacity))
    for line, x, y in positions:
        title_draw.text(
            (x, y), line, font=font, fill=fill_colour + (alpha,),
            stroke_width=stroke, stroke_fill=stroke_colour + (alpha,),
        )
    result = Image.alpha_composite(result, title_layer)

    # Raised bevel: dark inner edge at bottom-right, bright inner edge at
    # top-left. Reversing the old dark edge direction changes the perceived
    # depth from inset to embossed/convex.
    inner_shadow_blur = fill_mask.filter(
        ImageFilter.GaussianBlur(
            max(0.0, float(cfg["intro_title_inner_shadow_blur"]) * scale)
        )
    )
    inner_shadow_shifted = shifted_mask(
        inner_shadow_blur,
        -int(round(float(cfg["intro_title_inner_shadow_offset_x"]) * scale)),
        -int(round(float(cfg["intro_title_inner_shadow_offset_y"]) * scale)),
    )
    inner_shadow_mask = ImageChops.multiply(
        fill_mask, ImageChops.invert(inner_shadow_shifted)
    ).point(
        lambda value: int(
            value
            * max(
                0.0,
                min(1.0, float(cfg["intro_title_inner_shadow_opacity"])),
            )
            * title_opacity
        )
    )
    inner_shadow_colour = ImageColor.getrgb(
        str(cfg["intro_title_inner_shadow_color"])
    )
    inner_shadow_layer = Image.new(
        "RGBA", canvas.size, inner_shadow_colour + (0,)
    )
    inner_shadow_layer.putalpha(inner_shadow_mask)
    result = Image.alpha_composite(result, inner_shadow_layer)

    inner_highlight_blur = fill_mask.filter(
        ImageFilter.GaussianBlur(
            max(0.0, float(cfg["intro_title_inner_highlight_blur"]) * scale)
        )
    )
    inner_highlight_shifted = shifted_mask(
        inner_highlight_blur,
        int(round(float(cfg["intro_title_inner_highlight_offset_x"]) * scale)),
        int(round(float(cfg["intro_title_inner_highlight_offset_y"]) * scale)),
    )
    inner_highlight_mask = ImageChops.multiply(
        fill_mask, ImageChops.invert(inner_highlight_shifted)
    ).point(
        lambda value: int(
            value
            * max(
                0.0,
                min(1.0, float(cfg["intro_title_inner_highlight_opacity"])),
            )
            * title_opacity
        )
    )
    inner_highlight_colour = ImageColor.getrgb(
        str(cfg["intro_title_inner_highlight_color"])
    )
    inner_highlight_layer = Image.new(
        "RGBA", canvas.size, inner_highlight_colour + (0,)
    )
    inner_highlight_layer.putalpha(inner_highlight_mask)
    return Image.alpha_composite(result, inner_highlight_layer).convert("RGB")


def gcd_fraction(values: list[Fraction]) -> Fraction:
    denominator = reduce(math.lcm, (value.denominator for value in values), 1)
    numerators = [int(value * denominator) for value in values]
    return Fraction(reduce(gcd, numerators), denominator)


def analyse_bbox(info: ClipInfo, cycle_frames: int, threshold: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = info.width, info.height, 0, 0
    with av.open(str(info.path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            if index >= cycle_frames:
                break
            rgb = frame.to_ndarray(format="rgb24")
            mask = rgb.max(axis=2) > threshold
            ys, xs = np.nonzero(mask)
            if xs.size:
                x0, y0 = min(x0, int(xs.min())), min(y0, int(ys.min()))
                x1, y1 = max(x1, int(xs.max()) + 1), max(y1, int(ys.max()) + 1)
    if x1 <= x0 or y1 <= y0:
        return 0, 0, info.width, info.height
    side = int(math.ceil(max(x1 - x0, y1 - y0) * 1.12))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    left = max(0, min(info.width - side, cx - side // 2))
    top = max(0, min(info.height - side, cy - side // 2))
    return left, top, min(info.width, left + side), min(info.height, top + side)


def black_key(rgb: np.ndarray, low: int, high: int) -> tuple[np.ndarray, np.ndarray]:
    """Remove black plus its compressed fringe, including the centre glyphs."""
    intensity = rgb.max(axis=2).astype(np.float32)
    linear = np.clip((intensity - low) / max(1, high - low), 0.0, 1.0)
    alpha = linear * linear * (3.0 - 2.0 * linear)

    # The source clips were rendered against black, so antialiased edge RGB is
    # already darkened.  Unmatte only the feather pixels before compositing;
    # otherwise a technically transparent but visibly black rim survives.
    clean = rgb.astype(np.float32)
    partial = (alpha > 0.0) & (alpha < 1.0)
    matte_alpha = np.maximum(intensity / 255.0, 1.0 / 255.0)
    clean[partial] /= matte_alpha[partial, None]
    return np.clip(clean, 0, 255).astype(np.uint8), alpha.astype(np.float32)


def prepare_clip(
    info: ClipInfo,
    loop_period: Fraction,
    cfg: dict,
    target_size: int,
) -> PreparedClip:
    cycle_frames = max(1, int(loop_period * info.fps))
    frames_to_process = min(cycle_frames, info.frames) if info.frames else cycle_frames
    composite_mode = str(cfg.get("icon_composite_mode", "screen")).lower()
    if cfg.get("icon_preserve_source_canvas", True):
        bbox = (0, 0, info.width, info.height)
    else:
        bbox_threshold = (
            int(cfg.get("icon_glow_crop_threshold", 3))
            if composite_mode == "screen"
            else int(cfg["black_key_high"])
        )
        bbox = analyse_bbox(info, cycle_frames, bbox_threshold)
    size = max(1, int(target_size))
    GENERATION_CACHE.mkdir(parents=True, exist_ok=True)
    cache_tag = hashlib.sha256(
        repr(
            (
                info.path.stat().st_mtime_ns, frames_to_process, size, bbox,
                composite_mode, cfg.get("black_key_low"),
                cfg.get("black_key_high"), cfg.get("black_key_erode_pixels"),
            )
        ).encode("utf-8")
    ).hexdigest()[:16]
    cache_prefix = (
        f"tc_{info.path.stem.lower()}_{cache_tag}_{CACHE_SESSION}"
    )
    rgb_frames = open_session_memmap(
        GENERATION_CACHE / f"{cache_prefix}_rgb.npy",
        dtype=np.uint8,
        shape=(frames_to_process, size, size, 3),
    )
    alpha_frames = (
        np.ones((size, size), dtype=np.uint8)
        if composite_mode == "screen"
        else open_session_memmap(
            GENERATION_CACHE / f"{cache_prefix}_alpha.npy",
            dtype=np.float32,
            shape=(frames_to_process, size, size),
        )
    )
    with av.open(str(info.path)) as container:
        stream = container.streams.video[0]
        progress_interval = max(1, round(float(info.fps)))
        processed_frames = 0
        for index, frame in enumerate(container.decode(stream)):
            if index >= frames_to_process:
                break
            image = frame.to_image().crop(bbox).resize((size, size), Image.Resampling.LANCZOS)
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if composite_mode != "screen":
                rgb, alpha = black_key(
                    rgb, int(cfg["black_key_low"]), int(cfg["black_key_high"])
                )
                erode = max(0, int(cfg.get("black_key_erode_pixels", 0)))
                if erode:
                    alpha_image = Image.fromarray(
                        np.clip(alpha * 255.0, 0, 255).astype(np.uint8), mode="L"
                    ).filter(ImageFilter.MinFilter(erode * 2 + 1))
                    alpha = np.asarray(alpha_image, dtype=np.float32) / 255.0
                alpha_frames[index] = alpha
            rgb_frames[index] = rgb
            processed_frames = index + 1
            if (
                processed_frames == 1
                or processed_frames % progress_interval == 0
                or processed_frames == frames_to_process
            ):
                percent = min(
                    100.0, processed_frames * 100.0 / frames_to_process
                )
                print(
                    f"\rTC preprocessing {info.path.name}: "
                    f"{processed_frames}/{frames_to_process} frames "
                    f"({percent:5.1f}%)",
                    end="",
                    flush=True,
                )
        if processed_frames:
            print()
    if not processed_frames:
        raise RuntimeError(f"无法解码素材：{info.path.name}")
    if processed_frames < frames_to_process:
        rgb_frames[processed_frames:frames_to_process] = rgb_frames[
            processed_frames - 1
        ]
        if alpha_frames.ndim == 3:
            alpha_frames[processed_frames:frames_to_process] = alpha_frames[
                processed_frames - 1
            ]
        print(
            f"素材尾帧补齐：{info.path.name} "
            f"{processed_frames} -> {frames_to_process} 帧"
        )
    rgb_frames.flush()
    if isinstance(alpha_frames, np.memmap):
        alpha_frames.flush()
    return PreparedClip(info, rgb_frames, alpha_frames)


def prepare_landfall_clip(
    info: ClipInfo,
    cfg: dict,
    target_width: int,
) -> PreparedClip:
    """Load a complete one-shot landfall effect and key its black matte."""
    composite_mode = str(
        cfg.get("landfall_icon_composite_mode", "screen")
    ).lower()
    target_width = max(1, int(target_width))
    target_height = max(
        1, int(round(target_width * info.height / max(1, info.width)))
    )
    total_frames = max(1, info.frames)
    GENERATION_CACHE.mkdir(parents=True, exist_ok=True)
    cache_tag = hashlib.sha256(
        repr(
            (
                info.path.stat().st_mtime_ns, total_frames,
                target_width, target_height, composite_mode,
                cfg.get("landfall_icon_black_floor"),
                cfg.get("black_key_low"), cfg.get("black_key_high"),
            )
        ).encode("utf-8")
    ).hexdigest()[:16]
    cache_prefix = (
        f"landfall_{info.path.stem.lower()}_{cache_tag}_{CACHE_SESSION}"
    )
    rgb_frames = open_session_memmap(
        GENERATION_CACHE / f"{cache_prefix}_rgb.npy",
        dtype=np.uint8,
        shape=(total_frames, target_height, target_width, 3),
    )
    alpha_frames = (
        np.ones((target_height, target_width), dtype=np.uint8)
        if composite_mode == "screen"
        else open_session_memmap(
            GENERATION_CACHE / f"{cache_prefix}_alpha.npy",
            dtype=np.float32,
            shape=(total_frames, target_height, target_width),
        )
    )
    processed_frames = 0
    with av.open(str(info.path)) as container:
        stream = container.streams.video[0]
        progress_interval = max(1, round(float(info.fps)))
        for index, frame in enumerate(container.decode(stream)):
            if index >= total_frames:
                break
            image = frame.to_image()
            # A 1280x720 landfall material rendered at 1280x720 must remain an
            # exact decoded frame.  Avoid even a nominal same-size resampling;
            # it is unnecessary and makes the no-upscale guarantee explicit.
            if image.size != (target_width, target_height):
                image = image.resize(
                    (target_width, target_height), Image.Resampling.LANCZOS
                )
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if composite_mode == "screen":
                rgb_f = rgb.astype(np.float32)
                luminance = (
                    rgb_f[..., 0] * 0.2126
                    + rgb_f[..., 1] * 0.7152
                    + rgb_f[..., 2] * 0.0722
                )
                floor = max(
                    0.0, float(cfg.get("landfall_icon_black_floor", 16))
                )
                if floor > 0.0:
                    key = np.clip((luminance - floor) / floor, 0.0, 1.0)
                    key = key * key * (3.0 - 2.0 * key)
                    rgb_f *= key[..., None]
                    rgb = np.clip(rgb_f, 0, 255).astype(np.uint8)
            else:
                rgb, alpha = black_key(
                    rgb, int(cfg["black_key_low"]), int(cfg["black_key_high"])
                )
                alpha_frames[index] = alpha
            rgb_frames[index] = rgb
            processed = index + 1
            processed_frames = processed
            if (
                processed == 1
                or processed % progress_interval == 0
                or processed == total_frames
            ):
                percent = min(100.0, processed * 100.0 / total_frames)
                print(
                    f"\rLandfall preprocessing {info.path.name}: "
                    f"{processed}/{total_frames} frames ({percent:5.1f}%)",
                    end="",
                    flush=True,
                )
        if processed_frames:
            print()
    if not processed_frames:
        raise RuntimeError(f"无法解码登陆素材：{info.path.name}")
    if processed_frames < total_frames:
        rgb_frames[processed_frames:total_frames] = rgb_frames[
            processed_frames - 1
        ]
        if alpha_frames.ndim == 3:
            alpha_frames[processed_frames:total_frames] = alpha_frames[
                processed_frames - 1
            ]
        print(
            f"登陆素材尾帧补齐：{info.path.name} "
            f"{processed_frames} -> {total_frames} 帧"
        )
    rgb_frames.flush()
    if isinstance(alpha_frames, np.memmap):
        alpha_frames.flush()
    return PreparedClip(info, rgb_frames, alpha_frames)


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def ease_out_cubic(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return 1.0 - (1.0 - value) ** 3


def ease_in_cubic(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value**3


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def project(lat: float, lon: float, cfg: dict) -> tuple[float, float]:
    west = float(cfg.get("_camera_west", cfg["longitude_left"]))
    east = float(cfg.get("_camera_east", cfg["longitude_right"]))
    south = float(cfg.get("_camera_south", cfg["latitude_bottom"]))
    north = float(cfg.get("_camera_north", cfg["latitude_top"]))
    x = (lon - west) / (east - west)
    y = (north - lat) / (north - south)
    viewport_x = float(cfg.get("_map_viewport_x", 0.0))
    viewport_y = float(cfg.get("_map_viewport_y", 0.0))
    viewport_w = float(cfg.get("_map_viewport_width", cfg["width"]))
    viewport_h = float(cfg.get("_map_viewport_height", cfg["height"]))
    return viewport_x + x * viewport_w, viewport_y + y * viewport_h


def camera_zoom_factor(camera: CameraRect, cfg: dict) -> float:
    map_width = max(1e-9, float(cfg["longitude_right"]) - float(cfg["longitude_left"]))
    map_height = max(1e-9, float(cfg["latitude_top"]) - float(cfg["latitude_bottom"]))
    zoom = max(
        map_width / max(1e-9, camera.east - camera.west),
        map_height / max(1e-9, camera.north - camera.south),
    )
    return min(
        max(1.0, zoom),
        max(1.0, float(cfg.get("auto_zoom_max_scale", 3.0))),
    )


def set_camera(cfg: dict, camera: CameraRect) -> None:
    cfg["_camera_west"], cfg["_camera_east"] = camera.west, camera.east
    cfg["_camera_south"], cfg["_camera_north"] = camera.south, camera.north
    cfg["_map_viewport_x"], cfg["_map_viewport_y"] = 0, 0
    cfg["_map_viewport_width"] = int(cfg["width"])
    cfg["_map_viewport_height"] = int(cfg["height"])
    cfg["_map_zoom"] = camera_zoom_factor(camera, cfg)


def magnification_factor(cfg: dict, coefficient_key: str) -> float:
    zoom = max(1.0, float(cfg.get("_map_zoom", 1.0)))
    coefficient = float(cfg.get(coefficient_key, 0.0))
    return max(0.01, 1.0 + (zoom - 1.0) * coefficient)


def resize_icon_frame(
    frame: tuple[np.ndarray, np.ndarray],
    size: int | tuple[int, int],
    composite_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    rgb, alpha = frame
    if isinstance(size, tuple):
        target_width = max(1, int(size[0]))
        target_height = max(1, int(size[1]))
    else:
        target_width = target_height = max(1, int(size))
    if rgb.shape[:2] == (target_height, target_width):
        return rgb, alpha
    resized_rgb = np.asarray(
        Image.fromarray(rgb, "RGB").resize(
            (target_width, target_height), Image.Resampling.LANCZOS
        ),
        dtype=np.uint8,
    )
    if composite_mode == "screen":
        resized_alpha = np.ones((target_height, target_width), dtype=np.uint8)
    else:
        alpha_image = Image.fromarray(
            np.clip(alpha * 255.0, 0, 255).astype(np.uint8), "L"
        ).resize((target_width, target_height), Image.Resampling.LANCZOS)
        resized_alpha = np.asarray(alpha_image, dtype=np.float32) / 255.0
    return resized_rgb, resized_alpha


def render_camera_map(source_map: Image.Image, camera: CameraRect, cfg: dict) -> Image.Image:
    map_west, map_east = float(cfg["longitude_left"]), float(cfg["longitude_right"])
    map_south, map_north = float(cfg["latitude_bottom"]), float(cfg["latitude_top"])
    source_w, source_h = source_map.size
    left = (camera.west - map_west) / (map_east - map_west) * source_w
    right = (camera.east - map_west) / (map_east - map_west) * source_w
    top = (map_north - camera.north) / (map_north - map_south) * source_h
    bottom = (map_north - camera.south) / (map_north - map_south) * source_h
    return source_map.resize(
        (int(cfg["width"]), int(cfg["height"])),
        Image.Resampling.LANCZOS,
        box=(left, top, right, bottom),
    )


def _interpolate_track_segment_raw(
    points: list[TrackPoint], segment: int, u: float, cfg: dict
) -> tuple[float, float]:
    """Evaluate one known Hermite segment at its raw polynomial parameter."""
    before, after = points[segment], points[segment + 1]
    segment_seconds = (after.time - before.time).total_seconds()
    u = max(0.0, min(1.0, u))
    tension_scale = 1.0 - max(
        0.0, min(1.0, float(cfg.get("path_smoothing_tension", 0.15)))
    )

    def tangent(index: int, attribute: str) -> float:
        if index == 0:
            first, second = points[0], points[1]
        elif index == len(points) - 1:
            first, second = points[-2], points[-1]
        else:
            first, second = points[index - 1], points[index + 1]
        seconds = max(1.0, (second.time - first.time).total_seconds())
        return (getattr(second, attribute) - getattr(first, attribute)) / seconds * tension_scale

    u2, u3 = u * u, u * u * u
    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2

    def interpolate(attribute: str) -> float:
        value = (
            h00 * getattr(before, attribute)
            + h10 * segment_seconds * tangent(segment, attribute)
            + h01 * getattr(after, attribute)
            + h11 * segment_seconds * tangent(segment + 1, attribute)
        )
        neighbourhood = points[max(0, segment - 1):min(len(points), segment + 3)]
        lower = min(getattr(point, attribute) for point in neighbourhood)
        upper = max(getattr(point, attribute) for point in neighbourhood)
        return max(lower, min(upper, value))

    return interpolate("lat"), interpolate("lon")


_TRACK_ARC_CACHE: dict[
    tuple[int, int, datetime, datetime, int, float, int],
    tuple[np.ndarray, np.ndarray],
] = {}


def interpolate_track_segment(
    points: list[TrackPoint], segment: int, fraction: float, cfg: dict
) -> tuple[float, float]:
    """Evaluate a smoothed segment at a constant-distance time fraction."""
    fraction = max(0.0, min(1.0, fraction))
    if fraction <= 0.0:
        return points[segment].lat, points[segment].lon
    if fraction >= 1.0:
        return points[segment + 1].lat, points[segment + 1].lon

    tension = float(cfg.get("path_smoothing_tension", 0.15))
    sample_count = max(
        24, int(cfg.get("path_samples_per_segment", 12)) * 4
    )
    key = (
        id(points), len(points), points[0].time, points[-1].time,
        segment, round(tension, 9), sample_count,
    )
    cached = _TRACK_ARC_CACHE.get(key)
    if cached is None:
        parameters = np.linspace(0.0, 1.0, sample_count + 1, dtype=np.float64)
        positions = [
            _interpolate_track_segment_raw(points, segment, float(u), cfg)
            for u in parameters
        ]
        lengths = np.zeros(sample_count + 1, dtype=np.float64)
        for index, ((lat0, lon0), (lat1, lon1)) in enumerate(
            zip(positions, positions[1:]), 1
        ):
            mean_latitude = math.radians((lat0 + lat1) / 2.0)
            dx = (lon1 - lon0) * math.cos(mean_latitude)
            dy = lat1 - lat0
            lengths[index] = lengths[index - 1] + math.hypot(dx, dy)
        if lengths[-1] <= 1e-12:
            normalized_lengths = parameters.copy()
        else:
            normalized_lengths = lengths / lengths[-1]
        cached = parameters, normalized_lengths
        _TRACK_ARC_CACHE[key] = cached

    parameters, normalized_lengths = cached
    upper = int(np.searchsorted(normalized_lengths, fraction, side="right"))
    upper = max(1, min(len(parameters) - 1, upper))
    lower = upper - 1
    span = normalized_lengths[upper] - normalized_lengths[lower]
    weight = (
        0.0 if span <= 1e-12
        else (fraction - normalized_lengths[lower]) / span
    )
    u = parameters[lower] + (parameters[upper] - parameters[lower]) * weight
    return _interpolate_track_segment_raw(points, segment, float(u), cfg)


def point_at(points: list[TrackPoint], moment: datetime, cfg: dict) -> tuple[int, float, float]:
    """Interpolate through report points with a time-aware cubic Hermite curve."""
    if moment <= points[0].time:
        return 0, points[0].lat, points[0].lon
    if moment >= points[-1].time:
        last = points[-1]
        return len(points) - 1, last.lat, last.lon

    segment = 0
    for i in range(len(points) - 1):
        if moment < points[i + 1].time:
            segment = i
            break
    before, after = points[segment], points[segment + 1]
    segment_seconds = max(1.0, (after.time - before.time).total_seconds())
    u = (moment - before.time).total_seconds() / segment_seconds
    lat, lon = interpolate_track_segment(points, segment, u, cfg)
    return segment, lat, lon


def wind_at(points: list[TrackPoint], moment: datetime) -> float:
    if moment <= points[0].time:
        return float(points[0].wind)
    if moment >= points[-1].time:
        return float(points[-1].wind)
    for index in range(len(points) - 1):
        before, after = points[index], points[index + 1]
        if moment <= after.time:
            duration = max(1.0, (after.time - before.time).total_seconds())
            weight = (moment - before.time).total_seconds() / duration
            return before.wind + (after.wind - before.wind) * weight
    return float(points[-1].wind)


def hue_shift_icon(
    frame: tuple[np.ndarray, np.ndarray],
    state: str,
    wind: float,
) -> tuple[np.ndarray, np.ndarray]:
    if state == "ts":
        # 35 kt: +30 degrees; every additional 5 kt subtracts 10 degrees.
        bounded_wind = min(60.0, max(35.0, wind))
        degrees = 30.0 - (bounded_wind - 35.0) * 2.0
    elif state == "c5":
        # 140 kt is unchanged; every additional 5 kt subtracts 5 degrees.
        degrees = -max(0.0, wind - 140.0)
    else:
        return frame
    if abs(degrees) < 1e-9:
        return frame

    rgb, alpha = frame
    hsv = np.asarray(Image.fromarray(rgb, "RGB").convert("HSV")).copy()
    hue_offset = int(round(degrees * 256.0 / 360.0))
    hsv[..., 0] = (
        hsv[..., 0].astype(np.int16) + hue_offset
    ) % 256
    shifted = np.asarray(Image.fromarray(hsv, "HSV").convert("RGB"), dtype=np.uint8)
    return shifted, alpha


def detect_landfalls(
    tracks: list[SeasonTrack],
    coastline: CoastlineIndex,
    cfg: dict,
) -> list[Landfall]:
    samples = max(
        2, int(cfg.get("landfall_detection_samples_per_segment", 120))
    )
    deduplicate = timedelta(
        minutes=max(0.0, float(cfg.get("landfall_deduplicate_minutes", 30.0)))
    )
    eligible_states = {"td", "ts", "c1", "c2", "c3", "c4", "c5"}
    detected: list[Landfall] = []
    for track in tracks:
        track_landfalls: list[Landfall] = []
        for segment_index in range(len(track.points) - 1):
            segment_start = track.points[segment_index].time
            segment_end = track.points[segment_index + 1].time
            previous_time = segment_start
            _, previous_lat, previous_lon = point_at(
                track.points, previous_time, cfg
            )
            for sample_index in range(1, samples + 1):
                fraction = sample_index / samples
                current_time = segment_start + (
                    segment_end - segment_start
                ) * fraction
                _, current_lat, current_lon = point_at(
                    track.points, current_time, cfg
                )
                crossing = coastline.landfall_intersection(
                    previous_lon,
                    previous_lat,
                    current_lon,
                    current_lat,
                )
                if crossing is not None:
                    crossing_fraction, crossing_lon, crossing_lat = crossing
                    crossing_time = previous_time + (
                        current_time - previous_time
                    ) * crossing_fraction
                    point_index, _, _ = point_at(
                        track.points, crossing_time, cfg
                    )
                    wind = wind_at(track.points, crossing_time)
                    state = state_for(
                        track.points[point_index].nature,
                        int(round(wind)),
                    )
                    if state in eligible_states:
                        event = Landfall(
                            crossing_time,
                            crossing_lat,
                            crossing_lon,
                            state,
                            int(round(wind)),
                        )
                        if (
                            not track_landfalls
                            or event.time - track_landfalls[-1].time > deduplicate
                        ):
                            track_landfalls.append(event)
                previous_time = current_time
                previous_lat, previous_lon = current_lat, current_lon
        detected.extend(track_landfalls)
    return sorted(detected, key=lambda item: item.time)


def transition_at(points: list[TrackPoint], index: int, moment: datetime, cfg: dict, attr: str):
    if attr == "state":
        transition_seconds = max(0.001, float(cfg["transition_seconds"]))
        simulated_duration = timedelta(
            hours=transition_seconds * simulation_hours_per_second(cfg)
        )
        half_duration = simulated_duration / 2
        candidate_events: list[tuple[datetime, str, str]] = []
        if index > 0 and points[index - 1].state != points[index].state:
            candidate_events.append(
                (points[index].time, points[index - 1].state, points[index].state)
            )
        if index + 1 < len(points) and points[index].state != points[index + 1].state:
            candidate_events.append(
                (points[index + 1].time, points[index].state, points[index + 1].state)
            )
        for event_time, old_state, new_state in candidate_events:
            transition_start = event_time - half_duration
            transition_end = event_time + half_duration
            if transition_start <= moment <= transition_end:
                elapsed = (moment - transition_start).total_seconds()
                duration = max(1e-9, simulated_duration.total_seconds())
                return old_state, new_state, smoothstep(elapsed / duration)
        current_state = points[index].state
        return current_state, current_state, 1.0

    current = getattr(points[index], attr)
    if index == 0:
        return current, current, 1.0
    previous = getattr(points[index - 1], attr)
    if current == previous:
        return current, current, 1.0
    video_elapsed = (
        (moment - points[index].time).total_seconds()
        / 3600.0
        / simulation_hours_per_second(cfg)
    )
    duration_key = "label_transition_seconds" if attr == "label" else "transition_seconds"
    weight = smoothstep(video_elapsed / max(0.001, float(cfg[duration_key])))
    return previous, current, weight


def sample_clip(clip: PreparedClip, phase_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    index = int(math.floor(phase_seconds * float(clip.info.fps))) % len(clip.rgb)
    alpha = clip.alpha if clip.alpha.ndim == 2 else clip.alpha[index]
    return clip.rgb[index], alpha


def align_icon_frames(
    icon: tuple[np.ndarray, np.ndarray],
    target_height: int,
    target_width: int,
) -> tuple[np.ndarray, np.ndarray]:
    rgb, alpha = icon
    if rgb.shape[:2] == (target_height, target_width):
        return rgb, alpha
    canvas_rgb = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    canvas_alpha = np.zeros((target_height, target_width), dtype=np.float32)
    top = (target_height - rgb.shape[0]) // 2
    left = (target_width - rgb.shape[1]) // 2
    canvas_rgb[top:top + rgb.shape[0], left:left + rgb.shape[1]] = rgb
    canvas_alpha[top:top + alpha.shape[0], left:left + alpha.shape[1]] = alpha
    return canvas_rgb, canvas_alpha


def mix_icons(
    old: tuple[np.ndarray, np.ndarray],
    new: tuple[np.ndarray, np.ndarray],
    weight: float,
    composite_mode: str,
):
    if weight <= 0.0:
        return old
    if weight >= 1.0:
        return new
    target_height = max(old[0].shape[0], new[0].shape[0])
    target_width = max(old[0].shape[1], new[0].shape[1])
    old = align_icon_frames(old, target_height, target_width)
    new = align_icon_frames(new, target_height, target_width)
    rgb0, alpha0 = old
    rgb1, alpha1 = new
    if composite_mode == "screen":
        rgb = np.clip(
            rgb0.astype(np.float32) * (1.0 - weight)
            + rgb1.astype(np.float32) * weight,
            0,
            255,
        ).astype(np.uint8)
        return rgb, np.ones((target_height, target_width), dtype=np.float32)
    q0 = (1.0 - weight) * alpha0
    q1 = weight * alpha1
    alpha = q0 + q1  # identical opaque masks remain 100% opaque, not 75%
    premultiplied = rgb0.astype(np.float32) * q0[..., None] + rgb1.astype(np.float32) * q1[..., None]
    rgb = np.zeros_like(rgb0)
    valid = alpha > 1e-6
    rgb[valid] = np.clip(premultiplied[valid] / alpha[valid, None], 0, 255).astype(np.uint8)
    return rgb, alpha


def paste_rgba(
    base: np.ndarray,
    rgb: np.ndarray,
    alpha: np.ndarray,
    center: tuple[float, float],
    opacity: float,
    composite_mode: str,
) -> None:
    icon_h, icon_w = alpha.shape
    left = int(round(center[0] - icon_w / 2))
    top = int(round(center[1] - icon_h / 2))
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(base.shape[1], left + icon_w), min(base.shape[0], top + icon_h)
    if x1 <= x0 or y1 <= y0:
        return
    sx0, sy0 = x0 - left, y0 - top
    sx1, sy1 = sx0 + (x1 - x0), sy0 + (y1 - y0)
    a = np.clip(alpha[sy0:sy1, sx0:sx1] * opacity, 0.0, 1.0)[..., None]
    roi = base[y0:y1, x0:x1]
    source = rgb[sy0:sy1, sx0:sx1]
    if composite_mode == "screen":
        destination_f = roi.astype(np.float32) / 255.0
        source_f = source.astype(np.float32) / 255.0 * opacity
        roi[:] = np.clip(
            (1.0 - (1.0 - destination_f) * (1.0 - source_f)) * 255.0,
            0,
            255,
        ).astype(np.uint8)
        return
    roi[:] = np.clip(source * a + roi * (1.0 - a), 0, 255).astype(np.uint8)


def load_font(size: int, bold: bool = False, font_file: str | None = None):
    if not font_file or not str(font_file).strip():
        raise RuntimeError("config.py 缺少必需配置：font_file")
    configured_path = Path(font_file)
    if not configured_path.is_absolute():
        configured_path = ROOT / configured_path
    try:
        return ImageFont.truetype(str(configured_path), size)
    except OSError as exc:
        raise RuntimeError(f"无法读取字体：{configured_path}") from exc


def track_groups(
    points: list[TrackPoint], moment: datetime, cfg: dict
) -> list[tuple[str, list[tuple[float, float]]]]:
    """Return the smoothed, colour-grouped path geometry for one frame."""
    index, _, _ = point_at(points, moment, cfg)
    samples_per_segment = max(2, int(cfg.get("path_samples_per_segment", 12)))
    groups: list[tuple[str, list[tuple[float, float]]]] = []
    previous_xy = project(points[0].lat, points[0].lon, cfg)
    segment_count = min(index + 1, len(points) - 1)
    for segment in range(segment_count):
        start_time, report_end = points[segment].time, points[segment + 1].time
        end_time = min(moment, report_end)
        if end_time <= start_time:
            continue
        fraction = (end_time - start_time).total_seconds() / (
            report_end - start_time
        ).total_seconds()
        count = max(1, int(math.ceil(samples_per_segment * fraction)))
        state = points[segment].state
        if not groups or groups[-1][0] != state:
            groups.append((state, [previous_xy]))
        group_points = groups[-1][1]
        for sample in range(1, count + 1):
            sample_fraction = fraction * sample / count
            sample_time = start_time + (report_end - start_time) * sample_fraction
            _, lat, lon = point_at(points, sample_time, cfg)
            previous_xy = project(lat, lon, cfg)
            group_points.append(previous_xy)
    return groups


def track_groups_between(
    points: list[TrackPoint],
    start: datetime,
    end: datetime,
    cfg: dict,
) -> list[tuple[str, list[tuple[float, float]]]]:
    """Return only the continuous path geometry added in ``(start, end]``."""
    if end <= start or end <= points[0].time or start >= points[-1].time:
        return []
    start = max(start, points[0].time)
    end = min(end, points[-1].time)
    samples_per_segment = max(2, int(cfg.get("path_samples_per_segment", 12)))
    groups: list[tuple[str, list[tuple[float, float]]]] = []

    for segment in range(len(points) - 1):
        segment_start = points[segment].time
        segment_end = points[segment + 1].time
        visible_start = max(start, segment_start)
        visible_end = min(end, segment_end)
        if visible_end <= visible_start:
            continue
        duration = (segment_end - segment_start).total_seconds()
        first_fraction = (
            visible_start - segment_start
        ).total_seconds() / duration
        last_fraction = (
            visible_end - segment_start
        ).total_seconds() / duration
        fraction_span = last_fraction - first_fraction
        count = max(1, int(math.ceil(samples_per_segment * fraction_span)))
        state = points[segment].state
        first_lat, first_lon = interpolate_track_segment(
            points, segment, first_fraction, cfg
        )
        first_xy = project(first_lat, first_lon, cfg)
        if not groups or groups[-1][0] != state:
            groups.append((state, [first_xy]))
        elif groups[-1][1][-1] != first_xy:
            groups[-1][1].append(first_xy)
        group_points = groups[-1][1]
        for sample in range(1, count + 1):
            fraction = first_fraction + fraction_span * sample / count
            lat, lon = interpolate_track_segment(points, segment, fraction, cfg)
            group_points.append(project(lat, lon, cfg))
    return groups


def dashed_polyline(
    points: list[tuple[float, float]],
    dash_length: float,
    gap_length: float,
) -> list[list[tuple[float, float]]]:
    """Split a polyline into equally spaced visible dash segments."""
    if len(points) < 2:
        return []
    dash_length = max(0.1, dash_length)
    gap_length = max(0.1, gap_length)
    visible = True
    remaining = dash_length
    current: list[tuple[float, float]] = [points[0]]
    result: list[list[tuple[float, float]]] = []
    cursor = points[0]

    for target in points[1:]:
        dx, dy = target[0] - cursor[0], target[1] - cursor[1]
        distance = math.hypot(dx, dy)
        while distance > 1e-9:
            step = min(distance, remaining)
            ratio = step / distance
            next_point = (
                cursor[0] + dx * ratio,
                cursor[1] + dy * ratio,
            )
            if visible:
                if not current:
                    current = [cursor]
                current.append(next_point)
            cursor = next_point
            dx, dy = target[0] - cursor[0], target[1] - cursor[1]
            distance = math.hypot(dx, dy)
            remaining -= step
            if remaining <= 1e-9:
                if visible and len(current) >= 2:
                    result.append(current)
                visible = not visible
                remaining = dash_length if visible else gap_length
                current = [cursor] if visible else []
        cursor = target

    if visible and len(current) >= 2:
        result.append(current)
    return result


def dotted_polyline(
    points: list[tuple[float, float]], spacing: float
) -> list[tuple[float, float]]:
    """Sample evenly spaced dot centres along a polyline."""
    if not points:
        return []
    spacing = max(0.1, spacing)
    result = [points[0]]
    remaining = spacing
    cursor = points[0]
    for target in points[1:]:
        dx, dy = target[0] - cursor[0], target[1] - cursor[1]
        distance = math.hypot(dx, dy)
        while distance + 1e-9 >= remaining:
            ratio = remaining / max(distance, 1e-9)
            cursor = (
                cursor[0] + dx * ratio,
                cursor[1] + dy * ratio,
            )
            result.append(cursor)
            dx, dy = target[0] - cursor[0], target[1] - cursor[1]
            distance = math.hypot(dx, dy)
            remaining = spacing
        remaining -= distance
        cursor = target
    return result


def track_style(state: str, cfg: dict) -> dict:
    """Resolve one state's path style over the global solid-line defaults."""
    configured = cfg.get("track_styles", {})
    style = configured.get(state, configured.get(state.upper(), {}))
    if not isinstance(style, dict):
        style = {}
    return {
        "pattern": str(style.get("pattern", "solid")).lower(),
        "opacity": max(
            0.0,
            min(
                1.0,
                float(style.get("opacity", cfg.get("track_opacity", 0.55))),
            ),
        ),
        "spacing_widths": max(0.1, float(style.get("spacing_widths", 2.5))),
        "dot_radius_widths": max(
            0.1, float(style.get("dot_radius_widths", 0.75))
        ),
        "dash_widths": max(0.1, float(style.get("dash_widths", 3.0))),
        "gap_widths": max(0.1, float(style.get("gap_widths", 2.0))),
    }


def draw_track(draw: ImageDraw.ImageDraw, points: list[TrackPoint], moment: datetime, cfg: dict) -> None:
    if not cfg.get("show_track", True) or moment <= points[0].time:
        return

    groups = track_groups(points, moment, cfg)

    if not groups:
        return
    configured_colours = cfg.get("track_colors", {})
    line_width = max(
        1,
        int(
            round(
                float(cfg.get("track_width", 4))
                * magnification_factor(cfg, "track_width_mag")
            )
        ),
    )
    radius = line_width / 2.0

    def colour_for(state: str, opacity: float) -> tuple[int, int, int, int]:
        colour_name = configured_colours.get(
            state, STATE_COLOURS.get(state, "#888888")
        )
        return ImageColor.getrgb(colour_name) + (int(round(255 * opacity)),)

    for group_index, (state, group_points) in enumerate(groups):
        style = track_style(state, cfg)
        pattern = style["pattern"]
        colour = colour_for(state, style["opacity"])
        if pattern == "dotted":
            dot_radius = line_width * style["dot_radius_widths"]
            for x, y in dotted_polyline(
                group_points, line_width * style["spacing_widths"]
            ):
                draw.ellipse(
                    (x - dot_radius, y - dot_radius, x + dot_radius, y + dot_radius),
                    fill=colour,
                )
        elif pattern == "dashed":
            for dash in dashed_polyline(
                group_points,
                line_width * style["dash_widths"],
                line_width * style["gap_widths"],
            ):
                draw.line(dash, fill=colour, width=line_width, joint="curve")
        elif len(group_points) >= 2:
            draw.line(group_points, fill=colour, width=line_width, joint="curve")
            endpoints: list[tuple[float, float]] = []
            if group_index == 0:
                endpoints.append(group_points[0])
            if group_index == len(groups) - 1:
                endpoints.append(group_points[-1])
            for x, y in endpoints:
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius), fill=colour
                )


def draw_track_groups_native_aa(
    layer: Image.Image,
    groups: list[tuple[str, list[tuple[float, float]]]],
    cfg: dict,
) -> Image.Image:
    """Draw grouped path geometry with OpenCV's subpixel LINE_AA."""
    if not groups:
        return layer

    line_width = max(
        1,
        int(
            round(
                float(cfg.get("track_width", 4))
                * magnification_factor(cfg, "track_width_mag")
            )
        ),
    )
    configured_colours = cfg.get("track_colors", {})
    shift = 8
    unit = 1 << shift

    def colour_for(state: str, opacity: float) -> tuple[int, int, int, int]:
        colour_name = configured_colours.get(
            state, STATE_COLOURS.get(state, "#888888")
        )
        return ImageColor.getrgb(colour_name) + (int(round(255 * opacity)),)

    def fixed_point(value: tuple[float, float]) -> tuple[int, int]:
        return int(round(value[0] * unit)), int(round(value[1] * unit))

    def distinct_neighbour(
        points: list[tuple[float, float]],
        origin: tuple[float, float],
        reverse: bool,
    ) -> tuple[float, float] | None:
        candidates = reversed(points[:-1]) if reverse else iter(points[1:])
        for candidate in candidates:
            if math.hypot(candidate[0] - origin[0], candidate[1] - origin[1]) > 1e-6:
                return candidate
        return None

    def boundary_tangent(boundary_index: int) -> tuple[float, float, float, float]:
        """Return the shared point and forward unit tangent at a state boundary."""
        previous_points = groups[boundary_index][1]
        next_points = groups[boundary_index + 1][1]
        origin = previous_points[-1]
        before = distinct_neighbour(previous_points, origin, reverse=True)
        after = distinct_neighbour(next_points, origin, reverse=False)
        if before is not None and after is not None:
            dx, dy = after[0] - before[0], after[1] - before[1]
        elif after is not None:
            dx, dy = after[0] - origin[0], after[1] - origin[1]
        elif before is not None:
            dx, dy = origin[0] - before[0], origin[1] - before[1]
        else:
            dx, dy = 1.0, 0.0
        length = max(1e-9, math.hypot(dx, dy))
        return origin[0], origin[1], dx / length, dy / length

    boundaries = [boundary_tangent(index) for index in range(len(groups) - 1)]
    pixel_x = np.arange(layer.width, dtype=np.float32)[None, :] + 0.5
    pixel_y = np.arange(layer.height, dtype=np.float32)[:, None] + 0.5

    def endpoint_radius_pixels(state: str) -> float:
        style = track_style(state, cfg)
        if style["pattern"] == "dotted":
            return line_width * style["dot_radius_widths"]
        return line_width / 2.0

    # A boundary partition is valid only inside the shared round join. Applying
    # its half-plane to a whole curved group would erase any remote part that
    # happened to loop back across that infinite plane.
    boundary_partitions: list[tuple[np.ndarray, np.ndarray]] = []
    for boundary_index, (x, y, tangent_x, tangent_y) in enumerate(boundaries):
        offset_x = pixel_x - x
        offset_y = pixel_y - y
        signed_distance = offset_x * tangent_x + offset_y * tangent_y
        join_radius = max(
            endpoint_radius_pixels(groups[boundary_index][0]),
            endpoint_radius_pixels(groups[boundary_index + 1][0]),
        ) + 2.0
        local_join = offset_x * offset_x + offset_y * offset_y <= join_radius**2
        boundary_partitions.append((local_join, signed_distance))

    storm_rgba = np.zeros((layer.height, layer.width, 4), dtype=np.uint8)
    previous_rgba: np.ndarray | None = None
    previous_colour: tuple[int, int, int, int] | None = None
    for group_index, (state, group_points) in enumerate(groups):
        style = track_style(state, cfg)
        pattern = style["pattern"]
        colour = colour_for(state, style["opacity"])
        alpha = colour[3]
        if alpha <= 0:
            continue
        rgba = np.zeros((layer.height, layer.width, 4), dtype=np.uint8)
        radius = max(1, int(round(line_width * unit / 2.0)))

        if pattern == "dotted":
            dot_radius = max(
                1,
                int(round(line_width * style["dot_radius_widths"] * unit)),
            )
            for point in dotted_polyline(
                group_points, line_width * style["spacing_widths"]
            ):
                cv2.circle(
                    rgba, fixed_point(point), dot_radius, colour, thickness=-1,
                    lineType=cv2.LINE_AA, shift=shift,
                )
            endpoint_radius = dot_radius
        elif pattern == "dashed":
            for dash in dashed_polyline(
                group_points,
                line_width * style["dash_widths"],
                line_width * style["gap_widths"],
            ):
                coordinates = np.asarray(
                    [fixed_point(point) for point in dash], dtype=np.int32
                ).reshape((-1, 1, 2))
                cv2.polylines(
                    rgba, [coordinates], False, colour, line_width,
                    lineType=cv2.LINE_AA, shift=shift,
                )
                # OpenCV polylines do not guarantee round dash caps.
                for endpoint in (dash[0], dash[-1]):
                    cv2.circle(
                        rgba, fixed_point(endpoint), radius, colour, thickness=-1,
                        lineType=cv2.LINE_AA, shift=shift,
                    )
            endpoint_radius = radius
        else:
            coordinates = np.asarray(
                [fixed_point(point) for point in group_points], dtype=np.int32
            ).reshape((-1, 1, 2))
            if len(coordinates) >= 2:
                cv2.polylines(
                    rgba, [coordinates], False, colour, line_width,
                    lineType=cv2.LINE_AA, shift=shift,
                )
            endpoint_radius = radius

        # Every group gets a complete antialiased cap at both ends. Internal
        # caps are divided into complementary half-planes below, making one
        # continuous rounded join without alpha-stacking two colours.
        if group_points:
            for point in (group_points[0], group_points[-1]):
                cv2.circle(
                    rgba, fixed_point(point), endpoint_radius, colour, thickness=-1,
                    lineType=cv2.LINE_AA, shift=shift,
                )

        # OpenCV writes coverage into every RGBA channel. Convert RGB back to
        # straight alpha so Pillow does not apply edge coverage twice.
        coverage = rgba[..., 3].astype(np.float32) / float(alpha)
        edge = (coverage > 0.0) & (coverage < 1.0)
        if np.any(edge):
            rgba_rgb = rgba[..., :3]
            rgba_rgb[edge] = np.clip(
                rgba_rgb[edge].astype(np.float32) / coverage[edge, None],
                0,
                255,
            ).astype(np.uint8)
        # A later group's antialiased fringe must never overwrite the solid
        # centre of an earlier group.  That produced small transparent holes
        # wherever neighbouring/looping severity sections came close.  Keep
        # whichever sample has the greater geometric coverage; the dedicated
        # boundary blend below still owns actual state-change joins.
        owned = rgba[..., 3] > storm_rgba[..., 3]
        storm_rgba[owned] = rgba[owned]

        if previous_rgba is not None and previous_colour is not None:
            local_join, signed_distance = boundary_partitions[group_index - 1]
            union = local_join & (
                (previous_rgba[..., 3] > 0) | (rgba[..., 3] > 0)
            )
            if np.any(union):
                # Antialias the internal colour boundary by blending colour
                # weights across 1.5 pixels. Alpha is calculated once from a
                # single interpolated opacity, so two translucent paths never
                # stack or create a dark overlap seam.
                weight = np.clip(
                    0.5 + signed_distance[union] / 1.5, 0.0, 1.0
                ).astype(np.float32)
                old_alpha = float(previous_colour[3])
                new_alpha = float(colour[3])
                old_coverage = (
                    previous_rgba[..., 3][union].astype(np.float32)
                    / max(1.0, old_alpha)
                )
                new_coverage = (
                    rgba[..., 3][union].astype(np.float32)
                    / max(1.0, new_alpha)
                )
                outer_coverage = np.maximum(old_coverage, new_coverage)
                blended_alpha = outer_coverage * (
                    old_alpha * (1.0 - weight) + new_alpha * weight
                )
                old_rgb = np.asarray(previous_colour[:3], dtype=np.float32)
                new_rgb = np.asarray(colour[:3], dtype=np.float32)
                blended_rgb = (
                    old_rgb[None, :] * (1.0 - weight[:, None])
                    + new_rgb[None, :] * weight[:, None]
                )
                storm_rgba[..., :3][union] = np.clip(
                    blended_rgb, 0, 255
                ).astype(np.uint8)
                storm_rgba[..., 3][union] = np.clip(
                    blended_alpha, 0, 255
                ).astype(np.uint8)
        previous_rgba = rgba
        previous_colour = colour
    return Image.alpha_composite(layer, Image.fromarray(storm_rgba, "RGBA"))


def draw_track_native_aa(
    layer: Image.Image,
    points: list[TrackPoint],
    moment: datetime,
    cfg: dict,
) -> Image.Image:
    if not cfg.get("show_track", True) or moment <= points[0].time:
        return layer
    return draw_track_groups_native_aa(
        layer, track_groups(points, moment, cfg), cfg
    )


class DepositedTrackRenderer:
    """Checkpoint native-resolution AA paths and draw only the recent tail."""

    def __init__(
        self,
        tracks: list[SeasonTrack],
        cfg: dict,
        source_map: Image.Image,
        output_size: tuple[int, int],
    ):
        self.tracks = tracks
        self.source_map = source_map.convert("RGB")
        self.width, self.height = self.source_map.size
        self.output_width, self.output_height = output_size
        self.output_cfg = cfg
        self.interval = max(
            0.1, float(cfg.get("track_deposit_interval_seconds", 5.0))
        )
        self.checkpoint_video_time: float | None = None
        self.checkpoint_moment: datetime | None = None
        self.checkpoint_camera: CameraRect | None = None
        self.baked_tracks = Image.new(
            "RGBA", output_size, (0, 0, 0, 0)
        )
        self.cached_camera_key: tuple[object, ...] | None = None
        self.cached_camera_map: Image.Image | None = None

    def _bake(
        self, moment: datetime, video_time: float, camera: CameraRect
    ) -> None:
        # Rebuild complete paths directly at final output resolution.  Drawing
        # them in source-map pixels and enlarging that raster with the camera
        # magnified both stair steps and integer-width rounding errors.
        layer = Image.new(
            "RGBA", (self.output_width, self.output_height), (0, 0, 0, 0)
        )
        for track in self.tracks:
            if moment > track.points[0].time:
                layer = draw_track_native_aa(
                    layer,
                    track.points,
                    min(moment, track.points[-1].time),
                    self.output_cfg,
                )
        self.baked_tracks = layer
        self.checkpoint_video_time = video_time
        self.checkpoint_moment = moment
        self.checkpoint_camera = CameraRect(
            camera.west, camera.east, camera.south, camera.north
        )

    def _warped_checkpoint(self, camera: CameraRect) -> Image.Image:
        """Reproject the checkpoint raster without repeatedly resampling it."""
        assert self.checkpoint_camera is not None
        old = self.checkpoint_camera
        if (
            abs(old.west - camera.west) < 1e-10
            and abs(old.east - camera.east) < 1e-10
            and abs(old.south - camera.south) < 1e-10
            and abs(old.north - camera.north) < 1e-10
        ):
            return self.baked_tracks

        old_lon_span = max(1e-12, old.east - old.west)
        new_lon_span = max(1e-12, camera.east - camera.west)
        old_lat_span = max(1e-12, old.north - old.south)
        new_lat_span = max(1e-12, camera.north - camera.south)
        matrix = np.asarray(
            [
                [
                    old_lon_span / new_lon_span,
                    0.0,
                    (old.west - camera.west) / new_lon_span * self.output_width,
                ],
                [
                    0.0,
                    old_lat_span / new_lat_span,
                    (camera.north - old.north) / new_lat_span * self.output_height,
                ],
            ],
            dtype=np.float32,
        )

        rgba = np.asarray(self.baked_tracks, dtype=np.uint8)
        alpha = rgba[..., 3:4].astype(np.float32) / 255.0
        premultiplied = np.empty(rgba.shape, dtype=np.float32)
        premultiplied[..., :3] = rgba[..., :3].astype(np.float32) * alpha
        premultiplied[..., 3] = rgba[..., 3].astype(np.float32)
        warped = cv2.warpAffine(
            premultiplied,
            matrix,
            (self.output_width, self.output_height),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        warped = np.clip(warped, 0.0, 255.0)
        warped_alpha = warped[..., 3]
        visible = warped_alpha > 1e-4
        straight = np.zeros(rgba.shape, dtype=np.uint8)
        straight[..., 3] = np.rint(warped_alpha).astype(np.uint8)
        straight[..., :3][visible] = np.clip(
            warped[..., :3][visible]
            / (warped_alpha[visible, None] / 255.0),
            0.0,
            255.0,
        ).astype(np.uint8)
        return Image.fromarray(straight, "RGBA")

    def frame(
        self, moment: datetime, camera: CameraRect, video_time: float
    ) -> tuple[Image.Image, Image.Image]:
        if (
            self.checkpoint_video_time is None
            or video_time - self.checkpoint_video_time >= self.interval - 1e-9
        ):
            self._bake(moment, video_time, camera)

        camera_key = (
            round(camera.west, 10),
            round(camera.east, 10),
            round(camera.south, 10),
            round(camera.north, 10),
        )
        if camera_key != self.cached_camera_key or self.cached_camera_map is None:
            self.cached_camera_map = render_camera_map(
                self.source_map, camera, self.output_cfg
            )
            self.cached_camera_key = camera_key

        tail = Image.new(
            "RGBA", (self.output_width, self.output_height), (0, 0, 0, 0)
        )
        assert self.checkpoint_moment is not None
        for track in self.tracks:
            groups = track_groups_between(
                track.points, self.checkpoint_moment, moment, self.output_cfg
            )
            if groups:
                tail = draw_track_groups_native_aa(tail, groups, self.output_cfg)

        deposited = np.asarray(self._warped_checkpoint(camera)).copy()
        recent = np.asarray(tail)
        recent_visible = recent[..., 3] > 0
        # One path, one coverage value: never alpha-stack the checkpoint and
        # recent tail at their shared endpoint.  Stacking caused the temporary
        # swollen section that disappeared at the next checkpoint.
        deposited[..., :3][recent_visible] = recent[..., :3][recent_visible]
        deposited[..., 3][recent_visible] = np.maximum(
            deposited[..., 3][recent_visible], recent[..., 3][recent_visible]
        )
        return self.cached_camera_map, Image.fromarray(deposited, "RGBA")


def deterministic_phase(dat_path: Path, loop_period: Fraction) -> float:
    digest = hashlib.sha256(dat_path.stem.encode("utf-8")).digest()
    fraction = int.from_bytes(digest[:8], "big") / float(2**64)
    return fraction * float(loop_period)


def material_target_sizes(
    tracks: list[SeasonTrack],
    infos: dict[str, ClipInfo],
    cfg: dict,
) -> dict[str, int]:
    """Calculate each used material's maximum display size for the whole video."""
    base_size = max(1, int(cfg["icon_size"]))
    configured_scales = cfg.get("icon_state_scales", {})
    used_states = {point.state for track in tracks for point in track.points}
    sizes: dict[str, int] = {}
    for state in used_states:
        if state in configured_scales:
            # Scale is relative to the original source canvas.  A 400 px
            # source at 0.25 becomes a 100 px cache and is pasted at 100%.
            source_size = max(infos[state].width, infos[state].height)
            scale = max(0.01, float(configured_scales[state]))
            sizes[state] = max(1, int(math.ceil(source_size * scale)))
        else:
            sizes[state] = base_size
    return sizes


MUSIC_EXTENSIONS = {
    ".aac", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav", ".wma"
}


def discover_music_files() -> list[Path]:
    if not MUSIC_DIR.is_dir():
        return []
    return sorted(
        (
            path
            for path in MUSIC_DIR.iterdir()
            if path.is_file() and path.suffix.lower() in MUSIC_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )


class MusicSequencer:
    """Yield endless 48 kHz stereo frames with a reproducible track order."""

    def __init__(self, files: list[Path], seed: int, sample_rate: int):
        self.files = files
        self.random = random.Random(seed)
        self.sample_rate = sample_rate
        self.previous: Path | None = None

    def _choose(self) -> Path:
        if len(self.files) == 1:
            selected = self.files[0]
        else:
            candidates = [path for path in self.files if path != self.previous]
            selected = self.random.choice(candidates)
        self.previous = selected
        print(f"音乐：{selected.name}")
        return selected

    def frames(self):
        while True:
            path = self._choose()
            yielded = False
            with av.open(str(path)) as container:
                stream = next(
                    (item for item in container.streams if item.type == "audio"),
                    None,
                )
                if stream is None:
                    raise RuntimeError(f"音乐文件没有音频流：{path.name}")
                resampler = av.AudioResampler(
                    format="fltp", layout="stereo", rate=self.sample_rate
                )
                for decoded in container.decode(stream):
                    for frame in resampler.resample(decoded):
                        if frame.samples > 0:
                            yielded = True
                            yield frame
                for frame in resampler.resample(None):
                    if frame.samples > 0:
                        yielded = True
                        yield frame
            if not yielded:
                raise RuntimeError(f"无法解码音乐文件：{path.name}")


def decode_audio_samples(path: Path, sample_rate: int) -> np.ndarray | None:
    """Decode an optional media audio stream as planar float stereo samples."""
    with av.open(str(path)) as container:
        stream = next(
            (item for item in container.streams if item.type == "audio"), None
        )
        if stream is None:
            return None
        resampler = av.AudioResampler(
            format="fltp", layout="stereo", rate=sample_rate
        )
        chunks: list[np.ndarray] = []
        for decoded in container.decode(stream):
            for frame in resampler.resample(decoded):
                if frame.samples:
                    chunks.append(
                        frame.to_ndarray().astype(np.float32, copy=False)
                    )
        for frame in resampler.resample(None):
            if frame.samples:
                chunks.append(frame.to_ndarray().astype(np.float32, copy=False))
    if not chunks:
        return None
    return np.ascontiguousarray(np.concatenate(chunks, axis=1))


class AudioSampleReader:
    """Read an arbitrary number of samples from a frame iterator."""

    def __init__(self, frames):
        self.frames = frames
        self.pending = np.empty((2, 0), dtype=np.float32)

    def read(self, count: int) -> np.ndarray:
        while self.pending.shape[1] < count:
            frame = next(self.frames)
            samples = frame.to_ndarray().astype(np.float32, copy=False)
            self.pending = np.concatenate((self.pending, samples), axis=1)
        result = self.pending[:, :count]
        self.pending = self.pending[:, count:]
        return np.ascontiguousarray(result)


def configure_encoder_stream(
    container,
    codec_name: str,
    rate: int | Fraction,
    width: int,
    height: int,
    cfg: dict,
):
    stream = container.add_stream(codec_name, rate=rate)
    stream.width = width
    stream.height = height
    if codec_name.endswith("_qsv"):
        stream.pix_fmt = "nv12"
        stream.options = {
            "global_quality": str(cfg.get("qsv_global_quality", 20)),
            "preset": str(cfg.get("qsv_preset", "veryfast")),
            "async_depth": str(cfg.get("qsv_async_depth", 4)),
        }
    else:
        stream.pix_fmt = "yuv420p"
        stream.options = {
            "crf": str(cfg["crf"]),
            "preset": str(cfg["preset"]),
        }
    return stream


def qsv_encoder_available(cfg: dict, fps: int) -> bool:
    """Initialize and flush a tiny in-memory encode before touching output."""
    codec_name = str(cfg.get("qsv_codec", "h264_qsv"))
    try:
        buffer = io.BytesIO()
        with av.open(buffer, mode="w", format="mp4") as test_container:
            stream = configure_encoder_stream(
                test_container, codec_name, fps, 128, 128, cfg
            )
            frame = av.VideoFrame(128, 128, "nv12")
            for plane in frame.planes:
                plane.update(bytes(plane.buffer_size))
            frame.pts = 0
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                test_container.mux(packet)
            for packet in stream.encode():
                test_container.mux(packet)
        return True
    except Exception as exc:
        print(f"QSV 不可用，回退软件编码：{exc}")
        return False


def select_encoder(cfg: dict, fps: int) -> str:
    requested = str(cfg.get("encoder", "auto")).strip().lower()
    if requested not in {"auto", "qsv", "software"}:
        raise SystemExit("config.py 的 encoder 只能是 auto、qsv 或 software")
    if requested in {"auto", "qsv"} and qsv_encoder_available(cfg, fps):
        return str(cfg.get("qsv_codec", "h264_qsv"))
    if requested == "qsv":
        print("已请求 QSV，但初始化失败；本次使用 libx264")
    return "libx264"


def composite_storm_label(
    canvas: Image.Image,
    center: tuple[float, float],
    label0: str,
    label1: str,
    label_weight: float,
    visibility: float,
    display_size: int,
    cfg: dict,
    icon_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    icon_factor: float,
) -> Image.Image:
    """Composite one storm's label directly above that storm's icon."""
    shadow_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow_layer)
    label_layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    label_draw = ImageDraw.Draw(label_layer)
    shadow_x = float(cfg.get("label_shadow_offset_x", 3)) * icon_factor
    shadow_y = float(cfg.get("label_shadow_offset_y", 3)) * icon_factor
    shadow_opacity = max(
        0.0, min(1.0, float(cfg.get("label_shadow_opacity", 0.70)))
    )
    stroke = max(
        0, int(round(float(cfg["label_stroke_width"]) * icon_factor))
    )

    def draw_name(
        text: str, position: tuple[float, float], alpha: int
    ) -> None:
        if not text or alpha <= 0:
            return
        shadow_alpha = int(alpha * shadow_opacity)
        shadow_draw.text(
            (position[0] + shadow_x, position[1] + shadow_y),
            text,
            font=icon_font,
            fill=(0, 0, 0, shadow_alpha),
            stroke_width=stroke,
            stroke_fill=(0, 0, 0, shadow_alpha),
        )
        label_draw.text(
            position,
            text,
            font=icon_font,
            fill=(255, 255, 255, alpha),
            stroke_width=stroke,
            stroke_fill=(0, 0, 0, alpha),
        )

    label_x = center[0] + display_size * float(cfg["label_offset_x_icons"])
    label_y = center[1] + display_size * float(cfg["label_offset_y_icons"])
    fly_x = display_size * float(cfg["label_fly_x_icons"])
    fly_y = display_size * float(cfg["label_fly_y_icons"])
    if label0 != label1:
        old_motion = ease_in_cubic(label_weight)
        new_motion = ease_out_cubic(label_weight)
        old_opacity = (1.0 - label_weight) ** 3
        new_opacity = 1.0 - (1.0 - label_weight) ** 3
    else:
        old_motion = new_motion = 1.0
        old_opacity, new_opacity = 0.0, 1.0

    if label0 and label0 != label1 and old_opacity > 0.0:
        draw_name(
            label0,
            (label_x + fly_x * old_motion, label_y + fly_y * old_motion),
            int(255 * old_opacity * visibility),
        )
    if label1:
        draw_name(
            label1,
            (
                label_x + fly_x * (1.0 - new_motion),
                label_y + fly_y * (1.0 - new_motion),
            ),
            int(255 * visibility * new_opacity),
        )

    shadow_blur = max(
        0.0, float(cfg.get("label_shadow_blur", 3.0)) * icon_factor
    )
    if shadow_blur > 0.0:
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(shadow_blur))
    canvas = Image.alpha_composite(canvas.convert("RGBA"), shadow_layer)
    return Image.alpha_composite(canvas, label_layer).convert("RGB")


def ace_bar_colour(value: float, cfg: dict) -> tuple[int, int, int]:
    configured = cfg.get("ace_bar_colors", {})
    stops = sorted(
        (float(stop), ImageColor.getrgb(str(colour)))
        for stop, colour in configured.items()
    )
    if not stops:
        return 0, 255, 0
    if value <= stops[0][0]:
        return stops[0][1]
    if value >= stops[-1][0]:
        return stops[-1][1]
    for (left_value, left_colour), (right_value, right_colour) in zip(
        stops, stops[1:]
    ):
        if value <= right_value:
            weight = (value - left_value) / max(1e-9, right_value - left_value)
            return tuple(
                int(round(left + (right - left) * weight))
                for left, right in zip(left_colour, right_colour)
            )
    return stops[-1][1]


def draw_ace_bar(
    canvas: Image.Image,
    value: float,
    cfg: dict,
    font_file: str | None,
    font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont],
) -> Image.Image:
    """Draw the fixed top-left ACE title, outline, fill, and floored value."""
    if not cfg.get("show_ace_bar", True):
        return canvas
    scale = canvas.height / 720.0
    bar_left = int(round(float(cfg.get("ace_bar_offset_x", 26)) * scale))
    bar_top = int(round(float(cfg.get("ace_bar_offset_y", 66)) * scale))
    bar_width = max(1, int(round(float(cfg.get("ace_bar_width", 620)) * scale)))
    bar_height = max(1, int(round(float(cfg.get("ace_bar_height", 58)) * scale)))
    border = max(
        1, int(round(float(cfg.get("ace_bar_border_width", 3)) * scale))
    )
    inset = max(
        border + 1,
        int(round(float(cfg.get("ace_bar_inner_margin", 10)) * scale)),
    )
    title_left = int(round(float(cfg.get("ace_title_offset_x", 26)) * scale))
    title_top = int(round(float(cfg.get("ace_title_offset_y", 24)) * scale))
    title_size = max(
        1, int(round(float(cfg.get("ace_title_size", 27)) * scale))
    )
    number_size = max(
        1, int(round(float(cfg.get("ace_value_size", 34)) * scale))
    )
    for size in (title_size, number_size):
        if size not in font_cache:
            font_cache[size] = load_font(
                size, bold=True, font_file=font_file
            )
    title_font = font_cache[title_size]
    number_font = font_cache[number_size]
    title = "Accumulative Cyclone Energy (ACE)"
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (title_left, title_top), title, font=title_font, fill=(255, 255, 255)
    )
    right = bar_left + bar_width
    bottom = bar_top + bar_height
    draw.rectangle(
        (bar_left, bar_top, right, bottom),
        outline=(255, 255, 255),
        width=border,
    )

    inner_left = bar_left + inset
    inner_right = right - inset
    inner_top = bar_top + inset
    inner_bottom = bottom - inset
    upper_limit = max(1e-9, float(cfg.get("ace_bar_upper_limit", 500.0)))
    fill_ratio = max(0.0, min(1.0, value / upper_limit))
    fill_right = inner_left + int(round((inner_right - inner_left) * fill_ratio))
    if fill_right > inner_left:
        draw.rectangle(
            (inner_left, inner_top, fill_right, inner_bottom),
            fill=ace_bar_colour(value, cfg),
        )

    number = str(math.floor(max(0.0, value)))
    number_box = draw.textbbox((0, 0), number, font=number_font)
    value_right_margin = max(
        0,
        int(round(float(cfg.get("ace_value_right_margin", 10)) * scale)),
    )
    number_x = right - value_right_margin - (number_box[2] - number_box[0])
    number_y = bar_top + (bar_height - (number_box[3] - number_box[1])) / 2.0 - number_box[1]
    draw.text(
        (number_x, number_y), number, font=number_font, fill=(255, 255, 255)
    )
    return canvas


def render(dat_paths: list[Path], output_path: Path, cfg: dict) -> None:
    # Validate every mandatory external asset before parsing or preprocessing.
    # Intro and music are the only deliberately optional media inputs.
    map_path = required_asset_path(cfg, "map_file", IMAGE_ASSET_EXTENSIONS)
    timeline_path = required_asset_path(
        cfg, "timeline_video_file", VIDEO_ASSET_EXTENSIONS
    )
    font_path = required_asset_path(cfg, "font_file", FONT_ASSET_EXTENSIONS)
    try:
        ImageFont.truetype(str(font_path), 12)
    except OSError as exc:
        raise SystemExit(f"无法读取必需字体：{font_path}") from exc

    cleanup_stale_preprocess_cache()
    tracks = [SeasonTrack(path, parse_btk(path, cfg)) for path in dat_paths]
    # Painter's order: older storms first, newer storms last/on top.
    tracks.sort(key=lambda track: (track.points[0].time, track.path.name.lower()))
    ace_series = (
        AceSeries(dat_paths, bool(cfg.get("ace_includes_subtc", False)))
        if cfg.get("show_ace_bar", True)
        else None
    )
    if ace_series is not None:
        print(f"ACE：最终 {ace_series.total:.4f}")
    materials = discover_materials()
    infos = {state: probe_clip(state, path, cfg) for state, path in materials.items()}
    if not infos:
        raise SystemExit("没有在 tc_icons 文件夹中找到可用的 MP4 图标素材")

    loop_period = gcd_fraction([info.duration for info in infos.values()])
    if loop_period <= 0:
        raise SystemExit("无法计算素材循环周期")
    print("素材参数：")
    for info in infos.values():
        print(f"  {info.path.name:18} {info.width}x{info.height}  {info.fps} fps  {info.frames} 帧  {float(info.duration):g} 秒")
    print(f"公共循环周期（时长最大公因数）：{float(loop_period):g} 秒")

    required = sorted({point.state for track in tracks for point in track.points})
    missing = [state for state in required if state not in infos]
    if missing:
        raise SystemExit(f"缺少状态素材：{', '.join(missing)}")
    # Landfalls must be known before the season timeline is built: a late
    # one-shot effect extends the final active moment by its remaining runtime.
    landfalls: list[Landfall] = []
    landfall_infos: dict[str, ClipInfo] = {}
    if cfg.get("show_landfall_effects", True):
        coastline_path = ROOT / str(cfg["coastline_file"])
        if not coastline_path.exists():
            raise SystemExit(f"找不到海岸线数据：{coastline_path}")
        coastline = CoastlineIndex(coastline_path, cfg)
        print(
            f"海岸线：{coastline_path.name}，"
            f"索引 {len(coastline.segments)} 条线段"
        )
        landfalls = detect_landfalls(tracks, coastline, cfg)
        print(f"自动检测登陆：{len(landfalls)} 个")
    if landfalls:
        print(
            f"读取 {len(landfalls)} 个登陆点："
            + ", ".join(
                f"{item.time:%Y-%m-%d %H:%MZ} {item.intensity}kt"
                for item in landfalls
            )
        )
        landfall_materials = discover_landfall_materials()
        required_landfall_states = sorted({item.state for item in landfalls})
        missing_landfall_states = [
            state for state in required_landfall_states
            if state not in landfall_materials
        ]
        if missing_landfall_states:
            raise SystemExit(
                "缺少登陆特效素材："
                + ", ".join(
                    f"landfall_icons/{state}.mp4"
                    for state in missing_landfall_states
                )
            )
        landfall_probe_cfg = dict(cfg)
        landfall_probe_cfg["normalize_single_trailing_frame"] = False
        landfall_infos = {
            state: probe_clip(
                state, landfall_materials[state], landfall_probe_cfg
            )
            for state in required_landfall_states
        }

    final_landfall_effect_end: datetime | None = None
    if landfalls:
        key_points = cfg.get("landfall_icon_key_points", {})
        effect_ends: list[datetime] = []
        for landfall in landfalls:
            key_point = float(
                key_points.get(
                    landfall.state,
                    key_points.get(
                        landfall.state.upper(), cfg["landfall_icon_key_point"]
                    ),
                )
            )
            remaining_seconds = max(
                0.0, float(landfall_infos[landfall.state].duration) - key_point
            )
            effect_ends.append(
                landfall.time
                + timedelta(
                    hours=remaining_seconds * simulation_hours_per_second(cfg)
                )
            )
        final_landfall_effect_end = max(effect_ends)

    timeline = SeasonTimeline(tracks, cfg, final_landfall_effect_end)
    planned_camera_controller = (
        CameraController(timeline, tracks, cfg, 16.0 / 9.0)
        if cfg.get("auto_zoom", True)
        else None
    )
    base_target_sizes = material_target_sizes(tracks, infos, cfg)
    maximum_zoom = max(
        (
            camera_zoom_factor(camera, cfg)
            for camera in planned_camera_controller.targets.values()
        ),
        default=1.0,
    ) if planned_camera_controller is not None else 1.0
    maximum_icon_factor = max(
        0.01,
        1.0 + (maximum_zoom - 1.0) * float(cfg.get("tc_icon_mag", 0.125)),
    )
    prepared: dict[str, PreparedClip] = {}
    for state in required:
        target_size = max(
            1, int(math.ceil(base_target_sizes[state] * maximum_icon_factor))
        )
        print(
            f"预处理 {infos[state].path.name}："
            f"{infos[state].width}x{infos[state].height} -> "
            f"{target_size}x{target_size}"
        )
        prepared[state] = prepare_clip(
            infos[state], loop_period, cfg, target_size
        )

    for track in tracks:
        track.phase_offset = deterministic_phase(track.path, loop_period)

    resolution = int(cfg.get("resolution", cfg.get("height", 720)))
    if resolution not in {720, 1080}:
        raise SystemExit("config.py 的 resolution 只能是 720 或 1080")
    height, width = resolution, resolution * 16 // 9
    cfg["width"], cfg["height"] = width, height
    fps = int(cfg["fps"])
    if fps not in {30, 60}:
        raise SystemExit("config.py 的 fps 只能是 30 或 60")
    intro = float(cfg["intro_seconds"])
    opening_duration = max(0.0, float(cfg["intro_video_duration"]))
    map_duration = timeline.video_duration
    opening_frames = int(round(opening_duration * fps))
    map_frames = int(math.ceil(map_duration * fps))
    total_frames = opening_frames + map_frames
    configured_font = str(font_path)
    font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}

    intro_path = optional_asset_path(
        cfg, "intro_video_file", VIDEO_ASSET_EXTENSIONS
    )
    intro_reader: IntroReader | None = None
    if intro_path is not None and intro_path.is_file() and opening_duration > 0.0:
        intro_reader = IntroReader(intro_path, width, height, opening_duration)
        print(
            f"片头：{intro_path.name} 前 {opening_duration:g} 秒 -> "
            f"{width}x{height} 居中铺满"
        )
    else:
        opening_duration = 0.0
        opening_frames = 0
        total_frames = map_frames
        if intro_path is None:
            print("片头：未配置，跳过")
        else:
            print(f"片头：未找到 {intro_path.name}，跳过")

    timeline_info = ensure_timeline_proxy(
        timeline_path,
        cfg,
        timeline_source_seconds(timeline.start, cfg),
        timeline_source_seconds(timeline.end, cfg),
    )
    timeline_reader: TimelineReader | None = TimelineReader(timeline_info, cfg)
    print(
        f"Timeline: {timeline_path.name} -> "
        f"{timeline_info.width}x{timeline_info.height} at bottom-left"
    )

    print(f"读取 {len(tracks)} 个气旋：{', '.join(track.path.name for track in tracks)}")
    for segment in timeline.segments:
        if segment.speed > 1.0:
            print(
                f"  快进 {segment.speed:g}x："
                f"{segment.start:%Y-%m-%d %HZ} -> {segment.end:%Y-%m-%d %HZ}"
            )

    source_map: Image.Image | None = Image.open(map_path).convert("RGB")
    camera_controller: CameraController | None = None
    source_w, source_h = source_map.size
    if cfg.get("auto_zoom", True):
        camera_controller = planned_camera_controller
        background = Image.new("RGB", (width, height), (8, 8, 45))
        print(f"地图：{map_path.name} ({source_w}x{source_h})，自动缩放已启用")
    else:
        scale = min(width / source_w, height / source_h)
        map_w, map_h = int(round(source_w * scale)), int(round(source_h * scale))
        map_x, map_y = (width - map_w) // 2, (height - map_h) // 2
        background = Image.new("RGB", (width, height), (8, 8, 45))
        background.paste(
            source_map.resize((map_w, map_h), Image.Resampling.LANCZOS),
            (map_x, map_y),
        )
        cfg["_map_viewport_x"], cfg["_map_viewport_y"] = map_x, map_y
        cfg["_map_viewport_width"], cfg["_map_viewport_height"] = map_w, map_h
        print(f"地图：{map_path.name} ({source_w}x{source_h}) -> 视区 {map_w}x{map_h}+{map_x}+{map_y}")

    deposited_track_renderer: DepositedTrackRenderer | None = None
    if (
        cfg.get("show_track", True)
        and cfg.get("track_antialias", True)
        and source_map is not None
        and camera_controller is not None
        and abs(float(cfg.get("track_width_mag", 1.0)) - 1.0) < 1e-9
    ):
        deposited_track_renderer = DepositedTrackRenderer(
            tracks, cfg, source_map, (width, height)
        )
        print(
            "路径渲染：每 "
            f"{float(cfg.get('track_deposit_interval_seconds', 5.0)):g} 秒沉淀，"
            "区间内仅绘制新增路径"
        )
    elif cfg.get("show_track", True):
        print("路径渲染：逐帧模式（沉淀层要求自动缩放、抗锯齿且 track_width_mag=1）")

    prepared_landfalls: dict[str, PreparedClip] = {}
    if landfall_infos:
        for state, info in landfall_infos.items():
            landfall_scale = max(
                0.01, float(cfg.get("landfall_icon_scale", 1.0))
            )
            landfall_target_width = max(
                1, int(round(info.width * landfall_scale))
            )
            landfall_target_height = max(
                1,
                int(round(info.height * landfall_scale)),
            )
            print(
                f"预处理登陆特效 {info.path.name}："
                f"{info.width}x{info.height} -> "
                f"{landfall_target_width}x{landfall_target_height}"
            )
            prepared_landfalls[state] = prepare_landfall_clip(
                info, cfg, landfall_target_width
            )

    audio_rate = 48000
    landfall_audio_by_state: dict[str, np.ndarray] = {}
    for state, info in landfall_infos.items():
        samples = decode_audio_samples(info.path, audio_rate)
        if samples is not None:
            landfall_audio_by_state[state] = samples
            print(
                f"登陆特效：{info.path.name} "
                f"{samples.shape[1] / audio_rate:.3f} 秒"
            )
    landfall_audio_cues: list[tuple[int, np.ndarray]] = []
    key_points = cfg.get("landfall_icon_key_points", {})
    for landfall in landfalls:
        samples = landfall_audio_by_state.get(landfall.state)
        if samples is None:
            continue
        key_point = float(
            key_points.get(
                landfall.state,
                key_points.get(
                    landfall.state.upper(), cfg["landfall_icon_key_point"]
                ),
            )
        )
        cue_start = int(
            round(
                (
                    opening_duration
                    + timeline.video_offset_for(landfall.time)
                    - key_point
                )
                * audio_rate
            )
        )
        if cue_start < 0:
            samples = samples[:, min(samples.shape[1], -cue_start):]
            cue_start = 0
        if samples.shape[1]:
            landfall_audio_cues.append((cue_start, samples))
    landfall_audio_cues.sort(key=lambda item: item[0])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_started = time.perf_counter()
    encoder_name = select_encoder(cfg, fps)
    print(
        "编码器："
        + (
            f"{encoder_name}（Intel QSV 硬件编码）"
            if encoder_name.endswith("_qsv")
            else "libx264（CPU 软件编码）"
        )
    )
    cached_camera_key: tuple[float, float, float, float] | None = None
    cached_camera_canvas: Image.Image | None = None
    music_files = discover_music_files()
    if music_files:
        print(f"音乐库：{len(music_files)} 个文件，seed={int(cfg['seed'])}")
    elif landfall_audio_cues:
        print("音乐库：为空，仅输出登陆素材音效")
    else:
        print("音乐库：为空，输出不含音轨")
    with av.open(str(output_path), mode="w", options={"movflags": "+faststart"}) as container:
        stream = configure_encoder_stream(
            container, encoder_name, fps, width, height, cfg
        )
        audio_stream = None
        music_reader = None
        audio_cursor = 0
        total_audio_samples = int(round(total_frames * audio_rate / fps))
        if music_files or landfall_audio_cues:
            audio_stream = container.add_stream("aac", rate=audio_rate)
            audio_stream.layout = "stereo"
            audio_stream.bit_rate = int(cfg["music_bitrate"])
            if music_files:
                music_reader = AudioSampleReader(
                    MusicSequencer(
                        music_files, int(cfg["seed"]), audio_rate
                    ).frames()
                )

        def feed_audio_until(target_sample: int) -> None:
            nonlocal audio_cursor
            if audio_stream is None:
                return
            target_sample = min(target_sample, total_audio_samples)
            while audio_cursor < target_sample:
                sample_count = min(1024, target_sample - audio_cursor)
                samples = (
                    music_reader.read(sample_count)
                    if music_reader is not None
                    else np.zeros((2, sample_count), dtype=np.float32)
                )
                chunk_end = audio_cursor + sample_count
                for cue_start, cue_samples in landfall_audio_cues:
                    cue_end = cue_start + cue_samples.shape[1]
                    overlap_start = max(audio_cursor, cue_start)
                    overlap_end = min(chunk_end, cue_end)
                    if overlap_end <= overlap_start:
                        continue
                    target_start = overlap_start - audio_cursor
                    source_start = overlap_start - cue_start
                    length = overlap_end - overlap_start
                    samples[:, target_start:target_start + length] += cue_samples[
                        :, source_start:source_start + length
                    ]
                samples = np.ascontiguousarray(
                    np.clip(samples, -1.0, 1.0), dtype=np.float32
                )
                frame = av.AudioFrame.from_ndarray(
                    samples, format="fltp", layout="stereo"
                )
                frame.pts = audio_cursor
                frame.time_base = Fraction(1, audio_rate)
                frame.sample_rate = audio_rate
                for packet in audio_stream.encode(frame):
                    container.mux(packet)
                audio_cursor += sample_count

        def encode_canvas(canvas: Image.Image, frame_index: int) -> None:
            video_frame = av.VideoFrame.from_image(canvas)
            if encoder_name.endswith("_qsv"):
                video_frame = video_frame.reformat(
                    width=width, height=height, format="nv12"
                )
            video_frame.pts = frame_index
            video_frame.time_base = Fraction(1, fps)
            for packet in stream.encode(video_frame):
                container.mux(packet)
            feed_audio_until(
                int(round((frame_index + 1) * audio_rate / fps))
            )

            completed_frames = frame_index + 1
            if (
                completed_frames % 10 == 0
                or completed_frames == total_frames
            ):
                elapsed = max(1e-9, time.perf_counter() - render_started)
                frames_per_second = completed_frames / elapsed
                eta = (total_frames - completed_frames) / max(
                    1e-9, frames_per_second
                )
                percent = completed_frames * 100.0 / total_frames
                print(
                    f"\r正在编码最终视频：{completed_frames}/{total_frames} 帧 "
                    f"({percent:5.1f}%)  "
                    f"速度 {frames_per_second:.2f} 帧/s  "
                    f"已用 {format_duration(elapsed)}  "
                    f"ETA {format_duration(eta)}",
                    end="",
                    flush=True,
                )

        intro_fade = max(0.001, float(cfg["intro_video_fade_out_seconds"]))
        for opening_frame_index in range(opening_frames):
            assert intro_reader is not None
            opening_time = opening_frame_index / fps
            intro_rgb = intro_reader.frame_at(opening_time)
            fade_to_black = 1.0 - smoothstep(
                (opening_time - (opening_duration - intro_fade)) / intro_fade
            )
            if fade_to_black < 1.0:
                intro_rgb = np.clip(
                    intro_rgb.astype(np.float32) * fade_to_black, 0, 255
                ).astype(np.uint8)
            opening_canvas = Image.fromarray(intro_rgb, "RGB")
            opening_canvas = render_intro_title(
                opening_canvas,
                opening_time,
                fade_to_black,
                cfg,
                configured_font,
                font_cache,
            )
            encode_canvas(opening_canvas, opening_frame_index)
        if intro_reader is not None:
            intro_reader.close()

        for map_frame_index in range(map_frames):
            frame_index = opening_frames + map_frame_index
            video_time = map_frame_index / fps
            timeline_time = min(max(video_time, 0.0), timeline.video_duration)
            moment, _ = timeline.moment_at(timeline_time)
            track_timeline_time = max(
                0.0,
                timeline_time - max(0.0, float(cfg.get("track_lag", 0.2))),
            )
            track_moment, _ = timeline.moment_at(track_timeline_time)

            current_camera: CameraRect | None = None
            if camera_controller is not None:
                camera = camera_controller.at(timeline_time)
                current_camera = camera
                set_camera(cfg, camera)
                camera_key = (
                    round(camera.west, 10),
                    round(camera.east, 10),
                    round(camera.south, 10),
                    round(camera.north, 10),
                )
                if deposited_track_renderer is None:
                    if camera_key != cached_camera_key or cached_camera_canvas is None:
                        cached_camera_canvas = (
                            render_camera_map(source_map, camera, cfg)
                            if source_map is not None
                            else Image.new("RGB", (width, height), "white")
                        )
                        cached_camera_key = camera_key
                    canvas = cached_camera_canvas
                else:
                    # The batched renderer returns the map with its deposited
                    # path history already baked in.
                    canvas = background
            else:
                canvas = background.copy()
                cfg["_map_zoom"] = 1.0
            icon_factor = magnification_factor(cfg, "tc_icon_mag")
            label_font_size = max(
                1, int(round(float(cfg["label_size"]) * icon_factor))
            )
            if label_font_size not in font_cache:
                font_cache[label_font_size] = load_font(
                    label_font_size,
                    bold=True,
                    font_file=configured_font,
                )
            icon_font = font_cache[label_font_size]
            if deposited_track_renderer is not None and current_camera is not None:
                canvas, track_layer = deposited_track_renderer.frame(
                    track_moment, current_camera, video_time
                )
            else:
                track_layer = Image.new(
                    "RGBA", (width, height), (0, 0, 0, 0)
                )
                draw = ImageDraw.Draw(track_layer)
                for track in tracks:
                    points = track.points
                    if track_moment < points[0].time:
                        continue
                    visible_track_moment = min(track_moment, points[-1].time)
                    if cfg.get("track_antialias", True):
                        track_layer = draw_track_native_aa(
                            track_layer, points, visible_track_moment, cfg
                        )
                    else:
                        draw_track(draw, points, visible_track_moment, cfg)

            canvas = Image.alpha_composite(
                canvas.convert("RGBA"), track_layer
            ).convert("RGB")

            # Fixed lower layers: landfall effects, then timeline.
            pixels = np.asarray(canvas).copy()
            if prepared_landfalls:
                key_points = cfg.get("landfall_icon_key_points", {})
                effect_mode = str(
                    cfg.get("landfall_icon_composite_mode", "screen")
                ).lower()
                for landfall in landfalls:
                    if not (timeline.start <= landfall.time <= timeline.end):
                        continue
                    clip = prepared_landfalls.get(landfall.state)
                    if clip is None:
                        continue
                    landfall_scale = max(
                        0.01, float(cfg.get("landfall_icon_scale", 1.0))
                    )
                    effect_width = max(
                        1, int(round(clip.info.width * landfall_scale))
                    )
                    effect_height = max(
                        1,
                        int(round(clip.info.height * landfall_scale)),
                    )
                    key_point = float(
                        key_points.get(
                            landfall.state,
                            key_points.get(
                                landfall.state.upper(),
                                cfg.get("landfall_icon_key_point", 0.8),
                            ),
                        )
                    )
                    event_video_time = timeline.video_offset_for(landfall.time)
                    effect_elapsed = video_time - (
                        event_video_time - key_point
                    )
                    effect_duration = len(clip.rgb) / float(clip.info.fps)
                    if not 0.0 <= effect_elapsed < effect_duration:
                        continue
                    effect_index = min(
                        len(clip.rgb) - 1,
                        int(math.floor(effect_elapsed * float(clip.info.fps))),
                    )
                    effect_alpha = (
                        clip.alpha
                        if clip.alpha.ndim == 2
                        else clip.alpha[effect_index]
                    )
                    effect_frame = hue_shift_icon(
                        resize_icon_frame(
                            (clip.rgb[effect_index], effect_alpha),
                            (effect_width, effect_height),
                            effect_mode,
                        ),
                        landfall.state,
                        float(landfall.intensity),
                    )
                    paste_rgba(
                        pixels,
                        effect_frame[0],
                        effect_frame[1],
                        project(landfall.lat, landfall.lon, cfg),
                        1.0,
                        effect_mode,
                    )

            canvas = Image.fromarray(pixels, "RGB")
            if timeline_reader is not None:
                timeline_rgb = timeline_reader.frame_at(
                    timeline_source_seconds(moment, cfg)
                )
                timeline_h, timeline_w = timeline_rgb.shape[:2]
                margin_x = int(cfg.get("timeline_margin_x", 0))
                margin_y = int(cfg.get("timeline_margin_y", 0))
                timeline_pixels = np.asarray(canvas).copy()
                paste_rgba(
                    timeline_pixels,
                    timeline_rgb,
                    np.ones((timeline_h, timeline_w), dtype=np.float32),
                    (
                        margin_x + timeline_w / 2.0,
                        height - margin_y - timeline_h / 2.0,
                    ),
                    1.0,
                    "screen",
                )
                canvas = Image.fromarray(timeline_pixels, "RGB")

            if ace_series is not None:
                canvas = draw_ace_bar(
                    canvas,
                    ace_series.value_at(moment),
                    cfg,
                    configured_font,
                    font_cache,
                )

            # TC groups are painted oldest first. Within each group the icon is
            # painted first and its own label immediately afterward, so a newer
            # storm (icon and label together) covers an older storm as a unit.
            pixels = np.asarray(canvas).copy()
            for track in tracks:
                points = track.points
                start_video = timeline.video_offset_for(points[0].time)
                end_video = timeline.video_offset_for(points[-1].time)
                fade = max(0.001, float(cfg["fade_seconds"]))
                if video_time < start_video or video_time > end_video:
                    continue
                fade_in = smoothstep((video_time - start_video) / fade)
                fade_out = smoothstep((end_video - video_time) / fade)
                visibility = min(fade_in, fade_out)
                if visibility <= 0.0:
                    continue

                track_moment = min(max(moment, points[0].time), points[-1].time)
                point_index, lat, lon = point_at(points, track_moment, cfg)
                center = project(lat, lon, cfg)
                state0, state1, state_weight = transition_at(
                    points, point_index, track_moment, cfg, "state"
                )
                current_wind = wind_at(points, track_moment)
                phase = (
                    track.phase_offset + max(0.0, video_time - start_video)
                ) % float(loop_period)
                composite_mode = str(
                    cfg.get("icon_composite_mode", "screen")
                ).lower()
                old_icon = hue_shift_icon(
                    resize_icon_frame(
                        sample_clip(prepared[state0], phase),
                        int(round(base_target_sizes[state0] * icon_factor)),
                        composite_mode,
                    ),
                    state0,
                    current_wind,
                )
                if state0 == state1:
                    icon_rgb, icon_alpha = old_icon
                else:
                    new_icon = hue_shift_icon(
                        resize_icon_frame(
                            sample_clip(prepared[state1], phase),
                            int(round(base_target_sizes[state1] * icon_factor)),
                            composite_mode,
                        ),
                        state1,
                        current_wind,
                    )
                    icon_rgb, icon_alpha = mix_icons(
                        old_icon, new_icon, state_weight, composite_mode
                    )
                paste_rgba(
                    pixels,
                    icon_rgb,
                    icon_alpha,
                    center,
                    visibility,
                    composite_mode,
                )
                canvas = Image.fromarray(pixels, "RGB")
                label0, label1, label_weight = transition_at(
                    points, point_index, track_moment, cfg, "label"
                )
                canvas = composite_storm_label(
                    canvas,
                    center,
                    label0,
                    label1,
                    label_weight,
                    visibility,
                    max(icon_rgb.shape[0], icon_rgb.shape[1]),
                    cfg,
                    icon_font,
                    icon_factor,
                )
                pixels = np.asarray(canvas).copy()

            map_fade_in = (
                smoothstep(video_time / intro)
                if opening_frames > 0 and intro > 0.0
                else 1.0
            )
            if map_fade_in < 1.0:
                faded = np.asarray(canvas).astype(np.float32) * map_fade_in
                canvas = Image.fromarray(
                    np.clip(faded, 0, 255).astype(np.uint8), "RGB"
                )
            encode_canvas(canvas, frame_index)

        for packet in stream.encode():
            container.mux(packet)
        feed_audio_until(total_audio_samples)
        if audio_stream is not None:
            for packet in audio_stream.encode():
                container.mux(packet)
    if timeline_reader is not None:
        timeline_reader.close()
    print(f"\r编码完成：{total_frames}/{total_frames} 帧")
    print(f"输出：{output_path.resolve()}")
    input("请按任意键继续...")


def main() -> None:
    print("风季动画自动化制作程序启动！")
    args = parse_args()
    cfg = load_config(ROOT / "config.py")
    dat_paths = find_dats(args.dat)
    configured_output = Path(cfg["output"])
    output_path = args.output or configured_output
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    try:
        render(dat_paths, output_path, cfg)
    finally:
        # TC and landfall frame arrays are tied to this render and cannot be
        # reused safely. Timeline proxies remain in generation_cache because
        # their filenames fully describe reusable source/crop parameters.
        cleanup_session_preprocess_cache()


if __name__ == "__main__":
    main()
