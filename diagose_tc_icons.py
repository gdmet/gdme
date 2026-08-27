#!/usr/bin/env python3
"""诊断 tc_icons 目录下 MP4 视频素材的帧解码问题。

排查方向：
1. PyAV stream.frames 元数据是否与实际可解码帧数一致
2. 编码器/编码格式是否导致只解码出首帧
3. 视频是否真的只有一帧（导出错误）
4. loop_period / cycle_frames 计算是否导致只取一帧
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from fractions import Fraction

try:
    import av
    import numpy as np
except ImportError as exc:
    print(f"缺少依赖: {exc}")
    print("请先运行 env_initialize.bat 或 pip install av numpy")
    sys.exit(1)


def diagnose_file(path: Path) -> None:
    print(f"\n{'='*70}")
    print(f"诊断文件: {path.name}")
    print(f"文件大小: {path.stat().st_size:,} 字节")
    print(f"{'='*70}")

    # ── 1. 容器和流元数据 ──
    with av.open(str(path)) as container:
        print(f"\n[容器信息]")
        print(f"  格式: {container.format.name} ({container.format.long_name})")
        print(f"  时长: {float(container.duration or 0) / av.time_base:.4f} 秒")
        print(f"  流数量: {len(container.streams)}")

        if not container.streams.video:
            print("  ❌ 没有视频流！")
            return

        stream = container.streams.video[0]

        print(f"\n[视频流元数据]")
        print(f"  编码: {stream.codec_context.name} ({stream.codec_context.long_name})")
        print(f"  编解码器 profile: {stream.codec_context.profile}")
        print(f"  分辨率: {stream.width}x{stream.height}")
        print(f"  像素格式: {stream.codec_context.pix_fmt}")
        print(f"  average_rate: {stream.average_rate} ({float(stream.average_rate):.4f} fps)")
        print(f"  base_rate: {stream.base_rate}")
        print(f"  time_base: {stream.time_base}")
        print(f"  duration: {stream.duration} (秒: {float(stream.duration * stream.time_base):.4f})")

        # 关键：stream.frames 的值
        reported_frames = stream.frames
        print(f"  stream.frames (元数据): {reported_frames}")

        # 检查是否有 codec delay / has_b_frames
        print(f"  codec_context.has_b_frames: {stream.codec_context.has_b_frames}")

        # ── 2. 实际解码帧数 ──
        print(f"\n[实际解码测试]")
        decoded_frames = []
        frame_count = 0
        first_frame_data = None

        for frame in container.decode(stream):
            frame_count += 1
            pts = frame.pts
            time_sec = float(frame.pts * frame.time_base) if frame.pts is not None else None
            key_frame = frame.key_frame

            # 记录前5帧和后5帧的详细信息
            if frame_count <= 5:
                arr = frame.to_ndarray(format="rgb24")
                mean_val = arr.mean()
                # 计算与第一帧的差异
                if first_frame_data is None:
                    first_frame_data = arr.copy()
                    diff_from_first = 0.0
                else:
                    diff_from_first = np.abs(arr.astype(float) - first_frame_data.astype(float)).mean()

                print(f"  帧 {frame_count}: pts={pts}, time={time_sec:.4f}s, "
                      f"key_frame={key_frame}, "
                      f"size={arr.shape}, mean_rgb={mean_val:.1f}, "
                      f"diff_from_first={diff_from_first:.1f}")

            # 每50帧记录一次
            elif frame_count % 50 == 0:
                arr = frame.to_ndarray(format="rgb24")
                if first_frame_data is not None:
                    diff = np.abs(arr.astype(float) - first_frame_data.astype(float)).mean()
                else:
                    diff = 0.0
                print(f"  帧 {frame_count}: mean_rgb={arr.mean():.1f}, diff_from_first={diff:.1f}")

        total_decoded = frame_count
        print(f"\n  总解码帧数: {total_decoded}")

    # ── 3. 对比分析 ──
    print(f"\n[诊断结论]")

    if reported_frames is not None and reported_frames > 0:
        print(f"  stream.frames 报告: {reported_frames} 帧")
    else:
        print(f"  stream.frames 报告: 0 或 None (需要手动计数)")

    print(f"  实际解码帧数: {total_decoded} 帧")

    if total_decoded <= 1:
        print(f"\n  ❌ 问题确认：只能解码出 {total_decoded} 帧！")
        print(f"  可能原因:")
        print(f"    1. MP4 文件本身只有一帧（导出错误）")
        print(f"    2. 编码器使用了 PyAV 不支持的编码方式")
        print(f"    3. 文件损坏或不完整")
        print(f"  建议:")
        print(f"    - 用 ffprobe -v error -count_frames -select_streams v:0 "
              f"-show_entries stream=nb_read_frames \"{path.name}\" 验证")
        print(f"    - 用 ffmpeg -i \"{path.name}\" -f null - 检查解码错误")
        print(f"    - 重新导出视频素材，确保使用标准 H.264 编码")
    elif reported_frames and reported_frames != total_decoded:
        print(f"\n  ⚠️ 帧数不一致！元数据报告 {reported_frames}，实际解码 {total_decoded}")
        print(f"  这会导致 prepare_clip 只处理 {min(reported_frames, total_decoded)} 帧")
        print(f"  建议: 用 ffmpeg 重新封装: ffmpeg -i \"{path.name}\" -c copy fixed_{path.name}")
    else:
        print(f"\n  ✅ 帧数一致，解码正常")

    # ── 4. 检查帧间差异（判断是否所有帧内容相同）──
    if total_decoded > 1:
        print(f"\n[帧间差异分析]")
        with av.open(str(path)) as container2:
            stream2 = container2.streams.video[0]
            frames_data = []
            for i, frame in enumerate(container2.decode(stream2)):
                if i >= 100:  # 只检查前100帧
                    break
                arr = frame.to_ndarray(format="rgb24")
                frames_data.append(arr)

        all_same = True
        for i in range(1, len(frames_data)):
            diff = np.abs(frames_data[i].astype(float) - frames_data[0].astype(float)).mean()
            if diff > 1.0:
                all_same = False
                print(f"  帧 1 vs 帧 {i+1}: 平均差异 = {diff:.2f} (有变化)")
                break

        if all_same:
            print(f"  ⚠️ 前 {len(frames_data)} 帧内容完全相同！")
            print(f"  视频素材可能是静态图片被封装成了视频")
        else:
            print(f"  ✅ 帧间存在差异，素材确实是动画")

    # ── 5. 模拟 prepare_clip 的行为 ──
    print(f"\n[sample_clip 模拟]")
    with av.open(str(path)) as container3:
        stream3 = container3.streams.video[0]
        rate = Fraction(stream3.average_rate)
        meta_frames = int(stream3.frames or 0)
        if not meta_frames:
            meta_frames = total_decoded

    # 模拟 loop_period 计算（假设所有素材时长相同）
    duration = Fraction(meta_frames, 1) / rate if rate else Fraction(0)
    print(f"  素材时长: {float(duration):.4f} 秒")
    print(f"  素材帧率: {float(rate):.2f} fps")
    print(f"  probe_clip 报告的帧数: {meta_frames}")

    # 假设 loop_period = duration（单素材情况）
    loop_period = duration
    cycle_frames = max(1, int(loop_period * rate))
    frames_to_process = min(cycle_frames, meta_frames) if meta_frames else cycle_frames
    print(f"  loop_period: {float(loop_period):.4f} 秒")
    print(f"  cycle_frames: {cycle_frames}")
    print(f"  frames_to_process: {frames_to_process}")

    if frames_to_process <= 1:
        print(f"\n  ❌ 关键问题：frames_to_process = {frames_to_process}！")
        print(f"  prepare_clip 只会处理 {frames_to_process} 帧，其余帧用最后一帧填充")
        print(f"  这就是为什么 TC 图标看起来是静止的！")


def main():
    root = Path(__file__).resolve().parent
    tc_icons_dir = root / "tc_icons"

    if not tc_icons_dir.is_dir():
        print(f"找不到 tc_icons 目录: {tc_icons_dir}")
        print("请确保脚本与 tc_icons 目录在同一层级")
        sys.exit(1)

    mp4_files = sorted(tc_icons_dir.glob("*.mp4"))
    if not mp4_files:
        print(f"tc_icons 目录中没有 MP4 文件")
        sys.exit(1)

    print(f"找到 {len(mp4_files)} 个 MP4 文件:")
    for f in mp4_files:
        print(f"  {f.name}")

    # 也检查 mov 和 avi
    other_videos = sorted(
        p for ext in (".mov", ".avi")
        for p in tc_icons_dir.glob(f"*{ext}")
    )
    if other_videos:
        print(f"\n还有 {len(other_videos)} 个其他格式视频:")
        for f in other_videos:
            print(f"  {f.name}")

    for mp4 in mp4_files:
        try:
            diagnose_file(mp4)
        except Exception as exc:
            print(f"\n  ❌ 诊断 {mp4.name} 时出错: {exc}")

    print(f"\n{'='*70}")
    print("诊断完成")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
