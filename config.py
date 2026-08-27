"""TC路径动画配置。修改后直接运行 animation.py 或 preview_config.py。"""


CONFIG = {
    # 输出视频：resolution 只能是 720 或 1080；fps 只能是 30 或 60。分辨率和帧率越高，渲染速度越慢。
    "resolution": 720,
    "fps": 30,
    "output": "season_animation.mp4",

    # 素材文件名。视频支持 mp4 / mov / avi；地图支持 jpg / jpeg / png / webp。
    "map_file": "西北太平洋精细地图.png",
    "timeline_video_file": "时间轴.mp4",
    "intro_video_file": "intro.mp4",
    "font_file": "arial.ttf",

    # 片头视频与标题。
    "intro_video_duration": 8.0,
    "intro_video_fade_out_seconds": 1.0,
    "intro_title": "2025\nSuper Typhoon Ragasa\nTrack Animation",
    "intro_title_size": 92,
    "intro_title_line_spacing": 70,
    "intro_title_center_y": 0.50,
    "intro_title_pop_start": 0.35,
    "intro_title_pop_duration": 0.85,
    "intro_title_start_scale": 0.35,
    "intro_title_color": "#FFFFFF",
    "intro_title_stroke_width": 3,
    "intro_title_stroke_color": "#000000",
    "intro_title_shadow_offset_x": 6,
    "intro_title_shadow_offset_y": 7,
    "intro_title_shadow_blur": 7.0,
    "intro_title_shadow_opacity": 0.80,
    "intro_title_shadow_color": "#000000",
    "intro_title_inner_shadow_offset_x": 3,
    "intro_title_inner_shadow_offset_y": 3,
    "intro_title_inner_shadow_blur": 3.0,
    "intro_title_inner_shadow_opacity": 0.42,
    "intro_title_inner_shadow_color": "#202020",
    "intro_title_inner_highlight_offset_x": 2,
    "intro_title_inner_highlight_offset_y": 2,
    "intro_title_inner_highlight_blur": 2.0,
    "intro_title_inner_highlight_opacity": 0.38,
    "intro_title_inner_highlight_color": "#FFFFFF",

    # 音乐顺序随机种子与输出码率。
    "seed": 2026,
    "music_bitrate": 192000,

    # 时间映射与画面首尾。
    "timeline_sec_per_day": 3.0,
    "timeline_is_leap_year": False, #看你的时间轴是不是闰年，如果填错了会差1天
    "timeline_edge_buffer_hours": 12.0,
    "intro_seconds": 1.0,
    "fade_seconds": 0.75,
    "transition_seconds": 0.35,

    # 超过指定天数没有活跃气旋时，中间时段快进；两端保留 buffer。
    "fast_forward_gap_days": 3.0,
    "fast_forward_buffer_hours": 12.0,
    "fast_forward_multiplier": 20.0,

    # TC 图标。tc_icon_mag 控制地图每缩放一个单位对图标和标签大小的影响。0为图标完全不随着地图缩放,1为等比例缩放。
    "icon_size": 150,
    "tc_icon_mag": 0.125,
    "icon_state_scales": {},
    "icon_preserve_source_canvas": True,
    "icon_composite_mode": "screen",
    "icon_glow_crop_threshold": 3,

    # TC 图标去黑底。
    "black_key_low": 82,
    "black_key_high": 118,
    "black_key_erode_pixels": 1,

    # 名字标签及飞入、飞出动画。
    "label_size": 28,
    "label_stroke_width": 2,
    "label_offset_x_icons": 0.2,
    "label_offset_y_icons": -0.24,
    "label_transition_seconds": 0.35,
    "label_fly_x_icons": 0.85,
    "label_fly_y_icons": -0.85,
    "label_shadow_offset_x": 3,
    "label_shadow_offset_y": 3,
    "label_shadow_blur": 3.0,
    "label_shadow_opacity": 0.80,
    "promote_name_at_ts": True,
    "name_lookahead_hours": 6.0,

    # 地图覆盖范围。
    "latitude_bottom": 0.0,
    "latitude_top": 60.0,
    "longitude_left": 96.6,
    "longitude_right": 180.0,

    # 自动缩放。最大值 3.0 表示最多放大到 300%。
    "auto_zoom": True,
    "auto_zoom_buffer": 0.20,
    "auto_zoom_min_span_degrees": 2.0,
    "auto_zoom_max_scale": 3.0,

    # 路径。开启抗锯齿时使用 OpenCV 原生 LINE_AA。
    "show_track": True,
    "track_opacity": 0.70,
    "track_width": 1,
    "track_width_mag": 1.0, #控制地图每缩放一个单位对路径宽度的影响。推荐1，否则缩放会导致不同TC路径宽度不一致。
    "track_antialias": True,
    # 路径相对 TC 图标延迟出现的“视频秒数”，快进段也按视频时间计算。
    "track_lag": 0.2,
    # 每隔多少视频秒把完整的抗锯齿历史路径沉淀进地图。
    # 两次沉淀之间只逐帧绘制这段时间内新增的路径。
    "track_deposit_interval_seconds": 5.0,
    "path_samples_per_segment": 12,
    "path_smoothing_tension": 0.5, # 平滑TC路径
    "track_colors": {
        "low": "#C0EEEE",
        "td": "#00A0FF",
        "ts": "#00FF00",
        "c1": "#FFFF03",
        "c2": "#FFC003",
        "c3": "#FF9503",
        "c4": "#FF5803",
        "c5": "#FF00A5",
        "sd": "#B783F2",
        "ss": "#83F2B7",
        "ex": "#E0E0E0",
    },
    "track_styles": {
        "low": {
            "pattern": "dotted",
            "opacity": 0.30,
            "spacing_widths": 3.5,
            "dot_radius_widths": 0.5,
        },
        "sd": {"pattern": "dashed", "dash_widths": 3.0, "gap_widths": 3.0},
        "ss": {"pattern": "dashed", "dash_widths": 3.0, "gap_widths": 3.0},
        "ex": {"pattern": "dashed", "dash_widths": 3.0, "gap_widths": 3.0, "opacity": 0.30},
    },

    # 根据海岸线自动检测登陆，并播放 landfall_icons 中的对应特效。
    "show_landfall_effects": True,
    "coastline_file": "ne_10m_coastline.shp",
    "coastline_grid_degrees": 1.0,
    "landfall_detection_samples_per_segment": 120,
    "landfall_deduplicate_minutes": 30.0,
    "landfall_icon_scale": 0.7,
    "landfall_icon_composite_mode": "screen",
    "landfall_icon_black_floor": 0,
    "landfall_icon_key_point": 0.13, # 关键点，即登陆那一瞬间在登陆图标素材视频中的位置秒数，或者说前摇长度。视素材而定。
    # 可按强度分别决定关键点，覆盖刚才的选项，例如：{"td": 0.08, "ts": 0.15}，可留空
    "landfall_icon_key_points": {},

    # 左下角时间轴素材。
    "timeline_scale": 0.25,
    "timeline_margin_x": 0,
    "timeline_margin_y": 0,
    "timeline_black_floor": 16,
    "timeline_cache_crf": 20,
    "timeline_seek_threshold_seconds": 2.0,

    # 左上角累计气旋能量（ACE）条。
    "show_ace_bar": True,
    # 以下尺寸和坐标均以 720p 为基准，1080p 会按比例放大。
    # ACE 白色外框相对画面左上角的位置及其宽、高。
    "ace_bar_offset_x": 26,
    "ace_bar_offset_y": 40,
    "ace_bar_width": 450,
    "ace_bar_height": 45,
    "ace_bar_border_width": 2,
    # 内部彩色 ACE 条与白色外框四边的距离。
    "ace_bar_inner_margin": 6,
    # ACE栏标题相对画面左上角的位置和字号。
    "ace_title_offset_x": 26,
    "ace_title_offset_y": 10,
    "ace_title_size": 20,
    # 右侧 ACE 数字的字号。
    "ace_value_size": 32,
    # 右侧 ACE 数字与白色外框右边界的距离。
    "ace_value_right_margin": 10,
    # ACE条容量和色阶。即使ACE条装满，数字依然会继续增长
    "ace_bar_upper_limit": 500.0,
    "ace_bar_colors": {
        0: "#34ebcc",
        100: "#47ef42",
        200: "#ffff10",
        400: "#ff5305",
        500: "#ff01a8",
    },
    # False：ACE只计算纯热带系统；True：同时计算副热带系统。
    "ace_includes_subtc": False,

    # 901 帧等非整数帧数素材的自动规范化容差。
    "normalize_single_trailing_frame": True,
    "normalize_fps_tolerance": 0.05,
    "normalize_duration_tolerance": 0.10,

    # 编码器。auto 优先使用 Intel QSV，失败时回退到 libx264。
    "encoder": "auto",
    "qsv_codec": "h264_qsv",
    "qsv_global_quality": 14,
    "qsv_preset": "medium",
    "qsv_async_depth": 4,
    "crf": 20,
    "preset": "medium",
}
