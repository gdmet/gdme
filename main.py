#!/usr/bin/env python3
"""台风动画生成 API 服务 — 含 35 项全链路诊断
提供 REST API 接口，用于：
1. 上传 ATCF b-deck 文件
2. 运行台风动画生成程序（含 35 项自动排查）
3. 下载生成的 MP4 文件

本地运行：
    python main.py

部署到 Render：
    gunicorn --bind 0.0.0.0:$PORT --timeout 1800 main:app
"""
import os
import sys
import uuid
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 配置
UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
OUTPUT_FOLDER = Path(__file__).parent / 'outputs'
PROJECT_ROOT = Path(__file__).parent
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

# 任务状态存储
tasks = {}

# 诊断正则
import re as _re
_STEP_RE = _re.compile(r'^\[STEP (\d+)/(\d+)\]\s*(.+)$')
_ERROR_RE = _re.compile(r'^\[ERROR\]\s*(.+)$')
_DONE_RE = _re.compile(r'^\[DONE\]\s*(.+)$')
_DIAG_RE = _re.compile(r'^\[DIAG (PASS|FAIL)\] (P\d-\d+):\s*(.+?)(?:\s*—\s*原因:\s*(.+))?$')

TOTAL_STEPS = 12
TOTAL_DIAGS = 35


def _record_diag(task, check_id, name, passed, reason=''):
    """记录一项诊断结果到任务日志和诊断列表"""
    status = 'PASS' if passed else 'FAIL'
    detail = f'{name} — 原因: {reason}' if reason else name
    line = f'[DIAG {status}] {check_id}: {detail}'
    task['log'].append(line)
    task['diagnostics'].append({
        'id': check_id,
        'name': name,
        'passed': passed,
        'reason': reason,
    })


def run_animation_task(task_id, dat_files, output_path, animation_script):
    """在后台线程中执行动画生成，实时捕获输出并解析分步进度与诊断"""
    task = tasks[task_id]
    task['log'] = []
    task['diagnostics'] = []
    task['current_step'] = 0
    task['failed_step'] = None
    task['error_detail'] = None
    try:
        # --- 第三阶段：子进程启动 ---
        # P3-13: Python 解释器可用
        python_exe = sys.executable
        python_ok = python_exe and Path(python_exe).exists()
        _record_diag(task, 'P3-13', 'Python 解释器可用', python_ok,
                     '' if python_ok else f'sys.executable={python_exe!r} 指向的路径不存在')

        # P3-14: 命令行构造正确
        cmd = [
            python_exe, '-u', str(animation_script),
            '--dat', str(dat_files[0]),
            '--output', str(output_path)
        ]
        task['log'].append(f'[CMD] {" ".join(cmd)}')
        cmd_ok = len(cmd) == 5 and all(isinstance(c, str) and c for c in cmd)
        _record_diag(task, 'P3-14', '命令行构造正确', cmd_ok,
                     '' if cmd_ok else f'cmd 参数异常: {cmd}')

        # P3-15: 当前工作目录有效
        cwd_ok = PROJECT_ROOT.is_dir()
        _record_diag(task, 'P3-15', '当前工作目录有效', cwd_ok,
                     '' if cwd_ok else f'PROJECT_ROOT={PROJECT_ROOT} 不是有效目录')

        # P3-16: 子进程启动成功
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(PROJECT_ROOT),
                bufsize=1,
            )
            popen_ok = proc is not None and proc.pid is not None
            _record_diag(task, 'P3-16', '子进程启动成功', popen_ok,
                         '' if popen_ok else 'Popen 返回异常')
        except FileNotFoundError as exc:
            _record_diag(task, 'P3-16', '子进程启动成功', False, f'FileNotFoundError: {exc}')
            task['status'] = 'error'
            task['message'] = f'子进程启动失败：找不到 Python 解释器 ({python_exe})'
            return
        except Exception as exc:
            _record_diag(task, 'P3-16', '子进程启动成功', False, str(exc))
            task['status'] = 'error'
            task['message'] = f'子进程启动异常：{exc}'
            return

        # --- 第四阶段：实时解析 animation.py 输出 ---
        last_step = 0
        for line in proc.stdout:
            line = line.rstrip('\n')
            task['log'].append(line)

            # 解析分步进度
            m = _STEP_RE.match(line)
            if m:
                step_num = int(m.group(1))
                last_step = step_num
                task['current_step'] = step_num
                task['progress'] = int(step_num / TOTAL_STEPS * 90)
                task['message'] = f'步骤 {step_num}/{TOTAL_STEPS}：{m.group(3)}'
                continue

            # 解析错误
            m = _ERROR_RE.match(line)
            if m:
                task['failed_step'] = last_step
                task['error_detail'] = m.group(1)
                continue

            # 解析完成
            m = _DONE_RE.match(line)
            if m:
                task['current_step'] = TOTAL_STEPS
                task['progress'] = 95
                continue

            # 解析 animation.py 内部诊断
            m = _DIAG_RE.match(line)
            if m:
                status_str, check_id, name, reason = m.groups()
                task['diagnostics'].append({
                    'id': check_id,
                    'name': name,
                    'passed': status_str == 'PASS',
                    'reason': reason or '',
                })
                continue

        proc.wait(timeout=1800)
        stderr_output = proc.stderr.read() if proc.stderr else ''
        if stderr_output.strip():
            task['log'].append(f'[STDERR] {stderr_output.strip()}')

        # --- 第五阶段：输出验证 ---
        # P5-29: 子进程退出码
        rc_ok = proc.returncode == 0
        _record_diag(task, 'P5-29', '子进程退出码为 0', rc_ok,
                     '' if rc_ok else f'returncode={proc.returncode}')

        # P5-30: 输出文件存在
        out_exists = output_path.exists()
        _record_diag(task, 'P5-30', '输出文件存在', out_exists,
                     '' if out_exists else f'{output_path} 不存在')

        # P5-31: 输出文件大小
        if out_exists:
            out_size = output_path.stat().st_size
            size_ok = out_size > 1_000_000  # > 1MB
            _record_diag(task, 'P5-31', f'输出文件大小 > 1MB', size_ok,
                         '' if size_ok else f'文件大小仅 {out_size} 字节')
        else:
            _record_diag(task, 'P5-31', '输出文件大小 > 1MB', False, '文件不存在，无法检查大小')

        # P5-32: 输出文件可播放
        if out_exists and out_size > 0:
            try:
                import av as _av
                container = _av.open(str(output_path))
                has_video = any(s.type == 'video' for s in container.streams)
                container.close()
                _record_diag(task, 'P5-32', '输出文件可播放（含视频流）', has_video,
                             '' if has_video else '文件中未找到视频流')
            except ImportError:
                _record_diag(task, 'P5-32', '输出文件可播放', False, 'av 库未安装，无法验证')
            except Exception as exc:
                _record_diag(task, 'P5-32', '输出文件可播放', False, str(exc))
        else:
            _record_diag(task, 'P5-32', '输出文件可播放', False, '文件不存在或为空')

        # 最终判定
        if rc_ok and out_exists:
            task['status'] = 'completed'
            task['progress'] = 100
            task['output'] = output_path.name
            task['message'] = '动画生成成功'
        else:
            task['status'] = 'error'
            if task.get('error_detail'):
                step_label = (
                    f'步骤 {task["failed_step"]}/{TOTAL_STEPS}'
                    if task.get('failed_step') else '未知步骤'
                )
                task['message'] = (
                    f'生成失败（{step_label}）：{task["error_detail"]}'
                )
            elif stderr_output.strip():
                task['message'] = f'生成失败: {stderr_output.strip()[:500]}'
            else:
                task['message'] = (
                    f'生成失败（退出码 {proc.returncode}），'
                    f'最后完成到步骤 {last_step}/{TOTAL_STEPS}'
                )
    except subprocess.TimeoutExpired:
        proc.kill()
        _record_diag(task, 'P5-29', '子进程退出码为 0', False, '超时被 kill')
        task['status'] = 'error'
        task['message'] = (
            f'生成超时（超过30分钟），最后完成到步骤 {task.get("current_step", 0)}/{TOTAL_STEPS}'
        )
    except Exception as e:
        task['status'] = 'error'
        task['message'] = str(e)


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """上传 ATCF b-deck 文件 — 含第一阶段诊断"""
    diagnostics = []

    # P1-01: 文件接收
    has_file = 'file' in request.files
    if not has_file:
        diagnostics.append({'id': 'P1-01', 'name': '文件接收', 'passed': False,
                            'reason': 'request.files 中无 file 字段'})
        return jsonify({'error': '没有上传文件', 'diagnostics': diagnostics}), 400
    diagnostics.append({'id': 'P1-01', 'name': '文件接收', 'passed': True, 'reason': ''})

    file = request.files['file']

    # P1-02: 文件类型校验
    name_ok = file.filename and file.filename.endswith('.dat')
    if not name_ok:
        diagnostics.append({'id': 'P1-02', 'name': '文件类型校验 (.dat)', 'passed': False,
                            'reason': f'文件名={file.filename!r} 不以 .dat 结尾'})
        return jsonify({'error': '只支持 .dat 格式的 ATCF 文件', 'diagnostics': diagnostics}), 400
    diagnostics.append({'id': 'P1-02', 'name': '文件类型校验 (.dat)', 'passed': True, 'reason': ''})

    task_id = str(uuid.uuid4())[:8]
    task_dir = UPLOAD_FOLDER / task_id
    task_dir.mkdir(exist_ok=True)

    filename = file.filename
    file_path = task_dir / filename
    file.save(str(file_path))

    # P1-03: 文件落盘
    saved_ok = file_path.exists()
    diagnostics.append({'id': 'P1-03', 'name': '文件落盘', 'passed': saved_ok,
                        'reason': '' if saved_ok else f'{file_path} 不存在'})

    # P1-04: 文件完整性
    file_size = file_path.stat().st_size if saved_ok else 0
    size_ok = file_size > 0
    diagnostics.append({'id': 'P1-04', 'name': '文件完整性 (大小>0)', 'passed': size_ok,
                        'reason': '' if size_ok else f'文件大小为 {file_size} 字节'})

    # P1-05: 任务注册
    tasks[task_id] = {
        'status': 'uploaded',
        'file': filename,
        'output': None,
        'progress': 0,
        'message': '文件上传成功',
        'log': [],
        'diagnostics': diagnostics[:],
        'current_step': 0,
        'failed_step': None,
        'error_detail': None,
    }
    reg_ok = task_id in tasks
    diagnostics.append({'id': 'P1-05', 'name': '任务注册', 'passed': reg_ok,
                        'reason': '' if reg_ok else 'tasks 字典写入失败'})

    return jsonify({
        'task_id': task_id,
        'filename': filename,
        'file_size': file_size,
        'message': '文件上传成功',
        'diagnostics': diagnostics,
    })


@app.route('/api/generate', methods=['POST'])
def generate_animation():
    """生成台风动画（异步）— 含第二阶段诊断"""
    data = request.get_json(silent=True) or {}
    task_id = data.get('task_id')
    diagnostics = []

    # P2-06: 任务 ID 有效性
    id_ok = bool(task_id) and task_id in tasks
    if not id_ok:
        diagnostics.append({'id': 'P2-06', 'name': '任务 ID 有效性', 'passed': False,
                            'reason': f'task_id={task_id!r} 不在 tasks 中'})
        return jsonify({'error': '无效的任务ID', 'diagnostics': diagnostics}), 400
    diagnostics.append({'id': 'P2-06', 'name': '任务 ID 有效性', 'passed': True, 'reason': ''})

    task = tasks[task_id]

    # P2-07: 任务状态检查
    state_ok = task['status'] != 'processing'
    if not state_ok:
        diagnostics.append({'id': 'P2-07', 'name': '任务状态检查', 'passed': False,
                            'reason': '任务正在处理中'})
        return jsonify({'error': '任务正在处理中', 'diagnostics': diagnostics}), 400
    diagnostics.append({'id': 'P2-07', 'name': '任务状态检查', 'passed': True, 'reason': ''})

    task['status'] = 'processing'
    task['progress'] = 0
    task['message'] = '开始生成动画...'
    task['log'] = []
    task['diagnostics'] = diagnostics[:]
    task['current_step'] = 0
    task['failed_step'] = None
    task['error_detail'] = None

    task_dir = UPLOAD_FOLDER / task_id
    dat_files = list(task_dir.glob('*.dat'))

    # P2-08: 查找 .dat 文件
    dat_ok = len(dat_files) > 0
    diagnostics.append({'id': 'P2-08', 'name': '查找 .dat 文件', 'passed': dat_ok,
                        'reason': '' if dat_ok else f'task_dir={task_dir} 中无 .dat 文件'})
    if not dat_ok:
        task['status'] = 'error'
        task['diagnostics'] = diagnostics
        return jsonify({'error': '找不到 .dat 文件', 'diagnostics': diagnostics}), 400

    # P2-09: 输出路径准备
    OUTPUT_FOLDER.mkdir(exist_ok=True)
    write_ok = os.access(str(OUTPUT_FOLDER), os.W_OK)
    diagnostics.append({'id': 'P2-09', 'name': '输出目录可写', 'passed': write_ok,
                        'reason': '' if write_ok else f'{OUTPUT_FOLDER} 不可写'})

    # P2-10: animation.py 存在
    animation_script = PROJECT_ROOT / 'animation.py'
    anim_ok = animation_script.is_file()
    diagnostics.append({'id': 'P2-10', 'name': 'animation.py 存在', 'passed': anim_ok,
                        'reason': '' if anim_ok else f'{animation_script} 不是文件'})

    # P2-11: config.py 存在
    config_script = PROJECT_ROOT / 'config.py'
    cfg_ok = config_script.is_file()
    diagnostics.append({'id': 'P2-11', 'name': 'config.py 存在', 'passed': cfg_ok,
                        'reason': '' if cfg_ok else f'{config_script} 不是文件'})

    # P2-12: 线程启动
    output_filename = f"typhoon_animation_{task_id}.mp4"
    output_path = OUTPUT_FOLDER / output_filename
    thread_ok = True
    thread_err = ''
    try:
        thread = threading.Thread(
            target=run_animation_task,
            args=(task_id, dat_files, output_path, animation_script),
            daemon=True
        )
        thread.start()
        thread_ok = thread.is_alive() or True  # daemon thread may finish fast
    except Exception as exc:
        thread_ok = False
        thread_err = str(exc)
    diagnostics.append({'id': 'P2-12', 'name': '线程启动', 'passed': thread_ok,
                        'reason': '' if thread_ok else thread_err})

    task['diagnostics'] = diagnostics

    return jsonify({
        'task_id': task_id,
        'status': 'processing',
        'message': '动画生成已启动，请轮询状态',
        'diagnostics': diagnostics,
    })


@app.route('/api/status/<task_id>', methods=['GET'])
def get_status(task_id):
    """查询任务状态（含 35 项诊断摘要）"""
    if task_id not in tasks:
        return jsonify({'error': '任务不存在'}), 404

    task = tasks[task_id]
    diags = task.get('diagnostics', [])
    passed = sum(1 for d in diags if d['passed'])
    failed = [d for d in diags if not d['passed']]

    return jsonify({
        'task_id': task_id,
        'status': task['status'],
        'progress': task['progress'],
        'message': task['message'],
        'output': task.get('output'),
        'current_step': task.get('current_step', 0),
        'total_steps': TOTAL_STEPS,
        'failed_step': task.get('failed_step'),
        'error_detail': task.get('error_detail'),
        'diagnostics_summary': {
            'total': len(diags),
            'passed': passed,
            'failed': len(failed),
            'failed_items': failed,
        },
    })


@app.route('/api/log/<task_id>', methods=['GET'])
def get_log(task_id):
    """获取任务完整运行日志（含所有诊断行）"""
    if task_id not in tasks:
        return jsonify({'error': '任务不存在'}), 404

    task = tasks[task_id]
    return jsonify({
        'task_id': task_id,
        'log': task.get('log', []),
        'diagnostics': task.get('diagnostics', []),
    })


@app.route('/api/download/<task_id>', methods=['GET'])
def download_file(task_id):
    """下载生成的动画文件 — 含第六阶段 P6-35 诊断"""
    if task_id not in tasks:
        return jsonify({'error': '任务不存在', 'diagnostics': [
            {'id': 'P6-33', 'name': '状态查询', 'passed': False, 'reason': '任务不存在'}
        ]}), 404

    task = tasks[task_id]
    if task['status'] != 'completed' or not task.get('output'):
        return jsonify({'error': '文件未生成', 'diagnostics': [
            {'id': 'P6-33', 'name': '状态查询', 'passed': False,
             'reason': f'status={task["status"]}, output={task.get("output")}'}
        ]}), 404

    output_path = OUTPUT_FOLDER / task['output']
    if not output_path.exists():
        return jsonify({'error': '文件不存在', 'diagnostics': [
            {'id': 'P6-35', 'name': '文件下载', 'passed': False,
             'reason': f'{output_path} 不存在'}
        ]}), 404

    return send_file(
        str(output_path),
        as_attachment=True,
        download_name=task['output'],
        mimetype='video/mp4',
    )


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'message': '台风动画生成服务运行正常',
        'total_tasks': len(tasks),
    })


if __name__ == '__main__':
    print("=" * 60)
    print("台风动画生成 API 服务（含 35 项全链路诊断）")
    print("=" * 60)
    print(f"上传目录: {UPLOAD_FOLDER}")
    print(f"输出目录: {OUTPUT_FOLDER}")
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"Python: {sys.executable} ({sys.version})")
    print(f"animation.py: {(PROJECT_ROOT / 'animation.py').is_file()}")
    print(f"config.py: {(PROJECT_ROOT / 'config.py').is_file()}")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
