#!/usr/bin/env python3
"""台风动画生成 API 服务
提供 REST API 接口，用于：
1. 上传 ATCF b-deck 文件
2. 运行台风动画生成程序
3. 下载生成的 MP4 文件

本地运行：
    python api_server.py

部署到 Render：
    gunicorn --bind 0.0.0.0:$PORT --timeout 1800 api_server:app
"""
import os
import sys
import uuid
import shutil
import subprocess
import tempfile
import threading
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


import re as _re

_STEP_RE = _re.compile(r'^\[STEP (\d+)/(\d+)\]\s*(.+)$')
_ERROR_RE = _re.compile(r'^\[ERROR\]\s*(.+)$')
_DONE_RE = _re.compile(r'^\[DONE\]\s*(.+)$')

TOTAL_STEPS = 12


def run_animation_task(task_id, dat_files, output_path, animation_script):
    """在后台线程中执行动画生成，实时捕获输出并解析分步进度"""
    task = tasks[task_id]
    task['log'] = []
    task['current_step'] = 0
    task['failed_step'] = None
    task['error_detail'] = None
    try:
        cmd = [
            sys.executable, '-u', str(animation_script),
            '--dat', str(dat_files[0]),
            '--output', str(output_path)
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(PROJECT_ROOT),
            bufsize=1,
        )

        last_step = 0
        for line in proc.stdout:
            line = line.rstrip('\n')
            task['log'].append(line)

            m = _STEP_RE.match(line)
            if m:
                step_num = int(m.group(1))
                last_step = step_num
                task['current_step'] = step_num
                task['progress'] = int(step_num / TOTAL_STEPS * 95)
                task['message'] = f'步骤 {step_num}/{TOTAL_STEPS}：{m.group(3)}'
                continue

            m = _ERROR_RE.match(line)
            if m:
                task['failed_step'] = last_step
                task['error_detail'] = m.group(1)
                continue

            m = _DONE_RE.match(line)
            if m:
                task['current_step'] = TOTAL_STEPS
                task['progress'] = 100
                continue

        proc.wait(timeout=1800)
        stderr_output = proc.stderr.read() if proc.stderr else ''
        if stderr_output.strip():
            task['log'].append(f'[STDERR] {stderr_output.strip()}')

        if proc.returncode == 0 and output_path.exists():
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
        task['status'] = 'error'
        task['message'] = (
            f'生成超时（超过30分钟），最后完成到步骤 {task.get("current_step", 0)}/{TOTAL_STEPS}'
        )
    except Exception as e:
        task['status'] = 'error'
        task['message'] = str(e)


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """上传 ATCF b-deck 文件"""
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    if not file.filename.endswith('.dat'):
        return jsonify({'error': '只支持 .dat 格式的 ATCF 文件'}), 400

    task_id = str(uuid.uuid4())[:8]
    task_dir = UPLOAD_FOLDER / task_id
    task_dir.mkdir(exist_ok=True)

    filename = file.filename
    file_path = task_dir / filename
    file.save(str(file_path))

    tasks[task_id] = {
        'status': 'uploaded',
        'file': filename,
        'output': None,
        'progress': 0,
        'message': '文件上传成功'
    }

    return jsonify({
        'task_id': task_id,
        'filename': filename,
        'message': '文件上传成功'
    })


@app.route('/api/generate', methods=['POST'])
def generate_animation():
    """生成台风动画（异步）"""
    data = request.get_json()
    task_id = data.get('task_id')

    if not task_id or task_id not in tasks:
        return jsonify({'error': '无效的任务ID'}), 400

    task = tasks[task_id]
    if task['status'] == 'processing':
        return jsonify({'error': '任务正在处理中'}), 400

    task['status'] = 'processing'
    task['progress'] = 0
    task['message'] = '开始生成动画...'

    task_dir = UPLOAD_FOLDER / task_id
    dat_files = list(task_dir.glob('*.dat'))

    if not dat_files:
        task['status'] = 'error'
        task['message'] = '找不到 .dat 文件'
        return jsonify({'error': '找不到 .dat 文件'}), 400

    output_filename = f"typhoon_animation_{task_id}.mp4"
    output_path = OUTPUT_FOLDER / output_filename
    animation_script = PROJECT_ROOT / 'animation.py'

    thread = threading.Thread(
        target=run_animation_task,
        args=(task_id, dat_files, output_path, animation_script),
        daemon=True
    )
    thread.start()

    return jsonify({
        'task_id': task_id,
        'status': 'processing',
        'message': '动画生成已启动，请轮询状态'
    })


@app.route('/api/status/<task_id>', methods=['GET'])
def get_status(task_id):
    """查询任务状态（含分步诊断信息）"""
    if task_id not in tasks:
        return jsonify({'error': '任务不存在'}), 404

    task = tasks[task_id]
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
    })


@app.route('/api/log/<task_id>', methods=['GET'])
def get_log(task_id):
    """获取任务完整运行日志"""
    if task_id not in tasks:
        return jsonify({'error': '任务不存在'}), 404

    task = tasks[task_id]
    return jsonify({
        'task_id': task_id,
        'log': task.get('log', []),
    })


@app.route('/api/download/<task_id>', methods=['GET'])
def download_file(task_id):
    """下载生成的动画文件"""
    if task_id not in tasks:
        return jsonify({'error': '任务不存在'}), 404

    task = tasks[task_id]
    if task['status'] != 'completed' or not task.get('output'):
        return jsonify({'error': '文件未生成'}), 404

    output_path = OUTPUT_FOLDER / task['output']
    if not output_path.exists():
        return jsonify({'error': '文件不存在'}), 404

    return send_file(
        str(output_path),
        as_attachment=True,
        download_name=task['output']
    )


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'message': '台风动画生成服务运行正常'
    })


if __name__ == '__main__':
    print("=" * 60)
    print("台风动画生成 API 服务")
    print("=" * 60)
    print(f"上传目录: {UPLOAD_FOLDER}")
    print(f"输出目录: {OUTPUT_FOLDER}")
    print(f"项目根目录: {PROJECT_ROOT}")
    print("=" * 60)
    print("启动服务: http://localhost:5000")
    print("API 文档:")
    print("  POST /api/upload        - 上传 ATCF 文件")
    print("  POST /api/generate      - 生成动画")
    print("  GET  /api/status/<id>   - 查询任务状态")
    print("  GET  /api/download/<id> - 下载生成的文件")
    print("=" * 60)

    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
