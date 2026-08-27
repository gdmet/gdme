#!/usr/bin/env python3
"""
台风动画生成系统 - API 测试脚本
用于验证 API 服务器是否正常工作
"""

import requests
import os
import sys

API_BASE_URL = "http://localhost:5000"

def test_health():
    """测试健康检查"""
    print("1. 测试健康检查...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            print("   ✓ 健康检查通过")
            return True
        else:
            print(f"   ✗ 健康检查失败：{response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ 无法连接到 API 服务器：{e}")
        print("   提示：请先运行 python api_server.py")
        return False

def test_upload():
    """测试文件上传"""
    print("\n2. 测试文件上传...")
    
    # 查找测试文件
    test_file = None
    for filename in os.listdir('.'):
        if filename.endswith('.dat'):
            test_file = filename
            break
    
    if not test_file:
        print("   ⚠ 未找到 .dat 测试文件，跳过上传测试")
        return None
    
    try:
        with open(test_file, 'rb') as f:
            response = requests.post(
                f"{API_BASE_URL}/api/upload",
                files={'file': (test_file, f, 'application/octet-stream')}
            )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ 上传成功：task_id = {data['task_id']}")
            return data['task_id']
        else:
            print(f"   ✗ 上传失败：{response.status_code}")
            return None
    except Exception as e:
        print(f"   ✗ 上传错误：{e}")
        return None

def test_generate(task_id):
    """测试动画生成"""
    if not task_id:
        print("\n3. 跳过生成测试（无 task_id）")
        return
    
    print(f"\n3. 测试动画生成 (task_id: {task_id})...")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/generate",
            json={'task_id': task_id}
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✓ 生成成功：{data['output']}")
            return True
        else:
            print(f"   ✗ 生成失败：{response.status_code}")
            return False
    except Exception as e:
        print(f"   ✗ 生成错误：{e}")
        return False

def main():
    print("=" * 50)
    print("  台风动画生成系统 - API 测试")
    print("=" * 50)
    print()
    
    # 测试健康检查
    if not test_health():
        sys.exit(1)
    
    # 测试上传
    task_id = test_upload()
    
    # 测试生成
    test_generate(task_id)
    
    print("\n" + "=" * 50)
    print("  测试完成")
    print("=" * 50)

if __name__ == "__main__":
    main()
