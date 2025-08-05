#!/usr/bin/env python3
"""
Diagnostic Script - Checks for main.py import issues (Updated for OpenAI Adapter)
"""
import sys
import os
import traceback
from pathlib import Path

def check_files_exist():
    print("\nChecking required files:")
    src_path = Path('src')
    if not src_path.is_dir():
        print(f"  Error: `src` directory not found!")
        return

    # [修正] 更新了检查的文件列表
    required_files = [
        src_path / 'main.py',
        src_path / 'config.py',
        src_path / 'openai_adapter.py', # 检查新文件
        Path('requirements.txt'),
        Path('Dockerfile')
    ]
    
    # [移除] 不再检查 anthropic_api.py
    
    for file_path in required_files:
        exists = file_path.exists()
        print(f"  {file_path}: {'✓' if exists else '✗'}")
        if not exists:
            print(f"    Error: Missing file {file_path}")

def check_imports():
    print("\nChecking third-party library imports:")
    third_party_libs = [
        'fastapi', 'uvicorn', 'pydantic', 'loguru',
        'google.generativeai', 'dotenv', 'cachetools'
    ]
    for lib in third_party_libs:
        try:
            __import__(lib.split('.')[0])
            print(f"  {lib}: ✓")
        except ImportError as e:
            print(f"  {lib}: ✗ - {e}")

def check_main_module():
    print("\nChecking main application module (src.main):")
    sys.path.insert(0, os.getcwd())
    try:
        from src import main as app_main
        print("  Importing src.main module: ✓")

        if hasattr(app_main, 'app'):
            print("  Found 'app' attribute: ✓")
        else:
            print("  Found 'app' attribute: ✗")
    except ImportError as e:
        print(f"  Importing src.main module: ✗ - {e}")
        traceback.print_exc()
    finally:
        if sys.path[0] == os.getcwd():
            sys.path.pop(0)

def main():
    print("=== OpenAI-Compatible Gemini Adapter Diagnostic Tool ===\n") # [修正] 更新脚本标题
    check_files_exist()
    check_imports()
    check_main_module()
    print("\n=== Diagnostics Complete ===")

if __name__ == "__main__":
    main()
