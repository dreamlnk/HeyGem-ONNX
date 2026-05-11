#!/bin/bash
# HeyGem 流式管线服务端启动脚本 (WSL侧)
# 用法: bash start_wsl_server.sh

cd "$(dirname "$0")"
source venv/bin/activate
echo "启动 HeyGem 流式服务端..."
echo "端口: 7861"
echo "按 Ctrl+C 停止"
python stream_server_v2.py
