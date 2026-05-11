@echo off
REM HeyGem 实时流式客户端启动脚本 (Windows侧)
REM 依赖: pip install opencv-python requests pyvirtualcam sounddevice numpy

echo ========================================
echo HeyGem Live Streaming Client
echo ========================================
echo.
echo 用法: 双击运行 -> 预览窗口模式
echo       start_windows_client.bat --virtualcam -> 虚拟摄像头模式
echo.

cd /d "%~dp0"
python windows_client.py %*
pause
