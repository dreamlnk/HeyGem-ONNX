@echo off
REM HeyGem 实时流式客户端启动脚本 (Windows侧)
REM 依赖: pip install opencv-python requests pyvirtualcam sounddevice numpy

echo ========================================
echo HeyGem Live TCP Client
echo ========================================
echo.
echo 用法: start_windows_client.bat                        -> 摄像头预览
echo       start_windows_client.bat --virtualcam           -> 摄像头+OBS虚拟摄像头
echo       start_windows_client.bat --video video.mp4      -> 视频文件预览
echo       start_windows_client.bat --video video.mp4 --virtualcam -> 视频+OBS
echo.

cd /d "%~dp0"
python windows_client_tcp.py %*
pause
