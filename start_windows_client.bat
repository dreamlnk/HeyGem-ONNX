@echo off
REM HeyGem 实时流式客户端启动脚本 (Windows侧)
REM 依赖: pip install opencv-python pyvirtualcam sounddevice numpy

echo ========================================
echo HeyGem Live TCP Client
echo ========================================
echo.
echo 用法:
echo   start_windows_client.bat --video e:\480p测试去绿幕的素材.mp4 --virtualcam --loop --mic
echo   start_windows_client.bat --video xxx.mp4 --virtualcam --loop
echo   start_windows_client.bat --virtualcam  (摄像头+OBS)
echo.

cd /d "%~dp0"
python windows_client_tcp.py %*
pause
