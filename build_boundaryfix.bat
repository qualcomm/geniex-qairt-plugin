@echo off
setlocal
set VS=C:\Program Files\Microsoft Visual Studio\18\Community
call "%VS%\VC\Auxiliary\Build\vcvarsall.bat" amd64_arm64 || exit /b 1
cd /d C:\Users\zackli\zackli\qualcomm-code\geniex-qairt-plugin
cmake --build build-v73 --config Release --target gemma4_e2b --target gemma4_vlm -j32 -- /p:TrackFileAccess=false || exit /b 1
echo BUILD_OK
