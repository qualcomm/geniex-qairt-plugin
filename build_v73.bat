@echo off
REM Fresh ARM64 Release build of geniex-qairt (geniex_core + gemma4_e2b) on the X Elite laptop.
setlocal
set VS=C:\Program Files\Microsoft Visual Studio\18\Community
call "%VS%\VC\Auxiliary\Build\vcvarsall.bat" amd64_arm64 || exit /b 1
cd /d C:\Users\zackli\zackli\qualcomm-code\geniex-qairt-plugin
cmake -B build-v73 -A ARM64 -DGENIEX_BUILD_VLM=OFF -DGENIEX_BUILD_EXAMPLES=ON || exit /b 1
cmake --build build-v73 --config Release --target gemma4_e2b -j32 || exit /b 1
echo BUILD_OK
