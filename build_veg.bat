@echo off
REM Build the standalone gemma4_veg runner (VLM tier) on this X Elite box.
REM Reuses the existing build/ dir so third-party FetchContent + the sentencepiece
REM absl junction workaround are already in place; just flips VLM on.
setlocal
set VS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
if not exist "%VS%\VC\Auxiliary\Build\vcvarsall.bat" set VS=C:\Program Files\Microsoft Visual Studio\18\Community
call "%VS%\VC\Auxiliary\Build\vcvarsall.bat" amd64_arm64 || exit /b 1

set ROOT=%~dp0
cd /d "%ROOT%"
set PATH=%USERPROFILE%\.cargo\bin;C:\Program Files\CMake\bin;C:\Program Files\Git\cmd;%PATH%

cmake -B build -A ARM64 -DGENIEX_BUILD_VLM=ON -DGENIEX_BUILD_EXAMPLES=ON ^
      -DGIT_EXECUTABLE="C:/Program Files/Git/cmd/git.exe" || exit /b 1
cmake --build build --config Release --target gemma4_veg -j32 -- /p:TrackFileAccess=false || exit /b 1
echo BUILD_OK
