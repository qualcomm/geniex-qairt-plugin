@echo off
REM Fresh ARM64 Release build of geniex-qairt (geniex_core + gemma4_e2b) on the X Elite laptop.
REM
REM Three non-obvious workarounds are required on this box:
REM  1. -DGIT_EXECUTABLE - git otherwise resolves to WindowsApps\git.cmd, and cmd.exe
REM     eats the ^ in FetchContent's `git rev-parse HEAD^0` -> "ambiguous argument 'HEAD0'".
REM  2. sentencepiece does file(CREATE_LINK ... SYMBOLIC) for third_party/absl, which
REM     needs Developer Mode/admin. Pre-create it as a junction; its NOT EXISTS guard
REM     then skips the symlink.
REM  3. /p:TrackFileAccess=false - MSBuild FileTracker .tlog paths under
REM     build-v73\third-party\...\abseil-cpp\absl\... blow past MAX_PATH (FTK1011).
setlocal
set VS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
if not exist "%VS%\VC\Auxiliary\Build\vcvarsall.bat" set VS=C:\Program Files\Microsoft Visual Studio\18\Community
call "%VS%\VC\Auxiliary\Build\vcvarsall.bat" amd64_arm64 || exit /b 1

set ROOT=%~dp0
cd /d "%ROOT%"
set PATH=%USERPROFILE%\.cargo\bin;C:\Program Files\CMake\bin;C:\Program Files\Git\cmd;%PATH%

cmake -B build-v73 -A ARM64 -DGENIEX_BUILD_VLM=OFF -DGENIEX_BUILD_EXAMPLES=ON ^
      -DGIT_EXECUTABLE="C:/Program Files/Git/cmd/git.exe" || exit /b 1
cmake --build build-v73 --config Release --target gemma4_e2b --target gemma4_e4b -j32 -- /p:TrackFileAccess=false || exit /b 1
echo BUILD_OK
