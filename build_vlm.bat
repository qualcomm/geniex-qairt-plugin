@echo off
setlocal
set VS=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools
call "%VS%\VC\Auxiliary\Build\vcvarsall.bat" amd64_arm64 || exit /b 1
cd /d C:\Users\zhiyu\OneDrive\Desktop\qualcomm-code\geniex-qairt-plugin
set PATH=%USERPROFILE%\.cargo\bin;C:\Program Files\CMake\bin;C:\Program Files\Git\cmd;%PATH%
cmake -B build-v73 -A ARM64 -DGENIEX_BUILD_VLM=ON -DGENIEX_BUILD_EXAMPLES=ON ^
      -DGIT_EXECUTABLE="C:/Program Files/Git/cmd/git.exe" || exit /b 1
cmake --build build-v73 --config Release --target gemma4_prep_check --target gemma4_vlm --target gemma4_e2b -j32 -- /p:TrackFileAccess=false || exit /b 1
echo BUILD_OK