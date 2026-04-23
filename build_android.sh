#!/bin/bash
#=============================================================================
#
#  Copyright (c) 2025 Nexa AI
#  All Rights Reserved.
#  Confidential and Proprietary - Nexa AI Inc.
#
#=============================================================================

# Create a log directory
LOG_DIR="build_logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/android_build_$TIMESTAMP.log"

# Redirect all output to both console and log file
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Build started at $(date)"
echo "=============================================="

# Don't exit immediately on error, we want to log as much as possible
set +e

# Record environment information
echo "Environment Information:"
echo "- Bash version: $BASH_VERSION"
echo "- OS: $(uname -a)"
echo "- CMake version: $(cmake --version | head -n 1)"
echo "- Environment variables:"
env | grep -E "ANDROID|PATH|HOME|CMAKE" | sort

# Check for Rust and required targets
if command -v rustc &> /dev/null; then
    echo "- Rust version: $(rustc --version)"
    echo "- Cargo version: $(cargo --version)"
    
    # Check if the required Rust target is installed
    RUST_REQUIRED_TARGET=""
    if [ "$ABI" = "arm64-v8a" ]; then
        RUST_REQUIRED_TARGET="aarch64-linux-android"
    elif [ "$ABI" = "x86" ]; then
        RUST_REQUIRED_TARGET="i686-linux-android"
    elif [ "$ABI" = "x86_64" ]; then
        RUST_REQUIRED_TARGET="x86_64-linux-android"
    fi
    
    echo "- Required Rust target: $RUST_REQUIRED_TARGET"
    if rustup target list --installed | grep -q "$RUST_REQUIRED_TARGET"; then
        echo "  ✓ Target already installed"
    else
        echo "  ✗ Target not installed, installing now..."
        rustup target add "$RUST_REQUIRED_TARGET"
        if [ $? -ne 0 ]; then
            echo "ERROR: Failed to install Rust target $RUST_REQUIRED_TARGET"
            echo "Please run 'rustup target add $RUST_REQUIRED_TARGET' manually"
            exit 1
        fi
        echo "  ✓ Target installed successfully"
    fi
else
    echo "WARNING: Rust not found in PATH. If your build requires Rust, it may fail."
fi

# Check if NDK path is set
if [ -z "$ANDROID_NDK_ROOT" ]; then
    echo "ERROR: ANDROID_NDK_ROOT environment variable is not set."
    echo "Please set it to your Android NDK installation path, for example:"
    echo "  export ANDROID_NDK_ROOT=/path/to/android-ndk"
    exit 1
fi

echo "- Android NDK path: $ANDROID_NDK_ROOT"
echo "- Android NDK version: $(cat "$ANDROID_NDK_ROOT/source.properties" 2>/dev/null | grep Pkg.Revision || echo "Could not determine NDK version")"
echo "=============================================="

# Set default variables
BUILD_DIR="build-android"
ABI="arm64-v8a"  # Default to arm64-v8a
BUILD_TYPE="Release"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --abi)
            ABI="$2"
            shift
            shift
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift
            shift
            ;;
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --abi <abi>        Target ABI (arm64-v8a, x86, x86_64). Default: arm64-v8a"
            echo "  --build-dir <dir>  Build directory. Default: build-android"
            echo "  --debug            Build debug version. Default: Release"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate ABI
VALID_ABIS=("arm64-v8a" "x86" "x86_64")
VALID_ABI=false
for valid_abi in "${VALID_ABIS[@]}"; do
    if [ "$ABI" = "$valid_abi" ]; then
        VALID_ABI=true
        break
    fi
done

if [ "$VALID_ABI" = false ]; then
    echo "ERROR: Invalid ABI specified: $ABI"
    echo "Valid ABIs are: ${VALID_ABIS[*]}"
    exit 1
fi

# Set up Rust environment variables for cross-compilation
if command -v rustc &> /dev/null; then
    # Determine Rust target based on ABI
    if [ "$ABI" = "arm64-v8a" ]; then
        export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang"
    elif [ "$ABI" = "x86" ]; then
        export CARGO_TARGET_I686_LINUX_ANDROID_LINKER="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/i686-linux-android21-clang"
    elif [ "$ABI" = "x86_64" ]; then
        export CARGO_TARGET_X86_64_LINUX_ANDROID_LINKER="$ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/x86_64-linux-android21-clang"
    fi
    
    # Export Android SDK & NDK paths for Rust build scripts
    export ANDROID_HOME=${ANDROID_HOME:-"$HOME/Android/Sdk"}
    export ANDROID_NDK_HOME=$ANDROID_NDK_ROOT
    
    echo "Rust cross-compilation environment:"
    echo "- CARGO_TARGET linker set for $ABI"
    echo "- ANDROID_HOME: $ANDROID_HOME"
    echo "- ANDROID_NDK_HOME: $ANDROID_NDK_HOME"
fi

echo "Build configuration:"
echo "- ABI: $ABI"
echo "- Build type: $BUILD_TYPE"
echo "- Build directory: $BUILD_DIR"
echo "=============================================="

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || { echo "Failed to change to build directory"; exit 1; }

# Configure with CMake
echo "Configuring for Android with ABI: $ABI"
set -x  # Print commands before they're executed
cmake -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
      -DANDROID_ABI="$ABI" \
      -DANDROID_PLATFORM=android-21 \
      -DANDROID_STL=c++_static \
      -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
      -DCMAKE_VERBOSE_MAKEFILE=ON \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -DCMAKE_CXX_FLAGS="-Wno-error=unused-function -Wno-error=unused-local-typedef -Wno-error=for-loop-analysis -Wno-error" \
      -DBUILD_EXAMPLES=ON \
      ..
CMAKE_RESULT=$?
set +x

if [ $CMAKE_RESULT -ne 0 ]; then
    echo "=============================================="
    echo "CMake configuration failed with error code $CMAKE_RESULT"
    echo "Please check the log file: $LOG_FILE"
    exit $CMAKE_RESULT
fi

# Build geniex_core + model examples
echo "=============================================="
echo "Building for Android..."
set -x
cmake --build . -j$(nproc) --verbose
BUILD_RESULT=$?
set +x

if [ $BUILD_RESULT -ne 0 ]; then
    echo "=============================================="
    echo "Build failed with error code $BUILD_RESULT"
    echo "Please check the log file: $LOG_FILE"
    
    echo "Checking for common errors in the build log..."
    grep -E "error:|undefined reference|no such file|cannot find" "$LOG_FILE" > "$LOG_DIR/error_summary_$TIMESTAMP.txt"
    echo "Error summary saved to: $LOG_DIR/error_summary_$TIMESTAMP.txt"
    
    # Check specifically for Rust target issues
    if grep -q "can't find crate for \`core\`" "$LOG_FILE" || grep -q "the \`.*\` target may not be installed" "$LOG_FILE"; then
        echo ""
        echo "RUST TARGET ERROR DETECTED:"
        echo "This error indicates missing Rust targets required for cross-compilation."
        echo "Please run: rustup target add aarch64-linux-android i686-linux-android x86_64-linux-android"
        echo ""
    fi
    
    exit $BUILD_RESULT
fi

echo "=============================================="
echo "Build completed successfully at $(date)"
echo "Build outputs are located in: $BUILD_DIR/bin/"
echo "Full build log saved to: $LOG_FILE"
