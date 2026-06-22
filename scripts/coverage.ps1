<#
.SYNOPSIS
    Generate an HTML source-coverage report for the geniex-qairt CPU unit tests.

.DESCRIPTION
    Configures a clang-cl coverage build, builds the instrumented unit-test
    targets, runs them, and renders an llvm-cov HTML report scoped to first-party
    core/ sources. This is the local full-codebase view; CI runs the PR diff gate
    (scripts/diff_coverage.py) instead.

    Requires a Visual Studio install with the bundled LLVM (clang-cl, llvm-cov,
    llvm-profdata) under VC\Tools\Llvm\ARM64\bin.

.PARAMETER BuildDir
    Coverage build directory (default: build-coverage).

.PARAMETER Open
    Open the HTML report in the default browser when done.

.EXAMPLE
    pwsh scripts/coverage.ps1 -Open
#>
[CmdletBinding()]
param(
    [string]$BuildDir = "build-coverage",
    [switch]$Open
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

# ── Locate the VS-bundled LLVM toolchain ─────────────────────────────────────
$llvmBin = Get-ChildItem `
    "C:\Program Files\Microsoft Visual Studio\*\*\VC\Tools\Llvm\ARM64\bin" `
    -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $llvmBin) {
    throw "Could not find VC\Tools\Llvm\ARM64\bin. Install the 'C++ Clang tools for Windows' VS component."
}
$llvmCov     = Join-Path $llvmBin.FullName "llvm-cov.exe"
$llvmProfdata = Join-Path $llvmBin.FullName "llvm-profdata.exe"

# ── Coverage surface: keep in sync with scripts/coverage_common.py ───────────
$Targets       = @("utils_test", "graph_test", "input_provider_test")
$IncludeRegex  = "[\\/]core[\\/](src|include)[\\/]"
$IgnoreRegex   = @(
    "[\\/]tests[\\/]",
    "[\\/]third-party[\\/]",
    "[\\/]qnn-api[\\/]",
    "[\\/]_deps[\\/]",
    "[\\/]googletest",
    "[\\/]logging\.(h|cpp)$"
) -join "|"

# ── Configure + build the instrumented tests ────────────────────────────────
Write-Host "==> Configuring clang-cl coverage build in $BuildDir" -ForegroundColor Cyan
cmake -B $BuildDir -A ARM64 -T ClangCL `
    -DCMAKE_BUILD_TYPE=Debug `
    -DGENIEX_BUILD_TESTS=ON `
    -DGENIEX_COVERAGE=ON
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }

Write-Host "==> Building instrumented test targets" -ForegroundColor Cyan
cmake --build $BuildDir --config Debug -j --target $Targets
if ($LASTEXITCODE -ne 0) { throw "Build failed" }

# ── Run the tests, collecting one .profraw per target ────────────────────────
$profDir = Join-Path $BuildDir "coverage"
Remove-Item $profDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $profDir | Out-Null

$exes = @()
foreach ($t in $Targets) {
    $exe = Join-Path $BuildDir "bin\Debug\$t.exe"
    if (-not (Test-Path $exe)) { throw "Missing test exe: $exe" }
    $exes += $exe
    # %p expands to the PID so parallel/multiple runs never clobber each other.
    $env:LLVM_PROFILE_FILE = (Join-Path $profDir "$t-%p.profraw")
    Write-Host "==> Running $t" -ForegroundColor Cyan
    & $exe
    if ($LASTEXITCODE -ne 0) { throw "$t failed (exit $LASTEXITCODE)" }
}
Remove-Item Env:\LLVM_PROFILE_FILE -ErrorAction SilentlyContinue

# ── Merge + render ───────────────────────────────────────────────────────────
$profData = Join-Path $profDir "merged.profdata"
$raws = (Get-ChildItem (Join-Path $profDir "*.profraw")).FullName
& $llvmProfdata merge -sparse $raws -o $profData
if ($LASTEXITCODE -ne 0) { throw "llvm-profdata merge failed" }

# Multiple binaries: first is the object, the rest are -object args.
$objectArgs = @($exes[0])
foreach ($e in $exes[1..($exes.Count - 1)]) { $objectArgs += @("-object", $e) }

$htmlDir = Join-Path $profDir "html"
$commonArgs = @(
    "-instr-profile=$profData",
    "-ignore-filename-regex=$IgnoreRegex",
    "--sources", "core"
)

Write-Host "==> Rendering HTML report" -ForegroundColor Cyan
& $llvmCov show @objectArgs @commonArgs `
    -format=html -output-dir=$htmlDir -show-branches=count -show-line-counts-or-regions
if ($LASTEXITCODE -ne 0) { throw "llvm-cov show failed" }

# Machine-readable export for the diff-coverage gate (scripts/diff_coverage.py).
# -skip-functions keeps the JSON small; the gate only needs file segments.
$exportJson = Join-Path $profDir "export.json"
$json = & $llvmCov export @objectArgs @commonArgs -skip-functions | Out-String
[System.IO.File]::WriteAllText($exportJson, $json)
Write-Host "Export JSON: $exportJson" -ForegroundColor Green

Write-Host "`n==> Coverage summary (first-party core/ sources)" -ForegroundColor Green
& $llvmCov report @objectArgs @commonArgs

$index = Join-Path $htmlDir "index.html"
Write-Host "`nHTML report: $index" -ForegroundColor Green
if ($Open) { Start-Process $index }
