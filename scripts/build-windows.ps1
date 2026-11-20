<#
.SYNOPSIS
  Build the MKLight engine DLL and the native Windows app (apps/windows).

.DESCRIPTION
  1. cargo build --release -p mklight-ffi            -> target/release/mklight_ffi.dll
  2. copy it to apps/windows/native/ (git-ignored), where the app and the
     tests pick it up (Directory.Build.props, MKLightNativeDir)
  3. dotnet build of the solution, or dotnet publish of the app to
     build/windows/MKLight (self-contained, Windows App SDK included)

  scripts/build-windows.ps1                 # engine + publish the release app
  scripts/build-windows.ps1 -EngineOnly     # engine DLL only
  scripts/build-windows.ps1 -Configuration Debug   # debug build (no publish)
  scripts/build-windows.ps1 -Test           # also run dotnet test
  scripts/build-windows.ps1 -NoEngine       # skip cargo (DLL already in place)

  See docs/WINDOWS.md "Building".
#>
[CmdletBinding()]
param(
    [ValidateSet('Release', 'Debug')] [string] $Configuration = 'Release',
    [ValidateSet('x64', 'ARM64')] [string] $Platform = 'x64',
    [switch] $EngineOnly,
    [switch] $NoEngine,
    [switch] $Test
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$AppDir = Join-Path $Root 'apps\windows'
$NativeDir = Join-Path $AppDir 'native'

function Invoke-Step([string] $Title, [scriptblock] $Block) {
    Write-Host "==> $Title" -ForegroundColor Cyan
    & $Block
    if ($LASTEXITCODE -ne 0) { throw "$Title failed (exit $LASTEXITCODE)" }
}

if (-not $NoEngine) {
    if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
        throw 'cargo not found: install Rust with rustup (https://rustup.rs)'
    }
    $target = if ($Platform -eq 'ARM64') { 'aarch64-pc-windows-msvc' } else { $null }
    Push-Location $Root
    try {
        if ($target) {
            Invoke-Step "cargo build --release -p mklight-ffi --target $target" { cargo build --release -p mklight-ffi --target $target }
            $dll = Join-Path $Root "target\$target\release\mklight_ffi.dll"
        } else {
            Invoke-Step 'cargo build --release -p mklight-ffi' { cargo build --release -p mklight-ffi }
            $dll = Join-Path $Root 'target\release\mklight_ffi.dll'
        }
    } finally { Pop-Location }
    New-Item -ItemType Directory -Force $NativeDir | Out-Null
    Copy-Item $dll (Join-Path $NativeDir 'mklight_ffi.dll') -Force
    Write-Host "    engine -> $(Join-Path $NativeDir 'mklight_ffi.dll')"
}

if ($EngineOnly) { return }

if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
    throw 'dotnet not found: install the .NET 11 SDK (winget install Microsoft.DotNet.SDK.11)'
}
if (-not (Test-Path (Join-Path $NativeDir 'mklight_ffi.dll'))) {
    Write-Warning 'apps/windows/native/mklight_ffi.dll is missing: the app will report the engine unavailable and the FFI tests will be skipped.'
}

Push-Location $AppDir
try {
    if ($Configuration -eq 'Release') {
        $out = Join-Path $Root 'build\windows\MKLight'
        Invoke-Step "dotnet publish MKLight.App ($Platform) -> $out" {
            dotnet publish src/MKLight.App/MKLight.App.csproj -c Release -p:Platform=$Platform -o $out
        }
        Write-Host "    app -> $(Join-Path $out 'MKLight.exe')"
    } else {
        Invoke-Step "dotnet build MKLight.slnx ($Configuration, $Platform)" {
            dotnet build MKLight.slnx -c $Configuration -p:Platform=$Platform
        }
    }
    if ($Test) {
        Invoke-Step 'dotnet test' { dotnet test --project tests/MKLight.Tests/MKLight.Tests.csproj -c $Configuration }
    }
} finally { Pop-Location }
