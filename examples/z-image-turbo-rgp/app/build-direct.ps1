param(
    [switch]$Rebuild,
    [int]$Jobs = 4
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$makeBinDir = Join-Path $scriptDir ".tools\w64devkit\w64devkit\bin"
$makeExe = Join-Path $makeBinDir "make.exe"
$outputExe = Join-Path $scriptDir "direct-txt2img.exe"

if (!(Test-Path $makeExe)) {
    throw "make.exe not found: $makeExe"
}

if ($Rebuild -and (Test-Path $outputExe)) {
    Remove-Item -LiteralPath $outputExe -Force
}

if (Test-Path $outputExe) {
    Write-Host "Using existing build: $outputExe"
    exit 0
}

Push-Location $scriptDir
try {
    $cmd = 'set PATH=' + $makeBinDir + ';%PATH% && make -j' + $Jobs + ' LLAMA_PORTABLE=1 LLAMA_VULKAN=1 direct-txt2img'
    cmd /c $cmd
}
finally {
    Pop-Location
}

if ($LASTEXITCODE -ne 0) {
    throw "Direct txt2img build failed."
}

if (!(Test-Path $outputExe)) {
    throw "Build finished without producing $outputExe"
}

Write-Host "Build succeeded: $outputExe"
