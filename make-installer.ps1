# Сборка установщика: программа в dist\PhotoStab, затем dist\PhotoStab-Setup-<версия>.exe
# с ней внутри (src/setup.rs). Нужен только Rust; запуск из папки проекта.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$version = if ((Get-Content Cargo.toml -Raw) -match '(?m)^version\s*=\s*"([^"]+)"') { $Matches[1] } else { throw "нет версии в Cargo.toml" }

cargo build --release --bin photostab
if ($LASTEXITCODE) { throw "сборка PhotoStab не удалась" }
New-Item -ItemType Directory -Force dist\PhotoStab | Out-Null
Copy-Item target\release\photostab.exe, target\release\DirectML.dll dist\PhotoStab -Force

cargo build --release --features setup --bin photostab-setup
if ($LASTEXITCODE) { throw "сборка установщика не удалась" }
$setup = "dist\PhotoStab-Setup-$version.exe"
Copy-Item target\release\photostab-setup.exe $setup -Force
Write-Host "Установщик: $((Resolve-Path $setup).Path)"
