# Установка PhotoStab для текущего пользователя: программа в %LOCALAPPDATA%\Programs\PhotoStab,
# ярлыки на рабочем столе и в меню «Пуск» (по нему приложение находится в поиске Windows).
# Повторный запуск обновляет установленную версию. Права администратора не нужны. Программа
# появляется в «Установленных приложениях» (Параметры → Приложения), оттуда же удаляется.
param(
    [string]$Source = (Join-Path $PSScriptRoot "dist\PhotoStab"),
    [string]$Version = ""
)

$ErrorActionPreference = "Stop"
# Установщик читает вывод через канал — сообщения отдаём в UTF-8, чтобы русский текст не бился.
if ([Console]::IsOutputRedirected) { [Console]::OutputEncoding = [Text.Encoding]::UTF8 }
if (-not $Version) {
    $toml = Join-Path $PSScriptRoot "Cargo.toml"
    $Version = if ((Test-Path $toml) -and ((Get-Content $toml -Raw) -match '(?m)^version\s*=\s*"([^"]+)"')) { $Matches[1] } else { "dev" }
}
$dest = Join-Path $env:LOCALAPPDATA "Programs\PhotoStab"
if (-not (Test-Path (Join-Path $Source "photostab.exe"))) {
    throw "Не найдена сборка: $Source\photostab.exe"
}
New-Item -ItemType Directory -Force $dest | Out-Null
# Открытую программу не закрываем: занятый файл Windows разрешает переименовать, новая версия
# ложится рядом и запустится со следующего открытия. Старые копии убираем, когда они свободны.
Get-ChildItem $dest -Filter "*.old*" -ErrorAction SilentlyContinue | ForEach-Object {
    try { Remove-Item $_.FullName -Force -ErrorAction Stop } catch {}
}
$running = Get-Process photostab -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$dest*" }
foreach ($name in "photostab.exe", "DirectML.dll") {
    $from = Join-Path $Source $name
    $to = Join-Path $dest $name
    if ((Test-Path $to) -and (Get-FileHash $from).Hash -eq (Get-FileHash $to).Hash) { continue }
    if ($running -and (Test-Path $to)) {
        Rename-Item $to ("$name.old" + (Get-Date -Format "yyyyMMddHHmmss")) -Force
    }
    Copy-Item $from $to -Force
}

$exe = Join-Path $dest "photostab.exe"
$shell = New-Object -ComObject WScript.Shell
foreach ($dir in @([Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs"))) {
    $lnk = $shell.CreateShortcut((Join-Path $dir "PhotoStab.lnk"))
    $lnk.TargetPath = $exe
    $lnk.WorkingDirectory = $dest
    $lnk.IconLocation = "$exe,0"
    $lnk.Description = "PhotoStab — единая проявка RAW-съёмки с мероприятий"
    $lnk.Save()
}

# Удаление: ярлыки, запись в «Установленных приложениях» и папка программы. Правки съёмок
# (photostab.json рядом со снимками) и история открытых папок остаются.
$uninstall = Join-Path $dest "uninstall.ps1"
Set-Content -Path $uninstall -Encoding UTF8 -Value @'
# Удаление PhotoStab. Правки съёмок (photostab.json рядом со снимками) и история остаются.
$dest = $PSScriptRoot
Get-Process photostab -ErrorAction SilentlyContinue | Where-Object { $_.Path -like "$dest*" } | Stop-Process -Force
foreach ($dir in @([Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs"))) {
    Remove-Item (Join-Path $dir "PhotoStab.lnk") -Force -ErrorAction SilentlyContinue
}
Remove-Item "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\PhotoStab" -Recurse -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 500
Remove-Item $dest -Recurse -Force -ErrorAction SilentlyContinue
'@
$key = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\PhotoStab"
New-Item -Path $key -Force | Out-Null
$strings = @{
    DisplayName     = "PhotoStab"
    DisplayVersion  = $Version
    Publisher       = "Kuparo"
    DisplayIcon     = "$exe,0"
    InstallLocation = $dest
    UninstallString = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$uninstall`""
}
foreach ($k in $strings.Keys) { New-ItemProperty -Path $key -Name $k -Value $strings[$k] -PropertyType String -Force | Out-Null }
foreach ($k in "NoModify", "NoRepair") { New-ItemProperty -Path $key -Name $k -Value 1 -PropertyType DWord -Force | Out-Null }
$sizeKb = [int]((Get-ChildItem $dest -File | Measure-Object Length -Sum).Sum / 1KB)
New-ItemProperty -Path $key -Name EstimatedSize -Value $sizeKb -PropertyType DWord -Force | Out-Null

Write-Host "PhotoStab $Version установлен: $exe"
Write-Host "Ярлыки: рабочий стол и меню «Пуск» (поиск Windows: PhotoStab)."
if ($running) {
    Write-Host "PhotoStab сейчас открыт — новая версия заработает после перезапуска программы."
}