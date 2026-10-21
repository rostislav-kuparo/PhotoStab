//! Установщик PhotoStab: один exe, внутри — программа из `dist\PhotoStab` и `install.ps1`.
//! Ставит для текущего пользователя в `%LOCALAPPDATA%\Programs\PhotoStab` (права администратора не
//! нужны): ярлыки на рабочем столе и в меню «Пуск», запись в «Установленных приложениях» — оттуда
//! программа удаляется. Сама установка — тот же `install.ps1`, что и при сборке из исходников.
//! Собирается `make-installer.ps1`.
#![windows_subsystem = "windows"]

use anyhow::{Context, Result, bail};
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use windows_sys::Win32::UI::WindowsAndMessaging::{
    IDOK, IDYES, MB_ICONERROR, MB_ICONINFORMATION, MB_ICONQUESTION, MB_OK, MB_OKCANCEL, MB_YESNO, MessageBoxW,
};

static EXE: &[u8] = include_bytes!("../dist/PhotoStab/photostab.exe");
static DIRECTML: &[u8] = include_bytes!("../dist/PhotoStab/DirectML.dll");
static SCRIPT: &str = include_str!("../install.ps1");
const VERSION: &str = env!("CARGO_PKG_VERSION");
const CREATE_NO_WINDOW: u32 = 0x0800_0000;

fn message(text: &str, flags: u32) -> i32 {
    let wide = |s: &str| s.encode_utf16().chain(Some(0)).collect::<Vec<u16>>();
    unsafe { MessageBoxW(std::ptr::null_mut(), wide(text).as_ptr(), wide("Установка PhotoStab").as_ptr(), flags) }
}

fn main() {
    let Some(dest) = std::env::var_os("LOCALAPPDATA").map(|d| PathBuf::from(d).join("Programs").join("PhotoStab")) else {
        message("Не найдена папка программ пользователя (LOCALAPPDATA).", MB_OK | MB_ICONERROR);
        return;
    };
    let ask = format!(
        "Установить PhotoStab {VERSION}?\n\nПапка: {}\nЯрлыки: рабочий стол и меню «Пуск».\nПрава администратора не нужны. Если PhotoStab уже установлен, он обновится.",
        dest.display()
    );
    if message(&ask, MB_OKCANCEL | MB_ICONQUESTION) != IDOK {
        return;
    }
    match install() {
        Ok(()) => {
            let exe = dest.join("photostab.exe");
            if message("PhotoStab установлен.\n\nОткрыть программу сейчас?", MB_YESNO | MB_ICONINFORMATION) == IDYES {
                let _ = Command::new(&exe).current_dir(&dest).spawn();
            }
        }
        Err(e) => {
            message(&format!("Не удалось установить PhotoStab.\n\n{e:#}"), MB_OK | MB_ICONERROR);
        }
    }
}

/// Распаковка во временную папку и запуск `install.ps1` из неё.
fn install() -> Result<()> {
    let tmp = std::env::temp_dir().join(format!("PhotoStab-setup-{}", std::process::id()));
    let r = run_script(&tmp);
    let _ = std::fs::remove_dir_all(&tmp);
    r
}

fn run_script(tmp: &Path) -> Result<()> {
    std::fs::create_dir_all(tmp).context("временная папка")?;
    std::fs::write(tmp.join("photostab.exe"), EXE).context("распаковка photostab.exe")?;
    std::fs::write(tmp.join("DirectML.dll"), DIRECTML).context("распаковка DirectML.dll")?;
    // Скрипт с BOM: без него Windows PowerShell 5 читает русский текст как ANSI.
    let script = tmp.join("install.ps1");
    std::fs::write(&script, format!("\u{feff}{}", SCRIPT.trim_start_matches('\u{feff}'))).context("распаковка install.ps1")?;
    let out = Command::new("powershell.exe")
        .args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File"])
        .arg(&script)
        .arg("-Source")
        .arg(tmp)
        .args(["-Version", VERSION])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .context("не запускается PowerShell")?;
    if !out.status.success() {
        let err = String::from_utf8_lossy(&out.stderr);
        let text = if err.trim().is_empty() { String::from_utf8_lossy(&out.stdout) } else { err };
        bail!("{}", text.trim());
    }
    Ok(())
}
